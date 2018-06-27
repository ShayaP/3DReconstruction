#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::detail;

const float calibSquareDim = 0.032f;
const Size chessdim = Size(6, 9);

void processImages(vector<Mat> &images, char* dirName);
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& conrners);
void getChessboardCorners (vector<Mat> images, vector<vector<Point2f>>& allFoundCorners);
void ManualCalibrateCamera(char* dirName, Mat K, Mat distanceCoeffs, 
	vector<Mat> rVectors, verctor<Mat> tVectors);

int main(int argc, char** argv) {

	//vector used to hold the images
	vector<Mat> images;

	if (argc != 2) {
		cout << "wrong number of arguments" << endl;
		return -1;
	}
	Mat K;
	vector<Mat> rVectors, tVectors;
	Mat distanceCoeffs = Mat::zeros(8, 1, CV_64F);

	ManualCalibrateCamera(argv[2], K, distanceCoeffs, rVectors, tVectors);
	processImages(images, argv[1]);
	Mat img1 = images[7];
	Mat img2 = images[8];

	//using sift, we extract the key points in the image.
	int minHessian = 400;

	//this is our surf keypoints detector
	Ptr<SURF> surf = SURF::create(minHessian);
	Mat descriptor_img1, descriptor_img2;
	vector<KeyPoint> keypoint_img1, keypoint_img2;

	surf->detectAndCompute(img1, Mat(), keypoint_img1, descriptor_img1);
	surf->detectAndCompute(img2, Mat(), keypoint_img2, descriptor_img2);

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptor_img1, descriptor_img2, matches);

	cout << "found : " << matches.size() << " matches" << endl;

	Mat img_matches;
	drawMatches(img1, keypoint_img1, img2, keypoint_img2, matches, img_matches,
				Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("matches.jpg", img_matches);

	vector<Point2f> img1_obj;
	vector<Point2f> img2_obj;

	for (int i = 0; i < matches.size(); ++i) {
		img1_obj.push_back(keypoint_img1[matches[i].queryIdx].pt);
		img2_obj.push_back(keypoint_img2[matches[i].trainIdx].pt);
	}

	//find the fundamentalMatrix
	Mat F = findFundamentalMat(img1_obj, img2_obj, FM_RANSAC, 0.1, 0.99);

	//using the 8-point algorithm we find the Essential Matrix
	//and the Camera parameters.
	Mat H1(4, 4, img1.type());
	Mat H2(4, 4, img1.type());

	stereoRectifyUncalibrated(img1_obj, img2_obj, F, img1.size(), H1, H2);

	Mat rectified1;
	Mat rectified2;
	warpPerspective(img1, rectified1, H1, img1.size());
	warpPerspective(img2, rectified2, H2, img2.size());
	imwrite("rectified1.jpg", rectified1);
	imwrite("rectified2.jpg", rectified2);

	return 0;
}

/**
* this method processes the images in a directory and adds them to the list
*/
void processImages(vector<Mat> &images, char* dirName) {
	//access the image folder, and read each image into images.
	DIR *dir;
	struct dirent *entity;
	if (dir=opendir(dirName)) {
		while(entity=readdir(dir)) {
			if (entity == NULL) {
				cout << "Could not read the directory." << endl;
				break;
			} else if (((string)entity->d_name).find(".jpg")) {
				//insert the images into the vector
				string path;
				path.append(dirName);
				path.append("/");
				path.append(entity->d_name);
				Mat image = imread(path, IMREAD_GRAYSCALE);
				if (!image.data) {
					cout << "could not read image: " << entity->d_name << endl;
				} else {
					images.push_back(image);
				}
			}
		}
	}
	closedir(dir);
}

void createKnownBoardPosition (Size boardSize, float squareEdgeLength, vector<Point3f>& conrners) {
	for (int i = 0; i < boardSize.height; ++i) {
		for (int j = 0; j < boardSize.width; ++j) {
			conrners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

void getChessboardCorners (vector<Mat> images, vector<vector<Point2f>>& allFoundCorners) {
	for (auto it = images.begin(); it != images.end(); ++it) {
		vector<Point2f> pointBuffer;
		int flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
		bool res = findChessboardCorners(*it, Size(9, 6), pointBuffer, flags);
		if (res) {
			allFoundCorners.push_back(pointBuffer);
		}
	}
}

void ManualCalibrateCamera(char* dirName, Mat K, Mat distanceCoeffs, 
	vector<Mat> rVectors, verctor<Mat> tVectors) {

	//get the calibration images.
	vector<Mat> calibImages;
	processImages(calibImages, dirName);

	//calibrate the camera using these images.
	vector<vector<Point2f>> imageSpacePoints;
	getChessboardCorners(calibImages, imageSpacePoints);

	vector<vector<Point3f>> worldCornerPoints(1);
	createKnownBoardPosition(boardSize, squareEdgeLength, worldCornerPoints[0]);
	worldCornerPoints.resize(imageSpacePoints.size(), worldCornerPoints[0]);
	calibrateCamera(worldCornerPoints, imageSpacePoints, boardSize, K, distanceCoeffs, rVectors, tVectors);

}


//TODO: implement an autocalibration method for the camera intrinsics.

// Mat autoCalibrateCamera(vector<Mat> &images) {
// 	cout << "starting AutoCalibration: " << endl;
// 	Mat K;

// 	vector<Mat> homographies;
// 	Mat refImage = images[0];

// 	//find the homography between every pair of images.
// 	for (auto it = images.begin(); it != images.end(); ++it) {
// 		nextImg  = *it;

// 		//using sift, we extract the key points in the image.
// 		int minHessian = 400;

// 		//this is our surf keypoints detector
// 		Ptr<SURF> surf = SURF::create(minHessian);
// 		Mat descriptor_img1, descriptor_img2;
// 		vector<KeyPoint> keypoint_img1, keypoint_img2;

// 		surf->detectAndCompute(refImage, Mat(), keypoint_img1, descriptor_img1);
// 		surf->detectAndCompute(nextImg, Mat(), keypoint_img2, descriptor_img2);

// 		BFMatcher matcher(NORM_L2);
// 		vector<DMatch> matches;
// 		matcher.match(descriptor_img1, descriptor_img2, matches);

// 		vector<Point2f> img1_obj;
// 		vector<Point2f> img2_obj;

// 		for (int i = 0; i < matches.size(); ++i) {
// 			img1_obj.push_back(keypoint_img1[matches[i].queryIdx].pt);
// 			img2_obj.push_back(keypoint_img2[matches[i].trainIdx].pt);
// 		}

// 		//find the Homography between the 2 images.
// 		Mat H = findHomography(img1_obj, img2_obj);		
// 		homographies.push_back(H);
// 	}
// 	cout << "--found the homographies between all images." << endl;

	
// 	return K;
// }