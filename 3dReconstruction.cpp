#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <fstream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "CameraCalib.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::detail;

//decalre functions:
void processImages(vector<Mat> &images, char* dirName);

int main(int argc, char** argv) {

	//vector used to hold the images
	vector<Mat> images;
	bool show;

	//camera matricies.
	Mat K;
	vector<Mat> rVectors, tVectors;
	Mat distanceCoeffs = Mat::zeros(8, 1, CV_64F);

	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		return -1;
	} else if (argc == 4) {
		if (*(argv[3]) == 1) {
			show = true;
		} else {
			show = false;
		}
		CameraCalib cc(argv[2], K, distanceCoeffs, rVectors, tVectors, show);
	} else if (argc == 3) {
		//read in calibration info from file.
		//in the future maybe autocalibrate
	}

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
	//compute the essential matrix.
	Mat E = K.t() * F * K;
	SVD decomp = SVD(E, SVD::MODIFY_A);

	//decomposition of E matricies.
	Mat U = decomp.u;
	Mat vt = decomp.vt;
	Mat w = decomp.w;
	Matx33d W(0, -1, 0,
		1, 0, 0, 
		0, 0, 1);
	Matx33d Wt(0, 1, 0,
		-1, 0, 0, 
		0, 0, 1);
	Mat R1 = U * Mat(W) * vt;
	Mat R2 = U * Mat(Wt) * vt;
	Mat t1 = U.col(2);
	Mat t2 = -U.col(2);
	//this is the first of the 4 possible matricies.
	P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0), R1(2,1), R1(2,2), t1(2));
	//now test to see which of the 4 P1s are good.

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