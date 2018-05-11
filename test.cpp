#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void processImages(vector<Mat> &images, char* dirName);

int main(int argc, char** argv) {

	//vector used to hold the images
	vector<Mat> images;

	if (argc != 2) {
		cout << "wrong number of arguments" << endl;
		return -1;
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

	//find the Homography Matrix
	Mat H = findHomography(img1_obj, img2_obj, CV_RANSAC);

	//find the fundamentalMatrix
	Mat E = findFundamentalMat(img1_obj, img2_obj, FM_RANSAC, 3, 0.99);

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