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
	vector<vector<KeyPoint>> all_keypoints;

	if (argc != 2) {
		cout << "wrong number of arguments" << endl;
		return -1;
	}
	processImages(images, argv[1]);

	//using sift, we extract the key points in the image.
	int minHessian = 400;
	int index = 0;
	//this is our surf keypoints detector
	Ptr<SURF> surf = SURF::create(minHessian);
	vector<Mat> descriptors;
	/**
	* This loop finds the keypoints and features on every single image.
	* We store the new image in a new folder called features
	*/
	cout << "calculating the keypoints for the images.." << endl;
	for(auto it = images.begin(); it != images.end(); ++it) {
		vector<KeyPoint> keypoints;
		Mat descriptor;
		surf->detectAndCompute(*it, Mat(), keypoints, descriptor);
		all_keypoints.push_back(keypoints);
		descriptors.push_back(descriptor);
	}

	//begin the matching process! ..

	FlannBasedMatcher matcher;
	vector<DMatch> matches;

	Mat descriptor_img1 = descriptors.at(1);
	Mat descriptor_img2 = descriptors.at(2);

	matcher.match(descriptor_img1, descriptor_img2, matches);
	cout << "size of matches: " << matches.size() << endl;
	double min_dist = 0;
	double max_dist = 200;

	for (int i = 0; i < descriptor_img1.rows; ++i) {
		double dist = matches[i].distance;
		if (dist < min_dist) {
			min_dist = dist;
		} 
		if (dist > max_dist) {
			max_dist = dist;
		} 
	}

	vector<DMatch> good_matches;

	for (int i = 0; i < descriptor_img1.rows; ++i) {
		if( matches[i].distance <= max(2 * min_dist, 0.05)) {
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(images[1], all_keypoints[1], images[2], all_keypoints[2],
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cout << "good matches size: " << good_matches.size() << endl;

	vector<Point2f> img1_points;
	vector<Point2f> img2_points;

	for (int i = 0; i < good_matches.size(); ++i) {
		img1_points.push_back(all_keypoints[1][good_matches[i].queryIdx].pt);
		img2_points.push_back(all_keypoints[2][good_matches[i].trainIdx].pt);
	}

	Mat H;
	if(img1_points.size() != 0 && img2_points.size() != 0) {
		cout << "inside the if" << endl;
		H = findHomography(img1_points, img2_points, CV_RANSAC);
	}
	
	//get the corners from the images.
	vector<Point2f> corners(4);
	corners[0] = cvPoint(0, 0);
	corners[1] = cvPoint(images[1].cols, 0);
	corners[2] = cvPoint(images[1].cols, images[1].rows);
	corners[3] = cvPoint(0, images[1].rows);
	vector<Point2f> corners2(4);

	if (!H.empty()) {
		cout << "inside the second if" << endl;
		perspectiveTransform(corners, corners2, H);

		line(img_matches, corners2[0] + Point2f(images[1].cols, 0), corners2[1] + Point2f(images[1].cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, corners2[1] + Point2f(images[1].cols, 0), corners2[2] + Point2f(images[1].cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, corners2[2] + Point2f(images[1].cols, 0), corners2[3] + Point2f(images[1].cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, corners2[3] + Point2f(images[1].cols, 0), corners2[0] + Point2f(images[1].cols, 0), Scalar(0, 255, 0), 4);

		imwrite("test.jpg", img_matches);

	}
	
	// //if we have odd number of images have a triple.
	// int odd = descriptors.size() % 2;

	// //for each pair of images match the features
	// for (auto it = descriptors.begin(); it != descriptors.end(); ++it) {
	// 	if (!odd) {
	// 		matcher.match(*it, *(++it), matches);
	// 	} else {
	// 		odd = 0;
	// 	}
	// }

	// //now we find the "good" matches that are close to each other.
	// for (auto it = descriptors.begin(); it != descriptors.end(); ++it) {
	// 	double min_dist = 0;
	// 	double max_dist = 100;
	// 	cout << "calculating the min/max distance" << endl;
	// 	for (int i = 0; i < (*it).rows; ++i) {
	// 		double dist = matches[i].distance;
	// 		if (dist < min_dist) {
	// 			min_dist = dist;
	// 		}
	// 		if (dist > max_dist) {
	// 			max_dist = dist;
	// 		}
	// 	}

	// 	vector<DMatch> good_matches;
	// 	cout << "finding good matches.." << endl;
	// 	for (int i = 0; i < (*it).rows; ++i) {
	// 		if (matches[i].distance < (3 * min_dist)) {
	// 			good_matches.push_back(matches[i]);
	// 		}
	// 	}
	// 	Mat img_matches;
	// 	drawMatches(images[index], all_keypoints[index], images[index + 1], all_keypoints[index + 1],
	// 				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	// 				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// 	imwrite("./matchedimages" + to_string(index) + ".jpg", img_matches);

	// 	vector<Point2f> img1;
	// 	vector<Point2f> img2;

	// 	for (int i = 0; i < good_matches.size(); ++i) {
	// 		img1.push_back(all_keypoints[index][good_matches[i].queryIdx].pt);
	// 		img2.push_back(all_keypoints[index + 1][good_matches[i].trainIdx].pt);

	// 	}
	// 	Mat H = findHomography(img1, img2, CV_RANSAC);

	// 	++index;
	// }

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