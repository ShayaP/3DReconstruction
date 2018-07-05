#ifndef CAMERA_CALIB_HPP
#define CAMERA_CALIB_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <dirent.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class CameraCalib {
public: 
	CameraCalib(string fileName, Mat& K, Mat& distanceCoeffs) {
		readCameraInfo(fileName, K, distanceCoeffs);
	}
	CameraCalib(char* dirName, Mat& K, Mat& distanceCoeffs, 
		vector<Mat>& rVectors, vector<Mat>& tVectors, bool show) {
		ManualCalibrateCamera(dirName, K, distanceCoeffs, rVectors, tVectors, show);
	}
private:
	//declare the calibration board consts.
	const float calibSquareDim = 0.032f;
	const Size chessdim = Size(9, 6);
	const int FLAGS = CV_CALIB_FIX_PRINCIPAL_POINT +
		CV_CALIB_FIX_ASPECT_RATIO +
		CV_CALIB_ZERO_TANGENT_DIST +
		CV_CALIB_FIX_K4 +
		CV_CALIB_FIX_K5;

	void ManualCalibrateCamera(char* dirName, Mat& K, Mat& distanceCoeffs, 
		vector<Mat>& rVectors, vector<Mat>& tVectors, bool show) {
		cout << "entering ManualCalibrateCamera" << endl;

		//get the calibration images.
		vector<Mat> calibImages;
		processImages(calibImages, dirName);
		cout << "calib images processed" << endl;

		//calibrate the camera using these images.
		vector<vector<Point2f>> imageSpacePoints;
		getChessboardCorners(calibImages, imageSpacePoints, show);

		vector<vector<Point3f>> worldCornerPoints(1);
		createKnownBoardPosition(chessdim, calibSquareDim, worldCornerPoints[0]);
		worldCornerPoints.resize(imageSpacePoints.size(), worldCornerPoints[0]);
		cout << "starting calibration" << endl;
		double error = calibrateCamera(worldCornerPoints, imageSpacePoints, chessdim, K, 
			distanceCoeffs, rVectors, tVectors, FLAGS);
		cout << "calibration ended with error: " << error << endl;
		printCalibToFile(K, distanceCoeffs, error);
	}
	void createKnownBoardPosition (Size boardSize, float squareEdgeLength, vector<Point3f>& conrners) {
		cout << "creating createKnownBoardPosition" << endl;
		for (int i = 0; i < boardSize.height; ++i) {
			for (int j = 0; j < boardSize.width; ++j) {
				conrners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
			}
		}
	}

	void getChessboardCorners (vector<Mat> images, vector<vector<Point2f>>& allFoundCorners,
		bool draw) {
		int index = 1;
		cout << "entering getChessboardCorners" << endl;
 		for (auto it = images.begin(); it != images.end(); ++it) {
			vector<Point2f> pointBuffer;
			int flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
			bool res = findChessboardCorners(*it, Size(9, 6), pointBuffer, flags);
			if (res) {
				cout << "corner for image " << index << " found successfully" << endl;

				allFoundCorners.push_back(pointBuffer);
				if (draw) {
					drawChessboardCorners(*it, chessdim, pointBuffer, res);
					string name = "board" + to_string(index) + ".jpg";
					imwrite(name, *it);
				}
			} else {
				cout << "failed to find corners of image: " <<  index << endl;
			}
			++index;
		}
	}

	void printCalibToFile(Mat& K, Mat& distanceCoeffs, double error) {
		cout << "priting camera info to file .." << endl;
		FileStorage fs("calibInfo.yml", FileStorage::WRITE);
		fs << "K" << K;
		fs << "distanceCoeffs" << distanceCoeffs;
		fs << "error" << error;
		fs.release(); 
	}

	void readCameraInfo(string fileName, Mat& K, Mat& distanceCoeffs) {
		FileStorage fs(fileName, FileStorage::READ);
		fs["K"] >> K;
		fs["distanceCoeffs"] >> distanceCoeffs;
		fs.release();
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
};

#endif