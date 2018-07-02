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

struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};

//decalre functions:
void processImages(vector<Mat> &images, char* dirName);
Mat_<double> LinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2);
Mat_<double> IterativeLinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2);
double TriangulatePoints(const vector<KeyPoint>& keypoint_img1, 
	const vector<KeyPoint>& keypoint_img2,
	const Mat& K, const Mat& Kinv, 
	const Mat& distanceCoeffs, 
	const Matx34d& P1, const Matx34d& P2,
	vector<CloudPoint>& pointCloud,
	vector<KeyPoint>& correspondingImg1Pt);
void calibrateFromFile(Mat& K, string fileName);

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
		string fileName = argv[2];
		calibrateFromFile(K, fileName);
	} else if (argc == 2) {
		//auto calibrate
	}

	processImages(images, argv[1]);
	Mat img1 = images[0];
	Mat img2 = images[1];

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
	
	//TODO: prune matches.

	//find the fundamentalMatrix
	Mat F = findFundamentalMat(img1_obj, img2_obj, FM_RANSAC, 0.1, 0.99);
	//compute the essential matrix.
	Mat_<double> E = K.t() * F * K;
	SVD decomp = SVD(E, SVD::MODIFY_A);

	//decomposition of E.
	Mat U = decomp.u;
	Mat vt = decomp.vt;
	Mat w = decomp.w;
	Mat_<double> R1;
	Mat_<double> R2;
	Mat_<double> t1;
	Mat_<double> t2;

	Matx33d W(0, -1, 0,
		1, 0, 0, 
		0, 0, 1);
	Matx33d Wt(0, 1, 0,
		-1, 0, 0, 
		0, 0, 1);
	R1 = U * Mat(W) * vt;
	R2 = U * Mat(Wt) * vt;
	t1 = U.col(2);
	t2 = -U.col(2);

	//now we find our camera matricies P and P1.
	//we assume that P = [I|0]
	Matx34d P1(1, 0, 0, 0,
		0, 1, 0, 0, 
		0, 0, 1, 0);
	//this is the first of the 4 possible matricies for P1.
	Matx34d P2(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0), R1(2,1), R1(2,2), t1(2));
	//now test to see which of the 4 P1s are good.

	return 0;
}

Mat_<double> LinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2) {
	Matx43d A(u1.x*P1(2,0)-P1(0,0),	u1.x*P1(2,1)-P1(0,1),		u1.x*P1(2,2)-P1(0,2),		
			u1.y*P1(2,0)-P1(1,0),	u1.y*P1(2,1)-P1(1,1),		u1.y*P1(2,2)-P1(1,2),		
			u2.x*P2(2,0)-P2(0,0), u2.x*P2(2,1)-P2(0,1),	u2.x*P2(2,2)-P2(0,2),	
			u2.y*P2(2,0)-P2(1,0), u2.y*P2(2,1)-P2(1,1),	u2.y*P2(2,2)-P2(1,2));
	Matx41d B(-(u1.x*P1(2,3)	-P1(0,3)),
			-(u1.y*P1(2,3)	-P1(1,3)),
			-(u2.x*P2(2,3)	-P2(0,3)),
			-(u2.y*P2(2,3) -P2(1,3)));

	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);

	return X;
}

Mat_<double> IterativeLinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2) {
	int num_iterations = 10;
	float EPSILON = 0.0001;

	cout << "starting IterativeLinearLSTriangulation.. " << endl;
	double wi1 = 1;
	double wi2 = 1;
	Mat_<double> X(4, 1);
	Mat_<double> X_ = LinearLSTriangulation(u1, P1, u2, P2);
	X(0) = X_(0);
	X(1) = X_(1);
	X(2) = X_(2);
	X(3) = 1.0;


	for (int i = 0; i < num_iterations; ++i) {
		//find the weights
		double weight1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
		double weight2 = Mat_<double>(Mat_<double>(P2).row(2)*X)(0);

		//if diff is less than epsilon, break.
		if (fabsf(wi1 - weight1) <= EPSILON && fabsf(wi2 - weight2) <= EPSILON) {
			break;
		}

		wi1 = weight1;
		wi2 = weight2;

		//resolve with new weights.
		Matx43d A((u1.x*P1(2,0)-P1(0,0))/wi1,		(u1.x*P1(2,1)-P1(0,1))/wi1,			(u1.x*P1(2,2)-P1(0,2))/wi1,		
				  (u1.y*P1(2,0)-P1(1,0))/wi1,		(u1.y*P1(2,1)-P1(1,1))/wi1,			(u1.y*P1(2,2)-P1(1,2))/wi1,		
				  (u2.x*P2(2,0)-P2(0,0))/wi2,	(u2.x*P2(2,1)-P2(0,1))/wi2,		(u2.x*P2(2,2)-P2(0,2))/wi2,	
				  (u2.y*P2(2,0)-P2(1,0))/wi2,	(u2.y*P2(2,1)-P2(1,1))/wi2,		(u2.y*P2(2,2)-P2(1,2))/wi2
				  );
		Mat_<double> B = (Mat_<double>(4,1) <<	  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1,
												  -(u2.x*P2(2,3)	-P2(0,3))/wi2,
												  -(u2.y*P2(2,3)	-P2(1,3))/wi2);

		solve(A, B, X_, DECOMP_SVD);
		X(0) = X_(0);
		X(1) = X_(1);
		X(2) = X_(2);
		X(3) = 1.0;
	}
	return X;
}

double TriangulatePoints(const vector<KeyPoint>& keypoint_img1, 
	const vector<KeyPoint>& keypoint_img2,
	const Mat& K, const Mat& Kinv, 
	const Mat& distanceCoeffs, 
	const Matx34d& P1, const Matx34d& P2,
	vector<CloudPoint>& pointCloud,
	vector<KeyPoint>& correspondingImg1Pt) {

	cout << "starting TriangulatePoints.. " << endl;
	correspondingImg1Pt.clear();

	Matx44d P2_(P2(0,0),P2(0,1),P2(0,2),P2(0,3),
				P2(1,0),P2(1,1),P2(1,2),P2(1,3),
				P2(2,0),P2(2,1),P2(2,2),P2(2,3),
				0,		0,		0,		1);
	Matx44d P2inv(P2_.inv());

	vector<double> reprojectionError;
	int size = keypoint_img1.size();

	Mat_<double> KP2 = K * Mat(P2);

	//triangulate every single feature
	for (int i = 0; i < size; ++i) {
		//point from image 1
		Point2f kp1 = keypoint_img1[i].pt;
		Point3d u1(kp1.x, kp1.y, 1.0);

		//point from image 2
		Point2f kp2 = keypoint_img2[i].pt;
		Point3d u2(kp2.x, kp2.y, 1.0);

		Mat_<double> um1 = Kinv * Mat_<double>(u1);
		Mat_<double> um2 = Kinv * Mat_<double>(u2);

		u1.x = um1(0);
		u1.y = um1(1);
		u1.z = um1(2);

		u2.x = um2(0);
		u2.y = um2(1);
		u2.z = um2(2);

		//find the estimate from Linear LS Triangulation
		Mat_<double> X = IterativeLinearLSTriangulation(u1, P1, u2, P2);

		//reproject the point again and find the reprojection error
		Mat_<double> xPt_img = KP2 * X;
		Point2f xPt_img_(xPt_img(0)/xPt_img(2), xPt_img(1)/xPt_img(2));

		//calculate the cloud point and find the error
		double err = norm(xPt_img_ - kp2);
		reprojectionError.push_back(err);
		CloudPoint cp;
		cp.pt = Point3d(X(0), X(1), X(2));
		cp.reprojection_error = err;
		pointCloud.push_back(cp);
		correspondingImg1Pt.push_back(keypoint_img1[i]);
	} 
	Scalar mse = mean(reprojectionError);
	cout << "finished triangulation with: " << mse[0] << " error" << endl;
	return mse[0];
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

void calibrateFromFile(Mat& K, string fileName) {
	ifstream infile;
	string line;
	vector<float> temp;
	infile.open(fileName);
	if (infile.fail()) {
		cout << "failed to open the calibration file." << endl;
		return;
	} else {
		int i = 0;
		int j = 0;
		while (getline(infile, line)) {
			if (line == "K:") {
				temp.clear();
				while (i < 9 && infile >> temp[i]) {
					++i;
				}
			} else if (line == "D:") {
				temp.clear();
				while (j < 9)
			} 
		}
	}
}
//TODO: implement an autocalibration method for the camera intrinsics.