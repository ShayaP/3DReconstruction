#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <set>
#include <string>
#include <map>
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
bool DecomposeEssentialMat(Mat_<double>& E, Mat_<double>& R1, Mat_<double>& R2,
	Mat_<double>& t1, Mat_<double>& t2); 
bool checkRotationMat(Mat_<double>& R1);
bool testTriangulation(const vector<CloudPoint>& pointCloud, const Matx34d& P,
	vector<uchar>& status);
void transformCloudPoints(vector<Point3d>& points3d, vector<CloudPoint>& cloudPoints);
bool findP2Matrix(Matx34d& P1, Matx34d& P2, const Mat& K, const Mat& distanceCoeffs,
	vector<KeyPoint>& keypoint_img1, vector<KeyPoint>& keypoint_img2,
	Mat_<double> R1, Mat_<double> R2, Mat_<double> t1, Mat_<double> t2);


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
		if (atoi(argv[3]) == 1) {
			show = true;
		} else {
			show = false;
		}
		CameraCalib cc(argv[2], K, distanceCoeffs, rVectors, tVectors, show);
	} else if (argc == 2) {
		//read in calibration info from file.
		CameraCalib cc("calibInfo.yml", K, distanceCoeffs);
	}

	processImages(images, argv[1]);
	Mat img1 = images[0];
	Mat img2 = images[1];

	//this is our sift keypoints detector
	int min_hessian = 400;
	Ptr<Feature2D> surf = SURF::create(min_hessian);
	Mat descriptor_img1, descriptor_img2;
	vector<KeyPoint> keypoint_img1, keypoint_img2;
	cout << "<================== Begin Matching =============>\n" << endl;

	surf->detect(img1, keypoint_img1);
	surf->detect(img2, keypoint_img2);
	surf->compute(img1, keypoint_img1, descriptor_img1);
	surf->compute(img2, keypoint_img2, descriptor_img2);
	cout << "done detecting and computing features" << endl;

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;
	matcher.match(descriptor_img1, descriptor_img2, matches);

	cout << "found : " << matches.size() << " matches" << endl;

	//filter the matches
	cout << "\n<================== filtering matches =============>\n" << endl;

	set<int> existing_trainIdx;
	vector<DMatch> good_matches;
	vector<KeyPoint> pts1, pts2;

	for (unsigned int i = 0; i < matches.size(); ++i) {
		if (matches[i].trainIdx <= 0) {
			matches[i].trainIdx  = matches[i].imgIdx;
		}

		if (existing_trainIdx.find(matches[i].trainIdx) == existing_trainIdx.end() && 
			matches[i].trainIdx >= 0 && matches[i].trainIdx < (int)(keypoint_img2.size())) {
			good_matches.push_back(matches[i]);
			pts1.push_back(keypoint_img1[matches[i].queryIdx]);
			pts2.push_back(keypoint_img2[matches[i].trainIdx]);
			existing_trainIdx.insert(matches[i].trainIdx);
		}
	}

	vector<uchar> status;
	vector<KeyPoint> pts1_good, pts2_good;

	// now we allign the matched points.
	vector<KeyPoint> pts1_temp, pts2_temp;
	for (int i = 0; i < good_matches.size(); ++i) {
		assert(good_matches[i].queryIdx < keypoint_img1.size());
		pts1_temp.push_back(keypoint_img1[good_matches[i].queryIdx]);
		assert(good_matches[i].trainIdx < keypoint_img2.size());
		pts2_temp.push_back(keypoint_img2[good_matches[i].trainIdx]);
	}
	vector<Point2f> points1, points2;
	for (unsigned int i = 0; i < good_matches.size(); ++i) {
		points1.push_back(pts1_temp[i].pt);
		points2.push_back(pts2_temp[i].pt);
	}

	double min, max;
	minMaxIdx(points1, &min, &max);
	Mat F = findFundamentalMat(points1, points2, FM_RANSAC, 0.006 * max, 0.99, status);
	vector<DMatch> new_matches;
	for (unsigned int i = 0; i < status.size(); ++i) {
		if (status[i]) {
			pts1_good.push_back(pts1_temp[i]);
			pts2_good.push_back(pts2_temp[i]);

			new_matches.push_back(good_matches[i]);
		}
	}
	cout << "after fund matrix matches: " << new_matches.size() << endl;
	good_matches = new_matches;
	points1.clear();
	points2.clear();
	for (unsigned int i = 0; i < good_matches.size(); ++i) {
		points1.push_back(pts1_temp[i].pt);
		points2.push_back(pts2_temp[i].pt);
	}

	//draw the matches found.
	Mat img_matches;
	drawMatches(img1, keypoint_img1, img2, keypoint_img2, good_matches, img_matches,
				Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("matches.jpg", img_matches);
	cout << "image saved" << endl;

	cout << "\n<================== Decomposing E =============>\n" << endl;
	//find the fundamnetal matrix
	F = findFundamentalMat(points1, points2, FM_RANSAC, 0.006 * max, 0.99);

	//compute the essential matrix.
	Mat_<double> E = K.t() * F * K;
	Mat_<double> R1, R2, t1, t2;
	Mat u, vt, w;

	//decompose the essential matrix
	DecomposeEssentialMat(E, R1, R2, t1, t2);

	//check if the decomposition was successful.
	if (determinant(R1) + 1.0 < 1e-09) {
		E = -E;
		decomposeEssentialMat(E, R1, R2, t1);
	}
	if (!checkRotationMat(R1)) {
		cout << "Rotation Matrix is not correct" << endl;
		return -1;
	}
	if (determinant(E) > 1e-05) {
		cout << "Essential Matrix determinant must be 0, but was: " << endl;
		cout << determinant(E) << endl;
		return -1;
	}
	cout << "decomposition was successful" << endl;

	//now we find our camera matricies P and P1.
	//we assume that P = [I|0]
	cout << "\n<================== Searching for P2 =============>\n" << endl;
	Matx34d P1(1, 0, 0, 0,
		0, 1, 0, 0, 
		0, 0, 1, 0);
	Matx34d P2;
	//now test to see which of the 4 P2s are good.
	bool foundCameraMat = findP2Matrix(P1, P2, K, distanceCoeffs, pts1_good, 
		pts2_good, R1 , R2, t1, t2);
	if (!foundCameraMat) {
		cout << "p2 was not found successfully" << endl;
	} else {
		cout << "p2 found successfully" << endl;
	}

	return 0;
}

/**
* This function decomposes the Essential matrix E using the SVD class.
* Returns: Rotation Matricies (R1, R2) and Translation Matricies (t1, t2).
*/
bool DecomposeEssentialMat(Mat_<double>& E, Mat_<double>& R1, Mat_<double>& R2,
	Mat_<double>& t1, Mat_<double>& t2) {
	SVD decomp = SVD(E, SVD::MODIFY_A);

	//decomposition of E.
	Mat U = decomp.u;
	Mat vt = decomp.vt;
	Mat w = decomp.w;

	//check to see if first and second singular values are the same.
	double svr = fabsf(w.at<double>(0) / w.at<double>(1));
	if (svr > 1.0) {
		svr = 1.0 / svr;
	}
	if (svr < 0.7) {
		cout << "singular values are too far apart" << endl;
		return false;
	} else if (w.at<double>(2) > 1e-09) {
		cout << "final singluar value should be 0" << endl;
		return -1;
	}

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
	return true;
}

/**
* This is a helper function for that computes X from AX = B.
*/
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

/**
* This function solves the equation AX = B for the point X.
*/
Mat_<double> IterativeLinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2) {
	int num_iterations = 10;
	float EPSILON = 0.0001;

	//cout << "starting IterativeLinearLSTriangulation.. " << endl;
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

/**
* This function computes a point cloud from given key points in the 2 images
* and given camera matricies.
*/
double TriangulatePoints(const vector<KeyPoint>& pts1_good, 
	const vector<KeyPoint>& pts2_good,
	const Mat& K, const Mat& Kinv, 
	const Mat& distanceCoeffs, 
	const Matx34d& P1, const Matx34d& P2,
	vector<CloudPoint>& pointCloud,
	vector<KeyPoint>& correspondingImg1Pt) {

	cout << "starting TriangulatePoints.. " << endl;
	vector<double> reprojectionError;
	int size = pts1_good.size();
	correspondingImg1Pt.clear();

	Matx44d P2_(P2(0,0),P2(0,1),P2(0,2),P2(0,3),
				P2(1,0),P2(1,1),P2(1,2),P2(1,3),
				P2(2,0),P2(2,1),P2(2,2),P2(2,3),
				0,		0,		0,		1);
	Matx44d P2inv(P2_.inv());

	Mat_<double> KP2 = K * Mat(P2);

	//triangulate every single feature
	for (int i = 0; i < size; ++i) {
		//point from image 1
		Point2f kp1 = pts1_good[i].pt;
		Point3d u1(kp1.x, kp1.y, 1.0);

		//point from image 2
		Point2f kp2 = pts2_good[i].pt;
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
		correspondingImg1Pt.push_back(pts1_good[i]);
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
		//TODO implement.
	}
}

/**
* This function tests to see if a given rotation matrix is valid.
*/
bool checkRotationMat(Mat_<double>& R1) {
	if (fabsf(determinant(R1)) - 1.0 > 1e-07) {
		cout << "Rotation Matrix is not valid" << endl;
		return false;
	} else {
		return true;
	}
}

/**
* This function tests to see how many points with a certain Camera P lie
* infront of the camera. if more than 75% of the points lie infront of the camera,
* this test is a success.
*/
bool testTriangulation(vector<CloudPoint>& pointCloud, const Matx34d& P,
	vector<uchar>& status) {
	status.clear();
	vector<Point3d> pcloud_pt3d;
	transformCloudPoints(pcloud_pt3d, pointCloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());

	Matx44d P4x4 = Matx44d::eye();
	for (int i = 0; i < 12; ++i) {
		P4x4.val[i] = P.val[i];
	}

	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	status.resize(pointCloud.size(), 0);
	for (int i = 0; i < pointCloud.size(); ++i) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pointCloud.size());
	cout << "percentage of points infront of camera: " << percentage << endl;
	if (percentage <  0.75) {
		return false;
	} else {
		return true;
	}
}

/**
* This function transforms cloudpoints into 3d points.
*/
void transformCloudPoints(vector<Point3d>& points3d, vector<CloudPoint>& cloudPoints) {
	points3d.clear();
	for (auto it = cloudPoints.begin(); it != cloudPoints.end(); ++it) {
		points3d.push_back((*it).pt);
	}
}

/**
* This function tests all 3 different possible P2 matricies and picks one or none.
*/
bool findP2Matrix(Matx34d& P1, Matx34d& P2, const Mat& K, const Mat& distanceCoeffs,
	vector<KeyPoint>& pts1_good, vector<KeyPoint>& pts2_good,
	Mat_<double> R1, Mat_<double> R2, Mat_<double> t1, Mat_<double> t2) {
	P2 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0), R1(2,1), R1(2,2), t1(2));

	vector<CloudPoint> pointCloud1;
	vector<CloudPoint> pointCloud2;
	vector<KeyPoint> correspondingImg1pts;
	Mat Kinv = K.inv();

	cout << "starting test for P2, first configuration: \n" << endl;
	double reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
		distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts);
	double reproj_err2 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
		distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts);

	vector<uchar> temp_status;

	if (!testTriangulation(pointCloud1, P2, temp_status) 
		|| !testTriangulation(pointCloud2, P1, temp_status) || reproj_err1 > 100.0
		|| reproj_err2 > 100.0) {
		//try a new P2
		P2 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
			R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
			R1(2,0), R1(2,1), R1(2,2), t2(2));
		cout << "\nstarting test for P2, second configuration: \n" << endl;
		pointCloud1.clear();
		pointCloud2.clear();
		correspondingImg1pts.clear();

		reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
			distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts);
		reproj_err2 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
			distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts);

		if (!testTriangulation(pointCloud1, P2, temp_status) 
			|| !testTriangulation(pointCloud2, P1, temp_status) || reproj_err1 > 100.0
			|| reproj_err2 > 100.0) {
			if (!checkRotationMat(R2)) {
				cout << "R2 was not valid" << endl;
				P2 = 0;
				return false;
			}

			//try another P2
			P2 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
					R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
					R2(2,0), R2(2,1), R2(2,2), t1(2));
			cout << "\nstarting test for P2, thrid configuration: \n" << endl;

			pointCloud1.clear();
			pointCloud2.clear();
			correspondingImg1pts.clear();

			reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
				distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts);
			reproj_err2 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
				distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts);

			if (!testTriangulation(pointCloud1, P2, temp_status) 
				|| !testTriangulation(pointCloud2, P1, temp_status) || reproj_err1 > 100.0
				|| reproj_err2 > 100.0) {

				//try the last P2
				P2 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
						R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
						R2(2,0), R2(2,1), R2(2,2), t2(2));
				cout << "\nstarting test for last P2 configuration: \n" << endl;
				pointCloud1.clear();
				pointCloud2.clear();
				correspondingImg1pts.clear();

				reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
					distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts);
				reproj_err2 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
					distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts);
				if (!testTriangulation(pointCloud1, P2, temp_status) 
					|| !testTriangulation(pointCloud2, P1, temp_status) || reproj_err1 > 100.0
					|| reproj_err2 > 100.0) {
					cout << "\nerr: could not find a good P2.." << endl;
					return false;
				}
			}
		}
	}
}
//TODO: implement an autocalibration method for the camera intrinsics.