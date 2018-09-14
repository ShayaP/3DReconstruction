#ifndef THREE_D_RECONSTRUCTION_HPP
#define THREE_D_RECONSTRUCTION_HPP

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <fstream>
#include <stdlib.h>

// include files from PCL to show the cloud
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>

// include files from opencv to compute sfm
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching.hpp"

#include "CameraCalib.hpp"


struct CloudPoint {
	Point3d pt;
	vector<int> imgpt_for_img;
	double reprojection_error;
};


struct Point3DInMap {
    // 3D point.
    cv::Point3f p;

    // A mapping from image index to 2D point index in that image's list of features.
    std::map<int, int> originatingViews;
};

//boolean for extra debug info.
bool show;

//camera matricies.
Mat K;
vector<Mat> rVectors, tVectors;
Mat distanceCoeffs = Mat::zeros(8, 1, CV_64F);

//data structures used to hold the images and data
vector<Mat> images;
vector<Mat> imagesColored;
vector<vector<KeyPoint>> all_keypoints;
vector<Point3d> all_points;
vector<Mat> all_descriptors;
vector<CloudPoint> global_pcloud;
vector<Point3DInMap> globalPoints;

map<int, Matx34d> all_pmats;
map<pair<int, int>, vector<KeyPoint>> all_good_keypoints;
map<pair<int, int>, vector<DMatch>> all_matches;
set<int> done_views;
set<int> good_views;

int first_view = 0, second_view = 0;

///Rotational element in a 3x4 matrix
const cv::Rect ROT(0, 0, 3, 3);

///Translational element in a 3x4 matrix
const cv::Rect TRA(3, 0, 1, 3);

//decalre functions:
void processImages(char* dirName);
Mat_<double> LinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2);
Mat_<double> IterativeLinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2);
double TriangulatePoints(const vector<KeyPoint>& kpts_good, const Matx34d& P1, 
	const Matx34d& P2, vector<CloudPoint>& pointCloud,	
	vector<KeyPoint>& correspondingImg1Pt);
bool triangulateBetweenViews(vector<CloudPoint>& tri_pts, vector<int>& add_to_cloud,
	vector<KeyPoint>& correspondingImg1Pt, const Matx34d& P1, const Matx34d& P2,
	int idx1, int idx2);
// bool DecomposeEssentialMat(Mat_<double>& E, Mat_<double>& R1, Mat_<double>& R2,
// 	Mat_<double>& t1, bool show); 
bool checkRotationMat(Mat_<double>& R1);
bool testTriangulation(const vector<CloudPoint>& pointCloud, const Matx34d& P,
	vector<uchar>& status);
void transformCloudPoints(vector<Point3d>& points3d, vector<CloudPoint>& cloudPoints);
// bool findP2Matrix(Matx34d& P1, Matx34d& P2, const Mat& K, const Mat& distanceCoeffs,
// 	vector<KeyPoint>& keypoint_img1, vector<KeyPoint>& keypoint_img2,
// 	Mat_<double> R1, Mat_<double> R2, Mat_<double> t1, bool show);
void useage();
void allignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
	const vector<DMatch>& good_matches, vector<KeyPoint>& new_pts1,
	vector<KeyPoint>& new_pts2);
void showCloudNoColor(const pcl::PointCloud<pcl::PointXYZ>::Ptr& finalPC);
void getRGBCloudPoint(const vector<CloudPoint>& global_pcloud, vector<Vec3b>& out,
	vector<KeyPoint>& keypoint_img1, vector<KeyPoint>& keypoint_img2,
	Mat& img1_c, Mat& img2_c);
void populatePC(vector<Point3d>& global_pcloud_3d, vector<Vec3b>& RGBPoints,
	const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& finalPC);
void showCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& finalPC);
void populatePCNoColor(vector<Point3d>& global_pcloud_3d, 
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& finalPC);
bool computeMatches(int idx1, int idx2);
bool computeSFM(int idx1, int idx2);
void getPointRGB(vector<Vec3b>& RGBCloud);
void displayCloud(vector<Vec3b>& RGBCloud);
void adjustBundle(Mat& cam_matrix);
int get2DMeasurements(const vector<CloudPoint>& global_pcloud);
void Find2D3DCorrespondences(int curr_view, vector<Point3f>& cloud, 
	vector<Point2f>& imgPoints);
bool estimatePose(int curr_view, Mat_<double>& rvec, Mat_<double>& t, Mat_<double>& R,
	vector<Point3f>& cloud, vector<Point2f>& imgPoints);
void sortMatchesFromHomography(list<pair<int,pair<int,int>>>& percent_matches);
bool sortFromPercentage(pair<int, pair<int, int>> a, pair<int, pair<int, int>> b);
void findFeatures(Mat& img, int idx);
void allignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
	const vector<DMatch>& good_matches, vector<KeyPoint>& new_pts1,
	vector<KeyPoint>& new_pts2, vector<int>& leftBackRefrence, 
	vector<int>& rightBackRefrence, vector<Point2f>& points1, vector<Point2f>& points2);
void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps); 
void triangulate2Views(int first_view, int second_view);
void flipMatches(const vector<DMatch>& matches, vector<DMatch>& flipedMatches);
bool estimatePose(vector<Point3f>& points3d, vector<Point2f>& points2d, Matx34d& Pnew);
void addMoreViewsToReconstruction();
int get2DMeasurements(const vector<Point3DInMap>& globalPoints);

#endif