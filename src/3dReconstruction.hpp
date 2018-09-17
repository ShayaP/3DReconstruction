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
vector<Mat> all_descriptors;
vector<Point3DInMap> globalPoints;

map<int, Matx34d> all_pmats;
map<pair<int, int>, vector<KeyPoint>> all_good_keypoints;
map<pair<int, int>, vector<DMatch>> all_matches;
map<int, pair<vector<Point2f>, vector<Point3f>>> imageCorrespondences;
set<int> done_views;
set<int> good_views;

int first_view = 0, second_view = 0;

///Rotational element in a 3x4 matrix
const cv::Rect ROT(0, 0, 3, 3);

///Translational element in a 3x4 matrix
const cv::Rect TRA(3, 0, 1, 3);

//decalre functions:
void processImages(char* dirName);
bool triangulateBetweenViews(const Matx34d& P1, const Matx34d& P2,
	vector<Point3DInMap>& cloud, int idx1, int idx2);
bool checkRotationMat(Mat_<double>& R1);
void useage();
void allignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
	const vector<DMatch>& good_matches, vector<KeyPoint>& new_pts1,
	vector<KeyPoint>& new_pts2);
void showCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& finalPC);
bool computeMatches(int idx1, int idx2);
bool computeSFM(int idx1, int idx2, vector<Point3DInMap>& cloud);
void getPointRGB(vector<Vec3b>& RGBCloud);
void displayCloud(vector<Vec3b>& RGBCloud);
void adjustBundle(Mat& cam_matrix);
void Find2D3DCorrespondences();
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
void mergeClouds(const vector<Point3DInMap> cloud);
void autoCalibrate();
void saveCloudToPLY();

#endif