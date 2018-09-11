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

vector<Point3d> myCloud;

struct CloudPoint {
	Point3d pt;
	vector<int> imgpt_for_img;
	double reprojection_error;
};

//decalre functions:
void processImages(vector<Mat> &images, vector<Mat> &imagesColored, 
	char* dirName, bool show);
Mat_<double> LinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2);
Mat_<double> IterativeLinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2);
double TriangulatePoints(const vector<KeyPoint>& keypoint_img1, 
	const vector<KeyPoint>& keypoint_img2,
	const Mat& K, const Mat& Kinv, 
	const Mat& distanceCoeffs, 
	const Matx34d& P1, const Matx34d& P2,
	vector<CloudPoint>& pointCloud,
	vector<KeyPoint>& correspondingImg1Pt, bool show);
bool triangulateBetweenViews(const Matx34d& P1, const Matx34d& P2, 
	vector<CloudPoint>& tri_pts, map<pair<int, int>, vector<DMatch>>& all_matches,
	const vector<KeyPoint>& pts1_good, const vector<KeyPoint>& pts2_good,
	const Mat& K, Mat& distanceCoeffs, vector<int>& add_to_cloud,
	vector<KeyPoint>& correspondingImg1Pt, bool show, vector<CloudPoint>& global_pcloud, 
	int image_size, int idx1, int idx2);
bool DecomposeEssentialMat(Mat_<double>& E, Mat_<double>& R1, Mat_<double>& R2,
	Mat_<double>& t1, bool show); 
bool checkRotationMat(Mat_<double>& R1);
bool testTriangulation(const vector<CloudPoint>& pointCloud, const Matx34d& P,
	vector<uchar>& status, bool show);
void transformCloudPoints(vector<Point3d>& points3d, vector<CloudPoint>& cloudPoints);
bool findP2Matrix(Matx34d& P1, Matx34d& P2, const Mat& K, const Mat& distanceCoeffs,
	vector<KeyPoint>& keypoint_img1, vector<KeyPoint>& keypoint_img2,
	Mat_<double> R1, Mat_<double> R2, Mat_<double> t1, bool show);
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
bool computeMatches(int idx1, int idx2, Mat& img1, Mat& img2, Mat& desc1, Mat& desc2, 
	vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<DMatch>& matches, 
	vector<KeyPoint>& good_keypts, bool show);
bool computeSFM(vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, 
	map<pair<int, int>, vector<DMatch>>& all_matches, Mat& K, Mat& distanceCoeffs, 
	vector<KeyPoint>& kpts_good1, vector<KeyPoint>& kpts_good2,
	vector<CloudPoint>& global_pcloud, map<int, Matx34d>& all_pmats, 
	int idx1, int idx2, int image_size, bool show);
void getPointRGB(vector<CloudPoint>& global_pcloud, vector<Vec3b>& RGBCloud,
	vector<Mat>& imagesColored, vector<vector<KeyPoint>>& all_keypoints,
	int image_size);
void displayCloud(vector<CloudPoint>& global_pcloud, vector<Vec3b>& RGBCloud, 
	vector<Point3d>& all_points);
void adjustBundle(vector<CloudPoint>& global_pcloud, Mat& cam_matrix,
	const vector<vector<KeyPoint>>& all_keypoints, map<int, Matx34d>& all_pmats, bool show);
int get2DMeasurements(const vector<CloudPoint>& global_pcloud);
void Find2D3DCorrespondences(int curr_view, vector<Point3f>& cloud, 
	vector<Point2f>& imgPoints, vector<CloudPoint>& global_pcloud,
	set<int>& good_views, map<pair<int, int>, vector<DMatch>>& all_matches,
	vector<vector<KeyPoint>>& all_keypoints);
bool estimatePose(int curr_view, Mat_<double>& rvec, Mat_<double>& t, Mat_<double>& R,
	vector<Point3f>& cloud, vector<Point2f>& imgPoints, Mat& K, Mat& distanceCoeffs);
void sortMatchesFromHomography(map<pair<int, int>, vector<DMatch>>& matches,
	vector<vector<KeyPoint>>& keypoints, list<pair<int,pair<int,int>>>& percent_matches,
	bool show);
bool sortFromPercentage(pair<int, pair<int, int>> a, pair<int, pair<int, int>> b);
void findFeatures(vector<vector<KeyPoint>>& all_keypoints, vector<Mat>& all_descriptors,
	Mat& img, int idx);

#endif