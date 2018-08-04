#ifndef THREE_D_RECONSTRUCTION_HPP
#define THREE_D_RECONSTRUCTION_HPP

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include "CameraCalib.hpp"

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
	vector<CloudPoint>& tri_pts, vector<DMatch>& good_matches,
	const vector<KeyPoint>& pts1_good, const vector<KeyPoint>& pts2_good,
	const Mat& K, const Mat& Kinv, const Mat& distanceCoeffs, 
	vector<KeyPoint>& correspImg1Pt, vector<int>& add_to_cloud,
	bool show, vector<CloudPoint>& global_pcloud, int image_size);
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
<<<<<<< Updated upstream
void findFeatures(vector<Mat>& images, vector<vector<KeyPoint>>& keypoints, 
	vector<Mat>& descriptors);
void matchFeatures(vector<vector<KeyPoint>>& keypoints, vector<vector<KeyPoint>>& keypoints_good,
	vector<Mat>& images, map<pair<int, int>, vector<DMatch>>& matches, 
	vector<Mat>& descriptors, bool show);
void reverseMatches(const vector<DMatch>& matches, vector<DMatch>& reverse);
void filterMatches(vector<KeyPoint>& keypts1, vector<KeyPoint>& keypts2, 
	vector<DMatch>& ijMatches, vector<KeyPoint>& keypts1_good,
	vector<KeyPoint>& keypts2_good, Mat& img1, Mat& img2, int i, int j, bool show);
void sortMatchesFromHomography(map<pair<int, int>, vector<DMatch>>& matches,
	vector<vector<KeyPoint>>& keypoints, list<pair<int,pair<int,int>>>& percent_matches,
	bool show);
bool sortFromPercentage(pair<int, pair<int, int>> a, pair<int, pair<int, int>> b);
void pruneMatchesBasedOnF(vector<vector<KeyPoint>>& keypoints, 
	vector<vector<KeyPoint>>& keypoints_good, vector<Mat>& images,
	map<pair<int, int>, vector<DMatch>>& matches, bool show);
Mat findF(const vector<KeyPoint>& keypts1, const vector<KeyPoint>& keypts2, 
	vector<KeyPoint>& keypts1_good, vector<KeyPoint>& keypts2_good,
	vector<DMatch>& matches, bool show);
=======
boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
boost::shared_ptr<pcl::visualization::PCLVisualizer> grayVis (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
void showCloudNoColor(const pcl::PointCloud<pcl::PointXYZ>::Ptr& finalPC);
void getRGBCloudPoint(const vector<CloudPoint>& global_pcloud, vector<Vec3b>& out,
	vector<KeyPoint>& keypoint_img1, vector<KeyPoint>& keypoint_img2,
	Mat& img1_c, Mat& img2_c);
void populatePC(vector<Point3d>& global_pcloud_3d, vector<Vec3b>& RGBPoints,
	const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& finalPC);
void showCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& finalPC);
void populatePCNoColor(vector<Point3d>& global_pcloud_3d, 
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& finalPC);
bool computeSFM(Mat& img1, Mat& img2, Mat& K, Mat& desc1, Mat& desc2, 
	vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<Point3d>& cloud, 
	Mat& distanceCoeffs, bool show, int image_size);

>>>>>>> Stashed changes
#endif