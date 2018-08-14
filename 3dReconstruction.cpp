#include "3dReconstruction.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {

	//data structures used to hold the images and data
	vector<Mat> images;
	vector<Mat> imagesColored;
	vector<vector<KeyPoint>> all_keypoints;
	map<pair<int, int>, vector<KeyPoint>> all_good_keypoints;
	vector<Point3d> all_points;
	vector<Mat> all_descriptors;
	map<pair<int, int>, vector<DMatch>> all_matches;
	vector<CloudPoint> global_pcloud;

	bool show;

	//camera matricies.
	Mat K;
	vector<Mat> rVectors, tVectors;
	Mat distanceCoeffs = Mat::zeros(8, 1, CV_64F);

	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		useage();
		return -1;
	} else if (argc == 4) {
		if (atoi(argv[3]) == 1) {
			show = true;
		} else {
			show = false;
		}
		CameraCalib cc(argv[2], K, distanceCoeffs, rVectors, tVectors, show);
	} else if (argc == 3) {
		if (atoi(argv[2]) == 1) {
			show = true;
		} else {
			show = false;
		}
		CameraCalib cc("../calibInfo.yml", K, distanceCoeffs);
	} else if (argc == 2) {
		//read in calibration info from file.
		CameraCalib cc("../calibInfo.yml", K, distanceCoeffs);
		show = false;
	}

	//process the images and save them in their data structures
	processImages(images, imagesColored, argv[1], show);
	all_keypoints.resize(images.size(), vector<KeyPoint>());
	all_descriptors.resize(images.size(), Mat());

	//detect maching keypoints for different images.
	for (int i = 0; i < images.size() - 1; ++i) {
		for (int j = i + 1; j < images.size(); ++j) {
			bool matches_res = computeMatches(images[i], images[j], all_descriptors[i],
				all_descriptors[j], all_keypoints[i], all_keypoints[j],
				all_matches[make_pair(i, j)], all_good_keypoints[make_pair(i, j)],
				show);
			if (matches_res) {
				cout << "successful matching with image: [" << i << ", " << j << "]" << endl;
				//cout << "current cloud size: " << all_points.size() << "\n\n" << endl;
			} else {
				cout << "failed matching with image: [" << i << ", " << j << "]\n\n" << endl;
			}
		}
	}
	cout << "\nStarting SFM from the matches......" << endl;
	//compute sfm with the given matches.
	for (int i = 0; i < images.size() - 1; ++i) {
		for (int j = i + 1; j < images.size(); ++j) {
			bool sfm_res = computeSFM(all_keypoints[i], all_keypoints[j], 
				all_matches, K, distanceCoeffs, all_good_keypoints[make_pair(i, j)],
				global_pcloud, i, j, images.size(), show);
			if (sfm_res) {
				cout << "successful sfm with image: [" << i << ", " << j << "]" << endl;
				//cout << "current cloud size: " << all_points.size() << "\n\n" << endl;
			} else {
				cout << "failed sfm with image: [" << i << ", " << j << "]\n\n" << endl;
			}
		}
	}
	//find the RGB colors for the points that were found.
	vector<Vec3b> RGBCloud;
	getPointRGB(global_pcloud, RGBCloud, imagesColored, all_keypoints, images.size());
	if (RGBCloud.size() != global_pcloud.size()) {
		cout << "error: color values are not the same size as the points" << endl;
		return -1;
	}

	//convert the found cloudpoints into points and colors and write them to a file.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (int i = 0; i < global_pcloud.size(); ++i) {
		Vec3b rgbv(255, 255, 255);
		if (RGBCloud.size() > i) {
			rgbv = RGBCloud[i];
		}
		Point3d p = global_pcloud[i].pt;
		if (isnan(p.x) || isnan(p.y) || isnan(p.z)) {
			continue;
		}

		all_points.push_back(p);
		pcl::PointXYZRGB pt;
		pt.x = p.x;
		pt.y = p.y;
		pt.z = p.z;
		uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
		pt.rgb = *reinterpret_cast<float*>(&rgb);
		cloudPtr->push_back(pt);
	}
	// cloud.width = cloud.points.size();
	// cloud.height = 1;
	//pcl::io::savePCDFileASCII("points.pcd", cloud);
	pcl::visualization::CloudViewer viewer("SFM");
	viewer.showCloud(cloudPtr);
	while(!viewer.wasStopped()) {

	}

	return 0;
}

void getPointRGB(vector<CloudPoint>& global_pcloud, vector<Vec3b>& RGBCloud,
	vector<Mat>& imagesColored, vector<vector<KeyPoint>>& all_keypoints,
	int image_size) {
	RGBCloud.resize(global_pcloud.size());
	for (int i = 0; i < global_pcloud.size(); ++i) {
		vector<Vec3b> point_colors;
		for (int j = 0; j < image_size; ++j) {
			if (global_pcloud[i].imgpt_for_img[j] != -1) {
				int pt_idx = global_pcloud[i].imgpt_for_img[j];
				if (pt_idx >= all_keypoints[j].size()) {
					cout << "err: index out of bounds while getting point RGB" << endl;
					continue;
				}
				Point p = all_keypoints[j][pt_idx].pt;
				point_colors.push_back(imagesColored[j].at<Vec3b>(p));
			}
		}
		Scalar res_color = mean(point_colors);
		RGBCloud[i] = (Vec3b(res_color[0], res_color[1], res_color[2]));
	}
}

bool computeSFM(vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, 
	map<pair<int, int>, vector<DMatch>>& all_matches, Mat& K, Mat& distanceCoeffs, 
	vector<KeyPoint>& kpts_good, vector<CloudPoint>& global_pcloud,
	int idx1, int idx2, int image_size, bool show) {
	cout << "\nSFM with images: [" << idx1 << ", " << idx2 << "]" << endl;

	vector<KeyPoint> pts1_temp, pts2_temp;
	vector<DMatch> matches = all_matches[make_pair(idx1, idx2)];
	cout << "matches size: " << matches.size() << endl;
	allignPoints(kpts1, kpts2, matches, pts1_temp, pts2_temp);
	vector<Point2f> points1, points2;
	for (unsigned int i = 0; i < matches.size(); ++i) {
		points1.push_back(pts1_temp[i].pt);
		points2.push_back(pts2_temp[i].pt);
	}
	double min, max;
	minMaxIdx(points1, &min, &max);
	//find the fundamnetal matrix
	Mat F = findFundamentalMat(points1, points2, FM_RANSAC, 0.006 * max, 0.99);
	

	//compute the essential matrix.
	Mat_<double> E = K.t() * F * K;
	if (show) {
		cout << "E: \n" << endl;
		cout << E << "\n" << endl;
		cout << "F: \n" << endl;
		cout << F << "\n" << endl;
		cout << "K: \n" << endl;
		cout << K << "\n" << endl;
	}
	Mat_<double> R1, R2, t1;
	Mat u, vt, w;

	//decompose the essential matrix
	bool decomp_res = DecomposeEssentialMat(E, R1, R2, t1, show);

	//check if the decomposition was successful.
	if (!decomp_res) {
		return false;
	}
	if (determinant(R1) + 1.0 < 1e-09) {
		E = -E;
		DecomposeEssentialMat(E, R1, R2, t1, show);
	}
	if (!checkRotationMat(R1)) {
		cout << "Rotation Matrix is not correct" << endl;
		return false;
	}
	if (determinant(E) > 1e-05) {
		cout << "Essential Matrix determinant must be 0, but was: " << endl;
		cout << determinant(E) << endl;
		return false;
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
	bool foundCameraMat = findP2Matrix(P1, P2, K, distanceCoeffs, kpts_good, 
		kpts_good, R1 , R2, t1, show);
	if (!foundCameraMat) {
		cout << "p2 was not found successfully" << endl;
		return false;
	} else {
		cout << "p2 found successfully" << endl;

		//triangulate the points and create point cloud.
		vector<CloudPoint> tri_pts;
		vector<int> add_to_cloud;
		vector<KeyPoint> correspondingImg1Pt;

		bool tri_res = triangulateBetweenViews(P1, P2, tri_pts, all_matches, kpts_good, kpts_good,
			K, distanceCoeffs, correspondingImg1Pt, add_to_cloud, show, global_pcloud,
			image_size, idx1, idx2);
		if (tri_res && countNonZero(add_to_cloud) >= 10) {
			cout << "before triangulation: " << global_pcloud.size() << endl;
			for (int j = 0; j < add_to_cloud.size(); ++j) {
				if (add_to_cloud[j] == 1) {
					global_pcloud.push_back(tri_pts[j]);
				}
			}
			cout << "after triangulation: " << global_pcloud.size() << endl;
		} else {
			cout << "triangulation failed or not enough points were added" << endl;
			return false;
		}
	}
}

bool computeMatches(Mat& img1, Mat& img2, Mat& desc1, Mat& desc2, 
	vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<DMatch>& matches, 
	vector<KeyPoint>& good_keypts, bool show) {


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
	cout << "features for image 1: " << keypoint_img1.size() << endl;
	cout << "features for image 2: " << keypoint_img2.size() << endl;
	cout << "done detecting and computing features" << endl;

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches_;
	matcher.match(descriptor_img1, descriptor_img2, matches_);
	desc1 = descriptor_img1;
	desc2 = descriptor_img2;
	kpts1 = keypoint_img1;
	kpts2 = keypoint_img2;
	matches = matches_;


	if (show) {
		cout << "found : " << matches.size() << " matches" << endl;
	}

	//filter the matches
	cout << "\n<================== filtering matches =============>\n" << endl;

	set<int> existing_trainIdx;
	vector<DMatch> good_matches;

	for (unsigned int i = 0; i < matches_.size(); ++i) {
		if (matches_[i].trainIdx <= 0) {
			matches_[i].trainIdx  = matches_[i].imgIdx;
		}

		if (existing_trainIdx.find(matches_[i].trainIdx) == existing_trainIdx.end() && 
			matches_[i].trainIdx >= 0 && matches_[i].trainIdx < (int)(keypoint_img2.size())) {
			good_matches.push_back(matches_[i]);
			existing_trainIdx.insert(matches_[i].trainIdx);
		}
	}

	vector<uchar> status;
	vector<KeyPoint> pts1_good, pts2_good;

	// now we allign the matched points.
	vector<KeyPoint> pts1_temp, pts2_temp;
	allignPoints(keypoint_img1, keypoint_img2, good_matches, pts1_temp, pts2_temp);
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
	cout << "match pts1_good size: " << pts1_good.size() << endl;
	cout << "match pts2_good size: " << pts2_good.size() << endl;
	good_keypts = pts1_good;
	if (show) {
		cout << "after fund matrix matches: " << new_matches.size() << endl;
	}
	good_matches = new_matches;
	points1.clear();
	points2.clear();
	for (unsigned int i = 0; i < good_matches.size(); ++i) {
		points1.push_back(pts1_temp[i].pt);
		points2.push_back(pts2_temp[i].pt);
	}
	minMaxIdx(points1, &min, &max);
	//draw the matches found.
	Mat img_matches;
	drawMatches(img1, keypoint_img1, img2, keypoint_img2, good_matches, img_matches,
				Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	
	if (show) {
		imwrite("matches.jpg", img_matches);
		cout << "image saved" << endl;
	}
	matches = good_matches;
	cout << "matches size: " << matches.size() << endl;

	return true;
}

/**
* This function decomposes the Essential matrix E using the SVD class.
* Returns: Rotation Matricies (R1, R2) and Translation Matricies (t1, t2).
*/
bool DecomposeEssentialMat(Mat_<double>& E, Mat_<double>& R1, Mat_<double>& R2,
	Mat_<double>& t1, bool show) {
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
		if (show) {
			cout << "singular values are too far apart" << endl;
		}
		return false;
	} else if (w.at<double>(2) > 1e-06) {
		if (show) {
			cout << "final singluar value should be 0" << endl;
			cout << w.at<double>(2) << endl;
		}
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
	vector<KeyPoint>& correspondingImg1Pt, bool show) {

	if (show) {
		cout << "starting TriangulatePoints.. " << endl;
	}
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
	if (show) {
		cout << "finished triangulation with: " << mse[0] << " error" << endl;
	}
	return mse[0];
} 


/**
* this method processes the images in a directory and adds them to the list
*/
void processImages(vector<Mat> &images, vector<Mat> &imagesColored, 
	char* dirName, bool show) {
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
				Mat imageColored = imread(path, IMREAD_COLOR);
				if (!image.data || !imageColored.data) {
					cout << "could not read image: " << entity->d_name << endl;
				} else {
					images.push_back(image);
					imagesColored.push_back(imageColored);
				}
			}
		}
	}
	if (show) {
		cout << "found : " << imagesColored.size() << " colored images." << endl;
		cout << "found : " << images.size() << " grayscale images." << endl;
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
	vector<uchar>& status, bool show) {
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
	if (show) {
		cout << "percentage of points infront of camera: " << percentage << endl;
	}
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

void useage() {
	cout << "Useage: \n" << endl;
	cout << "./3drecon <image directory>" << endl;
	cout << "./3drecon <image directory> <verbose>" << endl;
	cout << "./3drecon <image directory> <calibration image path> <verbose>" << endl;
}

/**
* This function tests all 3 different possible P2 matricies and picks one or none.
*/
bool findP2Matrix(Matx34d& P1, Matx34d& P2, const Mat& K, const Mat& distanceCoeffs,
	vector<KeyPoint>& pts1_good, vector<KeyPoint>& pts2_good,
	Mat_<double> R1, Mat_<double> R2, Mat_<double> t1, bool show) {
	P2 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0), R1(2,1), R1(2,2), t1(2));

	vector<CloudPoint> pointCloud1;
	vector<CloudPoint> pointCloud2;
	vector<KeyPoint> correspondingImg1pts;
	Mat Kinv = K.inv();

	if (show) {
		cout << "starting test for P2, first configuration: \n" << endl;
		cout << "Testing P2 "<< endl << Mat(P2) << endl;
		cout << endl;
	}
	double reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
		distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts, show);
	double reproj_err2 = TriangulatePoints(pts2_good, pts1_good, K, Kinv,
		distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts, show);

	vector<uchar> temp_status;

	if (!testTriangulation(pointCloud1, P2, temp_status, show) 
		|| !testTriangulation(pointCloud2, P1, temp_status, show) || reproj_err1 > 100.0
		|| reproj_err2 > 100.0) {
		//try a new P2
		P2 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	-t1(0),
			R1(1,0),	R1(1,1),	R1(1,2),	-t1(1),
			R1(2,0), R1(2,1), R1(2,2), -t1(2));
		if (show) {
			cout << "\nstarting test for P2, second configuration: \n" << endl;
			cout << "Testing P2 "<< endl << Mat(P2) << endl;
			cout << endl;
		}
		pointCloud1.clear();
		pointCloud2.clear();
		correspondingImg1pts.clear();

		reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
			distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts, show);
		reproj_err2 = TriangulatePoints(pts2_good, pts1_good, K, Kinv,
			distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts, show);

		if (!testTriangulation(pointCloud1, P2, temp_status, show) 
			|| !testTriangulation(pointCloud2, P1, temp_status, show) || reproj_err1 > 100.0
			|| reproj_err2 > 100.0) {
			if (!checkRotationMat(R2)) {
				cout << "R2 was not valid" << endl;
				return false;
			}

			//try another P2
			P2 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
					R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
					R2(2,0), R2(2,1), R2(2,2), t1(2));

			if (show) {
				cout << "\nstarting test for P2, thrid configuration: \n" << endl;
				cout << "Testing P2 "<< endl << Mat(P2) << endl;
				cout << endl;
			}
			
			pointCloud1.clear();
			pointCloud2.clear();
			correspondingImg1pts.clear();

			reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
				distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts, show);
			reproj_err2 = TriangulatePoints(pts2_good, pts1_good, K, Kinv,
				distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts, show);

			if (!testTriangulation(pointCloud1, P2, temp_status, show) 
				|| !testTriangulation(pointCloud2, P1, temp_status, show) || reproj_err1 > 100.0
				|| reproj_err2 > 100.0) {

				//try the last P2
				P2 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	-t1(0),
						R2(1,0),	R2(1,1),	R2(1,2),	-t1(1),
						R2(2,0), R2(2,1), R2(2,2), -t1(2));

				if (show) {
					cout << "\nstarting test for last P2 configuration: \n" << endl;
					cout << "Testing P2 "<< endl << Mat(P2) << endl;
					cout << endl;
				}
				
				pointCloud1.clear();
				pointCloud2.clear();
				correspondingImg1pts.clear();

				reproj_err1 = TriangulatePoints(pts1_good, pts2_good, K, Kinv,
					distanceCoeffs, P1, P2, pointCloud1, correspondingImg1pts, show);
				reproj_err2 = TriangulatePoints(pts2_good, pts1_good, K, Kinv,
					distanceCoeffs, P2, P1, pointCloud2, correspondingImg1pts, show);
				if (!testTriangulation(pointCloud1, P2, temp_status, show) 
					|| !testTriangulation(pointCloud2, P1, temp_status, show) || reproj_err1 > 100.0
					|| reproj_err2 > 100.0) {
					return false;
				}
			}
		}
	}
	return true;
}

void allignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
	const vector<DMatch>& good_matches, vector<KeyPoint>& new_pts1,
	vector<KeyPoint>& new_pts2) {
	for (int i = 0; i < good_matches.size(); ++i) {
		assert(good_matches[i].queryIdx < imgpts1.size());
		new_pts1.push_back(imgpts1[good_matches[i].queryIdx]);
		assert(good_matches[i].trainIdx < imgpts2.size());
		new_pts2.push_back(imgpts2[good_matches[i].trainIdx]);
	}
}

bool triangulateBetweenViews(const Matx34d& P1, const Matx34d& P2, 
	vector<CloudPoint>& tri_pts, map<pair<int, int>, vector<DMatch>> all_matches,
	const vector<KeyPoint>& pts1_good, const vector<KeyPoint>& pts2_good,
	const Mat& K, const Mat& distanceCoeffs, 
	vector<KeyPoint>& correspImg1Pt, vector<int>& add_to_cloud,
	bool show, vector<CloudPoint>& global_pcloud, int image_size, int idx1, int idx2) {

	//allignPoints(pts1_good, pts2_good, good_matches, pts1_temp, pts2_temp);
	Mat Kinv = K.inv();
	vector<DMatch> good_matches = all_matches[make_pair(idx1, idx2)];
	double err = TriangulatePoints(pts1_good, pts2_good, K, Kinv, distanceCoeffs,
		P1, P2, tri_pts, correspImg1Pt, show);
	if (show) {
		cout << "reprojection error was: " << err << endl;
		cout << "\ntri_pts size: " << tri_pts.size() << endl;
	}
	vector<uchar> trig_status;
	if (!testTriangulation(tri_pts, P1, trig_status, show) 
		|| !testTriangulation(tri_pts, P2, trig_status, show)) {
		cout << "err: triangulation failed" << endl;
		return false;
	}

	vector<double> reproj_error;
	for (int i = 0; i < tri_pts.size(); ++i) {
		reproj_error.push_back(tri_pts[i].reprojection_error);
	}
	sort(reproj_error.begin(), reproj_error.end());
	double err_cutoff = reproj_error[4 * reproj_error.size() / 5] * 2.4;

	vector<CloudPoint> new_triangulated_filtered;
	vector<DMatch> new_matches;
	for (int i = 0; i < tri_pts.size(); ++i) {
		if (trig_status[i] == 0) {
			continue;
		}
		if (tri_pts[i].reprojection_error > 16.0) {
			continue;
		}
		if (tri_pts[i].reprojection_error < 4.0 ||
			tri_pts[i].reprojection_error < err_cutoff) {
			new_triangulated_filtered.push_back(tri_pts[i]);
			new_matches.push_back(good_matches[i]);
		} else {
			continue;
		}
	}

	if (show) {
		cout << "# points that are triangulated: " << new_triangulated_filtered.size() << endl;
	}
	if (new_triangulated_filtered.size() <= 0) {
		cout << "err: could not find enough points" << endl;
		return false;
	}
	tri_pts = new_triangulated_filtered;
	good_matches = new_matches;
	add_to_cloud.clear();
	add_to_cloud.resize(tri_pts.size(), 1);
	int found_in_other_views = 0;
	for (int j = 0; j < tri_pts.size(); ++j) {

		tri_pts[j].imgpt_for_img = vector<int>(image_size, -1);
		tri_pts[j].imgpt_for_img[idx1] = good_matches[j].queryIdx;
		tri_pts[j].imgpt_for_img[idx2] = good_matches[j].trainIdx;
		bool found = false;
		for (unsigned int image = 0; image < image_size; ++image) {
			if (image != idx1) {
				//get the matches of all other points.
				vector<DMatch> submatches = all_matches[make_pair(image, idx2)];
				for (int k = 0; k < submatches.size(); ++k) {
					if (submatches[k].trainIdx == good_matches[j].trainIdx &&
						!found) {
						for (unsigned int pt3d = 0; pt3d < global_pcloud.size(); ++pt3d) {
							if (global_pcloud[pt3d].imgpt_for_img[image] == submatches[k].queryIdx) {
								global_pcloud[pt3d].imgpt_for_img[idx1] = good_matches[j].trainIdx;
								global_pcloud[pt3d].imgpt_for_img[idx2] = good_matches[j].queryIdx;
								found = true;
								add_to_cloud[j] = 0;
							}
						}
					}
				}
			}
		}

		if (found) {
			found_in_other_views++;
		} else {
			add_to_cloud[j] = 1;		
		}
	}
	cout << "add_to_cloud size: " << countNonZero(add_to_cloud) << endl;
	return true;
}
//TODO: implement an autocalibration method for the camera intrinsics.