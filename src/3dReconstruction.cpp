#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "3dReconstruction.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {

	//process the images and save them in their data structures
	processImages(argv[1]);
	all_keypoints.resize(images.size(), vector<KeyPoint>());
	all_descriptors.resize(images.size(), Mat());
	cout << "processed images\n" << endl;

	debug = false;
	auto_calibrate = false;

	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		useage();
		return -1;
	} else if (argc == 4) {
		if (string(argv[3]) == "-V") {
			debug = true;
		} 
		if (string(argv[2]) == "-A") {
			//auto-calibrate.
			auto_calibrate = true;
			Size imgSize = images[0].size();
			double max_w_h = MAX(imgSize.height, imgSize.width);
			K = (Mat_<double>(3, 3) << max_w_h, 0, imgSize.width/2.0,
										0, max_w_h, imgSize.height/2.0,
										0, 0, 1);
			distortionCoeffs = Mat_<double>::zeros(1, 4);
		} else {
			CameraCalib cc(argv[2], K, distortionCoeffs, rVectors, tVectors, debug);	
		}
		
	} else if (argc == 3) {
		if (string(argv[2]) == "-V") {
			debug = true;
		} else if (string(argv[2]) == "-A") {
			auto_calibrate = true;
			Size imgSize = images[0].size();
			double max_w_h = MAX(imgSize.height, imgSize.width);
			K = (Mat_<double>(3, 3) << max_w_h, 0, imgSize.width/2.0,
										0, max_w_h, imgSize.height/2.0,
										0, 0, 1);
			distortionCoeffs = Mat_<double>::zeros(1, 4);
		} else {
			CameraCalib cc("../src/calibInfo.yml", K, distortionCoeffs);	
		}
	} else if (argc == 2) {
		//read in calibration info from file.
		CameraCalib cc("../src/calibInfo.yml", K, distortionCoeffs);
		debug = false;
	}

	if (images.size() == 0) {
		cout << "found no images" << endl;
		return -1;
	}

	//Detect features for each image.
	for (int i = 0; i < images.size(); ++i) {
		findFeatures(images[i], i);
	}
	cout << "---done detecting and computing features\n" << endl;

	cout << "Matching Descriptors......." << endl;
	//detect maching keypoints for different images.
	for (int i = 0; i < images.size() - 1; ++i) {
		for (int j = i + 1; j < images.size(); ++j) {
			bool matches_res = computeMatches(i, j);
			if (matches_res) {
				cout << "successful matching with image: [" << i << ", " << j << "]\n" << endl;
				//cout << "current cloud size: " << all_points.size() << "\n\n" << endl;
			} else {
				cout << "failed matching with image: [" << i << ", " << j << "]\n\n" << endl;
			}
		}
	}

	//get the baseline triangulation with 2 views
	triangulate2Views(first_view, second_view);
	if (globalPoints.size() == 0) {
		//could not find baseline triangulation
		return -1;
	}
	addMoreViewsToReconstruction();
	saveCloudToPLY();

	vector<Vec3b> RGBCloud;
	getPointRGB(RGBCloud);
	if (RGBCloud.size() != globalPoints.size()) {
		cout << "error: color values are not the same size as the points" << endl;
		return -1;
	}

	//convert the found cloudpoints into points and colors and display them using PCL.
	displayCloud(RGBCloud);

	return 0;
}

void addMoreViewsToReconstruction() {
	while(done_views.size() != images.size()) {
		Find2D3DCorrespondences();
		unsigned int max_2d3d_view = -1, max_2d3d_count = 0;
		for (const auto& match2D3D : imageCorrespondences) {
			int matchSize = match2D3D.second.first.size();
			if (matchSize > max_2d3d_count) {
				max_2d3d_view = match2D3D.first;
				max_2d3d_count = matchSize;
			}
		}
		int i = max_2d3d_view;
		done_views.insert(i);

		cout << "adding view: " << i << " with " << max_2d3d_count << " matches." << endl;

		Matx34d Pnew;
		vector<Point3f> max_3d = imageCorrespondences[i].second; 
		vector<Point2f> max_2d = imageCorrespondences[i].first;

		bool good_poseEstimation = estimatePose(max_3d, max_2d, Pnew);
		if (!good_poseEstimation) {
			cout << "could not recover pose for view: " << i << endl;
			continue;
		}
		cout << "--found good pose" << endl;
		all_pmats[i] = Pnew;

		bool success = false;
		for (const int good_view : good_views) {
			size_t idx1 = (good_view < i) ? good_view : i;
			size_t idx2 = (good_view < i) ? i : good_view;
			if (idx1 == idx2) continue;
			vector<Point3DInMap> cloud;
			Matx34d P0, P1;
			vector<DMatch> newMatches;
			bool sfm_res = computeSFM(idx1, idx2, cloud, P0, P1, newMatches);
			if (!sfm_res) {
				cout << "could not compute the SFM" << endl;
				continue;
			}

			//update the matches.
			all_matches[make_pair(idx1, idx2)] = newMatches;
			vector<DMatch> flipedMatches;
			flipMatches(newMatches, flipedMatches);
			all_matches[make_pair(idx2, idx1)] = flipedMatches;

			//Triangulate the points for the new view
			cout << "before triangulation: " << cloud.size() << endl;
			bool tri_res = triangulateBetweenViews(P0, P1, cloud, idx1, idx2);
			cout << "after triangulation: " << cloud.size() << endl;	

			if (tri_res) {
				cout << "\nMerge points for: " << idx1 << " and " << idx2 << "....." << endl;
				cout << "globalPoints size: before" << globalPoints.size() << endl;
				mergeClouds(cloud);
				cout << "globalPoints after size: " << globalPoints.size() << endl;
				success = true;
			} else {
				cout << "could not compute sfm" << endl;
			}

			if (success) {
				cout << "\n-- Starting Bundle Adjustment --" << endl;
				Mat cam_matrix = K;
				bool ba_res = adjustBundleCeres(globalPoints, cam_matrix);
				cout << "...finished bundle adjustment\n" << endl;
				K = cam_matrix;
			}
			good_views.insert(i);

		}
	}
}

void triangulate2Views(int first_view, int second_view) {

	//getting baseline reconstruction from 2 view
	list<pair<int,pair<int,int>>> percent_matches;
	sortMatchesFromHomography(percent_matches);

	//parameters for ComputeSFM method
	Matx34d P0 = Matx34d::eye();
	Matx34d P1 = Matx34d::eye();
	vector<DMatch> newMatches;
	vector<Point3DInMap> cloud;

	cout << "\nGetting baseline triangulation......" << endl;
	for (auto it = percent_matches.begin(); it != percent_matches.end(); ++it) {
		int i = it->second.first;
		int j = it->second.second;
		first_view  = i;
		second_view = j;
		
		cout << "image [" << i << ", " << j << "] with " << it->first << "%" << endl;
		bool sfm_res = computeSFM(i, j, cloud, P0, P1, newMatches);
		cout << "before triangulation: " << cloud.size() << endl;
		bool tri_res = triangulateBetweenViews(P0, P1, cloud, i, j);
		cout << "after triangulation: " << cloud.size() << endl;	

		//update the matches
		all_matches[make_pair(i, j)] = newMatches;
		vector<DMatch> flipedMatches;
		flipMatches(newMatches, flipedMatches);
		all_matches[make_pair(j, i)] = flipedMatches;

		cout << "new matches size: " << newMatches.size() << endl;

		if (!sfm_res) {
			cout << "failed sfm with image: [" << i << ", " << j << "]" << endl;
			continue;
		} 

		done_views.insert(first_view);
		done_views.insert(second_view);
		good_views.insert(first_view);
		good_views.insert(second_view);

		all_pmats[i] = P0;
		all_pmats[j] = P1;
		globalPoints = cloud;

		cout << "\n-- Starting Bundle Adjustment --" << endl;
		Mat cam_matrix = K;
		bool ba_res = adjustBundleCeres(globalPoints, cam_matrix);
		cout << "...finished bundle adjustment\n" << endl;
		K = cam_matrix;
		break;
	}
	cout << "\n...after baseline triangulation found: " << cloud.size() << " points.\n" << endl;
}

bool estimatePose(vector<Point3f>& points3d, vector<Point2f>& points2d, Matx34d& Pnew) {

	Mat rvec, tvec;
	Mat inliers;
	solvePnPRansac(points3d, points2d, K, distortionCoeffs, rvec, tvec, false, 1000, 
		10.0f, 0.99, inliers);
	if ((float)countNonZero(inliers) < ((float)points2d.size() / 5)) {
		cout << "error: inliers ratio is too small in pose estimation" << endl;
		cout << "ratio was: " << ((float)countNonZero(inliers) / (float)points2d.size()) << endl;
		return false;
	}
	Mat rot;
	Rodrigues(rvec, rot);
	rot.copyTo(Mat(3, 4, CV_32FC1, Pnew.val)(ROT));
	tvec.copyTo(Mat(3, 4, CV_32FC1, Pnew.val)(TRA));
	return true;
}

/**
* this function finds the points that have a 3d coordinate in the global point cloud
* and an image coordinate in the keypoints that correspond.
* and further refines the cloud point.
*/ 
void Find2D3DCorrespondences() {

	imageCorrespondences.clear();

	for (int i = 0; i < images.size(); ++i) {
		if (done_views.find(i) != done_views.end()) {
			continue;
		}
		vector<Point2f> points2d;
		vector<Point3f> points3d;
		for (const Point3DInMap& p : globalPoints) {
			bool found = false;

			for (const auto& view : p.originatingViews) {
				const int origViewIdx = view.first;
				const int origViewFeatureIdx = view.second;

				const int viewIdx1 = (origViewIdx < i) ? origViewIdx : i;
				const int viewIdx2 = (origViewIdx < i) ? i : origViewIdx;

				for (const DMatch& m : all_matches[make_pair(viewIdx1, viewIdx2)]) {
					int matched2DPoint = -1;
					if (origViewIdx < i) {
						if (m.queryIdx == origViewFeatureIdx) {
							matched2DPoint = m.trainIdx;
						}
					} else {
						if (m.trainIdx == origViewFeatureIdx) {
							matched2DPoint = m.queryIdx;
						}
					}
					if (matched2DPoint >= 0) {
						const vector<KeyPoint> kpts = all_keypoints[i];
						vector<Point2f> points;
						KeyPointsToPoints(kpts, points);
						points2d.push_back(points[matched2DPoint]);
						points3d.push_back(p.p);
						found = true;
						break;
					}
				}
				if (found) {
					break;
				}
			}
		}
		imageCorrespondences[i] = make_pair(points2d, points3d);
	}
}

bool adjustBundleCeres(vector<Point3DInMap>& cloud, Mat& cam_matrix) {
	cout << "BA cloud size: " << cloud.size() << endl;
	ceres::Problem problem;
	typedef Matx<double, 1, 6> CameraVector;
	vector<CameraVector> cameraPoses6d;
	cameraPoses6d.reserve(all_pmats.size());
	for (int i = 0; i < all_pmats.size(); ++i) {
		const Matx34d& pose = all_pmats[i];
		if (pose(0, 0) == 0 && pose(1, 1) == 0 && pose(2, 2) == 0) {
			cameraPoses6d.push_back(CameraVector());
			continue;
		}
		Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
		 Matx33f R = pose.get_minor<3, 3>(0, 0);
        float angleAxis[3];
		ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);

		cameraPoses6d.push_back(CameraVector(angleAxis[0], angleAxis[1], angleAxis[2],
			t(0), t(1), t(2)));
	}

	double focal = cam_matrix.at<double>(0, 0);
	vector<Vec3d> points3d(cloud.size());
	for (int i = 0; i < cloud.size(); ++i) {
		const Point3DInMap& p = cloud[i];
		points3d[i] = Vec3d(p.p.x, p.p.y, p.p.z);

		for (const auto& kv : p.originatingViews) {
			Point2f p2d = all_keypoints[kv.first][kv.second].pt;
			p2d.x -= cam_matrix.at<double>(0, 2);
			p2d.y -= cam_matrix.at<double>(1, 2);

			ceres::CostFunction* cost_function = SimpleReprojectionError::Create(p2d.x, p2d.y);

			problem.AddResidualBlock(cost_function, NULL, cameraPoses6d[kv.first].val,
				points3d[i].val, &focal);
		}
	}

	ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.eta = 1e-2;
    options.max_solver_time_in_seconds = 20;
	options.logging_type = ceres::LoggingType::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
	cout << summary.BriefReport() << endl;

	if (!(summary.termination_type == ceres::CONVERGENCE)) {
		cout << "error BA failed" << endl;
		return false;
	}

	//update the camera parameters
	cam_matrix.at<double>(0, 0) = focal;
	cam_matrix.at<double>(0, 0) = focal;

	for (int i = 0; i < all_pmats.size(); ++i) {
		Matx34d& pose = all_pmats[i];
		Matx34d poseBefore = pose;

		if (pose(0, 0) == 0 && pose(1, 1) == 0 && pose(2, 2) == 0) {
			continue;
		}

		double rotationMat[9] = {0};
		ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);

		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; c++) {
				pose(c, r) = rotationMat[r * 3 + c];
			}
		}

		//Translation
        pose(0, 3) = cameraPoses6d[i](3);
        pose(1, 3) = cameraPoses6d[i](4);
		pose(2, 3) = cameraPoses6d[i](5);

	}

	for (int i = 0; i < cloud.size(); ++i) {
		cloud[i].p.x = points3d[i](0);
		cloud[i].p.y = points3d[i](1);
		cloud[i].p.z = points3d[i](2);
	}
	return true;
}

int get2DMeasurements(const vector<Point3DInMap>& globalPoints) {
	int count = 0;
	for (int i = 0; i < globalPoints.size(); ++i) {
		for (int j = 0; j < globalPoints[i].originatingViews.size(); ++j) {
			count++;
		}
	}
	return count;
}

//this function displays the point cloud.
void displayCloud(vector<Vec3b>& RGBCloud) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (int i = 0; i < globalPoints.size(); ++i) {
		Vec3b rgbv(255, 255, 255);
		if (RGBCloud.size() > i) {
			rgbv = RGBCloud[i];
		}
		// Point3d p = global_pcloud[i].pt;
		Point3f p = globalPoints[i].p;

		// check if any of the points are nan.
		if (isnan(p.x) || isnan(p.y) || isnan(p.z)) {
			continue;
		}

		pcl::PointXYZRGB pt;
		pt.x = p.x;
		pt.y = p.y;
		pt.z = p.z;
		uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
		pt.rgb = *reinterpret_cast<float*>(&rgb);
		cloudPtr->push_back(pt);
	}
	cout << "cloud size: " << cloudPtr->size() << endl;
	pcl::visualization::CloudViewer viewer("SFM");
	viewer.showCloud(cloudPtr);
	while(!viewer.wasStopped()) {
		
	}
}

/**
* this function computes the RGB values for the triangulated points.
* returns a vector of RGB values.
*/
void getPointRGB(vector<Vec3b>& RGBCloud) {
	int image_size = images.size();
	// RGBCloud.resize(global_pcloud.size());
	RGBCloud.resize(globalPoints.size());
	for (int i = 0; i < globalPoints.size(); ++i) {
		vector<Vec3b> point_colors;
		for (int j = 0; j < image_size; ++j) {
			int pt_idx = globalPoints[i].originatingViews[j];
			if (pt_idx >= all_keypoints[j].size()) {
				cout << "err: index out of bounds while getting point RGB" << endl;
				continue;
			}
			Point p = all_keypoints[j][pt_idx].pt;
			point_colors.push_back(imagesColored[j].at<Vec3b>(p));
		}
		Scalar res_color = mean(point_colors);
		RGBCloud[i] = (Vec3b(res_color[0], res_color[1], res_color[2]));
	}
}

/**
* this function computes SFM from a series of images, matches and camera parameters.
* returns a point cloud that contains 3d points.
*/
bool computeSFM(int idx1, int idx2, vector<Point3DInMap>& cloud, Matx34d& P0,
 Matx34d& P1, vector<DMatch>& newMatches) {

	vector<KeyPoint> kpts1 = all_keypoints[idx1];
	vector<KeyPoint> kpts2 = all_keypoints[idx2];
	int image_size = images.size();

	cout << "---SFM with images: [" << idx1 << ", " << idx2 << "]" << endl;

	vector<KeyPoint> pts1_temp, pts2_temp;
	vector<DMatch> matches = all_matches[make_pair(idx1, idx2)];
	cout << "matches size: " << matches.size() << endl;
	alignPoints(kpts1, kpts2, matches, pts1_temp, pts2_temp);
	vector<Point2f> points1, points2;
	for (unsigned int i = 0; i < matches.size(); ++i) {
		points1.push_back(pts1_temp[i].pt);
		points2.push_back(pts2_temp[i].pt);
	}
	Mat E, R, t;
	Mat mask;

	double f0 = K.at<double>(0, 0);
	Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));
	E = findEssentialMat(points1, points2, f0, pp, CV_RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points1, points2, R, t, f0, pp, mask);
	cout << "done recover pose " << endl;
	P0 = Matx34d::eye();
	P1 = Matx34d(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                    R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
					R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
	

	//prune the matches based on the inliers.
	newMatches.clear();
	for (int i = 0; i < mask.rows; ++i) {
		if (mask.at<uchar>(i)) {
			newMatches.push_back(matches[i]);
		}
	}

	return true;
}

/**
* this function computes the keypoints between 2 images, matches them and filters those matches.
* returns the filtered matches and their corresponding "good" keypoints.
*/
bool computeMatches(int idx1, int idx2) {

	Mat img1 = images[idx1];
	Mat img2 = images[idx2];

	Mat desc1 = all_descriptors[idx1];
	Mat desc2 = all_descriptors[idx2];

	vector<KeyPoint> kpts1 = all_keypoints[idx1];
	vector<KeyPoint> kpts2 = all_keypoints[idx2];

	vector<DMatch> matches = all_matches[make_pair(idx1, idx2)];

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches_;
	matcher.match(desc1, desc2, matches_);
	matches = matches_;


	cout << "found : " << matches.size() << " matches" << endl;

	//filter the matches
	cout << "----- filtering matches" << endl;

	set<int> existing_trainIdx;
	vector<DMatch> good_matches;

	for (unsigned int i = 0; i < matches_.size(); ++i) {
		if (matches_[i].trainIdx <= 0) {
			matches_[i].trainIdx  = matches_[i].imgIdx;
		}

		if (existing_trainIdx.find(matches_[i].trainIdx) == existing_trainIdx.end() && 
			matches_[i].trainIdx >= 0 && matches_[i].trainIdx < (int)(kpts2.size())) {
			good_matches.push_back(matches_[i]);
			existing_trainIdx.insert(matches_[i].trainIdx);
		}
	}

	vector<uchar> status;
	vector<KeyPoint> pts1_good, pts2_good;

	// now we align the matched points.
	vector<KeyPoint> pts1_temp, pts2_temp;
	alignPoints(kpts1, kpts2, good_matches, pts1_temp, pts2_temp);
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
	};
	all_good_keypoints[make_pair(idx1, idx2)] = pts1_good;
	all_good_keypoints[make_pair(idx2, idx1)] = pts1_good;
	if (debug) {
		cout << "match pts1_good size: " << pts1_good.size() << endl;
		cout << "match pts2_good size: " << pts2_good.size() << endl;
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
	drawMatches(img1, kpts1, img2, kpts2, good_matches, img_matches,
				Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	
	if (debug) {
		string name = "matches" + to_string(idx1) + "_" + to_string(idx2) + ".jpg";
 		imwrite(name, img_matches);
		cout << "image saved" << endl;
	}
	all_matches[make_pair(idx1, idx2)] = good_matches;
	vector<DMatch> flipedMatches;
	flipMatches(good_matches, flipedMatches);
	all_matches[make_pair(idx2, idx1)] = flipedMatches;
	cout << "matches size: " << good_matches.size() << endl;

	return true;
}


/**
* this method processes the images in a directory and adds them to the list
*/
void processImages(char* dirName) {
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
	cout << "found : " << imagesColored.size() << " colored images." << endl;
	cout << "found : " << images.size() << " grayscale images." << endl;
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

void useage() {
	cout << "Useage: \n" << endl;
	cout << "./3drecon <image directory>" << endl;
	cout << "./3drecon <image directory> <verbose>" << endl;
	cout << "./3drecon <image directory> <calibration image path> <verbose>" << endl;
}

/**
* this function aligns keypoints between two images.
* parameters: takes in the keypoints for the first and second image.
* 	a vector of matches.
* output: 2 vectors of aligned keypoints between the images.
*/
void alignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
	const vector<DMatch>& good_matches, vector<KeyPoint>& new_pts1,
	vector<KeyPoint>& new_pts2) {
	for (int i = 0; i < good_matches.size(); ++i) {
		assert(good_matches[i].queryIdx < imgpts1.size());
		new_pts1.push_back(imgpts1[good_matches[i].queryIdx]);
		assert(good_matches[i].trainIdx < imgpts2.size());
		new_pts2.push_back(imgpts2[good_matches[i].trainIdx]);
	}
}

/**
* this function aligns keypoints between two images.
* parameters: takes in the keypoints for the first and second image.
* 	a vector of matches.
* output: 2 vectors of aligned keypoints between the images.
*/
void alignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
	const vector<DMatch>& good_matches, vector<KeyPoint>& new_pts1,
	vector<KeyPoint>& new_pts2, vector<int>& leftBackRefrence, 
	vector<int>& rightBackRefrence, vector<Point2f>& points1, vector<Point2f>& points2) {
	
	new_pts1.clear();
	new_pts2.clear();
	for (int i = 0; i < good_matches.size(); ++i) {
		new_pts1.push_back(imgpts1[good_matches[i].queryIdx]);
		new_pts2.push_back(imgpts2[good_matches[i].trainIdx]);
		leftBackRefrence.push_back(good_matches[i].queryIdx);
		rightBackRefrence.push_back(good_matches[i].trainIdx);
	}
	KeyPointsToPoints(new_pts1, points1);
	KeyPointsToPoints(new_pts2, points2);
}


/**
* this function triangulates points between two images.
* parameters: takes in the 2 camera matrices, thier keypoints and their matches.
* output: a vector of points to add to the cloud.
*/
bool triangulateBetweenViews(const Matx34d& P1, const Matx34d& P2,
	vector<Point3DInMap>& cloud, int idx1, int idx2) {

	const float MIN_REPROJECTION_ERROR = 10.0;

	vector<int> leftBackReference;
	vector<int> rightBackReference;
	vector<KeyPoint> kpts1 = all_keypoints[idx1];
	vector<KeyPoint> kpts2 = all_keypoints[idx2];
	vector<KeyPoint> kpts1_temp, kpts2_temp;
	vector<DMatch> matches = all_matches[make_pair(idx1, idx2)];
	vector<Point2f> points1, points2;

	alignPoints(kpts1, kpts2, matches, kpts1_temp, kpts2_temp, 
		leftBackReference, rightBackReference, points1, points2);

	Mat normilized_pts1, normilized_pts2;
	undistortPoints(points1, normilized_pts1, K, Mat());
	undistortPoints(points2, normilized_pts2, K, Mat());
	Mat points3dHomogeneous;
	triangulatePoints(P1, P2, normilized_pts1, normilized_pts2, points3dHomogeneous);
	Mat points3d;
	convertPointsFromHomogeneous(points3dHomogeneous.t(), points3d);

	Mat rvec1;
	Rodrigues(P1.get_minor<3, 3>(0, 0), rvec1);
	Mat tvec1(P1.get_minor<3, 1>(0, 3).t());

	vector<Point2f> projectedImg1(points1.size());
	projectPoints(points3d, rvec1, tvec1, K, Mat(), projectedImg1);

	Mat rvec2;
	Rodrigues(P2.get_minor<3, 3>(0, 0), rvec2);
	Mat tvec2(P2.get_minor<3, 1>(0, 3).t());

	vector<Point2f> projectedImg2(points2.size());
	projectPoints(points3d, rvec2, tvec2, K, Mat(), projectedImg2);

	for (size_t i = 0; i < points3d.rows; ++i) {
		if (norm(projectedImg1[i] - points1[i]) > MIN_REPROJECTION_ERROR ||
			norm(projectedImg2[i] - points2[i]) > MIN_REPROJECTION_ERROR) {
			// cout << "reprojection error too high in triangulateViews" << endl;
			continue;
		}

		Point3DInMap p;
		p.p = Point3f(points3d.at<float>(i, 0),
                      points3d.at<float>(i, 1),
                      points3d.at<float>(i, 2));
		p.originatingViews[idx1] = leftBackReference[i];
		p.originatingViews[idx2] = rightBackReference[i];
		cloud.push_back(p);
	}
	return true;
}

void sortMatchesFromHomography(list<pair<int, pair<int, int>>>& percent_matches) {

	for (auto it = all_matches.begin(); it != all_matches.end(); ++it) {
		if ((*it).second.size() < 100) {
			percent_matches.push_back(make_pair(100, (*it).first));
		} else {
			vector<KeyPoint> keypts1, keypts2;
			vector<Point2f> pts1, pts2;
			int idx1 = (*it).first.first;
			int idx2 = (*it).first.second;

			alignPoints(all_keypoints[idx1], all_keypoints[idx2], all_matches[make_pair(idx1, idx2)],
				keypts1, keypts2);
			for (int i = 0; i < keypts1.size(); ++i) {
				pts1.push_back(keypts1[i].pt);
				pts2.push_back(keypts2[i].pt);
			}

			double min,max;
			minMaxIdx(pts1, &min, &max);
			vector<uchar> status;
			Mat H = findHomography(pts1, pts2, status, CV_RANSAC, 0.004 * max);
			int inliers = countNonZero(status);
			int percent = (int)(((double)inliers) / ((double)(*it).second.size()) * 100);
			percent_matches.push_back(make_pair((int)percent, (*it).first));
			if (debug) {
				cout << "percentage inliers for: " << idx1 << ", " << idx2 << ": " 
				<< percent <<  endl;
			}
		}
	}
	percent_matches.sort(sortFromPercentage);
}

int findHomographyInliers(int idx1, int idx2) {
	vector<KeyPoint> kpts1 = all_keypoints[idx1];
	vector<KeyPoint> kpts2 = all_keypoints[idx2];
	vector<DMatch> matches = all_matches[make_pair(idx1, idx2)];
	vector<KeyPoint> kpts1_temp, kpts2_temp;
	vector<Point2f> points1, points2;

	alignPoints(kpts1, kpts2, matches, kpts1_temp, kpts2_temp);
	KeyPointsToPoints(kpts1_temp, points1);
	KeyPointsToPoints(kpts2_temp, points2);

	Mat inliers, H;
	if (matches.size() >= 4) {
		H = findHomography(points1, points2, CV_RANSAC, 10.0f, inliers);
	}
	if (matches.size() < 4 || H.empty()) {
		return 0;
	}
	return countNonZero(inliers);

}

bool sortFromPercentage(pair<int, pair<int, int>> a, pair<int, pair<int, int>> b) {
	return a.first < b.first;
}

void findFeatures(Mat& img, int idx) {
	int min_hessian = 400;
	Ptr<SURF> detector = SURF::create(min_hessian);
	vector<KeyPoint> kp;
	Mat des;
	detector->detectAndCompute(img, Mat(), kp, des);
	all_descriptors[idx] = des;
	all_keypoints[idx] = kp;
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
    ps.clear();
    for (const auto& kp : kps) {
        ps.push_back(kp.pt);
    }
}
void flipMatches(const vector<DMatch>& matches, vector<DMatch>& flipedMatches) {
	for (int i = 0; i < matches.size(); ++i) {
		flipedMatches.push_back(matches[i]);
		swap(flipedMatches.back().queryIdx, flipedMatches.back().trainIdx);
	}
}

void mergeClouds(const vector<Point3DInMap> cloud) {

	const float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE   = 0.08;
	const float MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE = 20.0;

	map<pair<int, int>, vector<DMatch>> mergedMatches;
	int newPoints = 0, mergedPoints = 0;

	for (const Point3DInMap& p : cloud) {
		Point3f newPoint = p.p;
		bool foundMatchingViews = false, foundMatchingPoint = false;
		for (Point3DInMap& existingPoint : globalPoints) {
			if (norm(existingPoint.p - newPoint) < MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE) {
				//point is very close to an existing point.
				foundMatchingPoint = true;
				for (const auto& newKv : p.originatingViews) {
					for (const auto& existingKv : existingPoint.originatingViews) {
						bool foundMatchingFeature = false;
						bool isLeft = newKv.first < existingKv.first;
						int viewIdx1 = (isLeft) ? newKv.first : existingKv.first;
						int viewIdx2 = (isLeft) ? existingKv.first : newKv.first;
						int featureIdx1 = (isLeft) ? newKv.second : existingKv.second;
						int featureIdx2 = (isLeft) ? existingKv.second : newKv.second;
						vector<DMatch> matches = all_matches[make_pair(viewIdx1, viewIdx2)];

						for (const DMatch& match : matches) {
							if (match.queryIdx == featureIdx1 &&
								match.trainIdx == featureIdx2 &&
								match.distance < MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE) {
								mergedMatches[make_pair(viewIdx1, viewIdx2)].push_back(match);
								foundMatchingFeature = true;
								break;
							}
						}

						if (foundMatchingFeature) {
							existingPoint.originatingViews[newKv.first] = newKv.second;
							foundMatchingViews = true;
						}
					}
				}
			}
			if (foundMatchingViews) {
				mergedPoints++;
				break;
			}
		}
		if (!foundMatchingViews && !foundMatchingPoint) {
			globalPoints.push_back(p);
			newPoints++;
		}
	}
	cout << "new points size: " << newPoints << endl;
	cout << "merged points size: " << mergedPoints << endl;
} 

void autoCalibrate(Mat& E) {
	cout << "--auto calibrating--\n" << endl;
	double sigma1 = E.at<double>(0, 0);
	double sigma2 = E.at<double>(1, 1);
	
	//our cost function becomes 1 - (sigma1 / sigma2) 
	//and we want that to be as close to 0 as possible
	cout << E << endl;


}

void saveCloudToPLY() {

	cout << "saving cloud to: points.ply" << endl;

    ofstream ofs("points.ply");

    //write PLY header
    ofs << "ply                 " << endl <<
           "format ascii 1.0    " << endl <<
           "element vertex " << globalPoints.size() << endl <<
           "property float x    " << endl <<
           "property float y    " << endl <<
           "property float z    " << endl <<
           "property uchar red  " << endl <<
           "property uchar green" << endl <<
           "property uchar blue " << endl <<
           "end_header          " << endl;

    for (const Point3DInMap& p : globalPoints) {
    	//get color from first originating view
		auto originatingView = p.originatingViews.begin();
		const int viewIdx = originatingView->first;
		// Point2f p2d = mImageFeatures[viewIdx].points[originatingView->second];
		vector<KeyPoint>& kpts = all_keypoints[viewIdx];
		vector<Point2f> points;
		KeyPointsToPoints(kpts, points);
		Point2f p2d = points[originatingView->second];

		Vec3b pointColor = imagesColored[viewIdx].at<Vec3b>(p2d);

		//write vertex
        ofs << p.p.x              << " " <<
        	   p.p.y              << " " <<
			   p.p.z              << " " <<
			   (int)pointColor(2) << " " <<
			   (int)pointColor(1) << " " <<
			   (int)pointColor(0) << " " << endl;
    }

    ofs.close();

    cout << "saving cameras to: points.ply" << endl;
    ofstream ofsc("cameras.ply");

    //write PLY header
    ofsc << "ply                 " << endl <<
           "format ascii 1.0    " << endl <<
           "element vertex " << (all_pmats.size() * 4) << endl <<
           "property float x    " << endl <<
           "property float y    " << endl <<
           "property float z    " << endl <<
           "element edge " << (all_pmats.size() * 3) << endl <<
           "property int vertex1" << endl <<
           "property int vertex2" << endl <<
           "property uchar red  " << endl <<
           "property uchar green" << endl <<
           "property uchar blue " << endl <<
           "end_header          " << endl;

    //save cameras polygons..
    for (const auto& pmat : all_pmats) {
    	Matx34d pose = pmat.second;
        Point3d c(pose(0, 3), pose(1, 3), pose(2, 3));
        Point3d cx = c + Point3d(pose(0, 0), pose(1, 0), pose(2, 0)) * 0.2;
        Point3d cy = c + Point3d(pose(0, 1), pose(1, 1), pose(2, 1)) * 0.2;
        Point3d cz = c + Point3d(pose(0, 2), pose(1, 2), pose(2, 2)) * 0.2;

        ofsc << c.x  << " " << c.y  << " " << c.z  << endl;
        ofsc << cx.x << " " << cx.y << " " << cx.z << endl;
        ofsc << cy.x << " " << cy.y << " " << cy.z << endl;
        ofsc << cz.x << " " << cz.y << " " << cz.z << endl;
    }

    const int camVertexStartIndex = globalPoints.size();

    for (size_t i = 0; i < all_pmats.size(); i++) {
        ofsc << (i * 4 + 0) << " " <<
                (i * 4 + 1) << " " <<
                "255 0 0" << endl;
        ofsc << (i * 4 + 0) << " " <<
                (i * 4 + 2) << " " <<
                "0 255 0" << endl;
        ofsc << (i * 4 + 0) << " " <<
                (i * 4 + 3) << " " <<
                "0 0 255" << endl;
    }
}