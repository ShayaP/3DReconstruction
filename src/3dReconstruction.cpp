//include files from SSBA for bundle adjustment
#define V3DLIB_ENABLE_SUITESPARSE

#include <Math/v3d_linear.h>
#include <Base/v3d_vrmlio.h>
#include <Geometry/v3d_metricbundle.h>

#include "3dReconstruction.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace V3D;

int main(int argc, char** argv) {

	//process the images and save them in their data structures
	processImages(argv[1]);
	all_keypoints.resize(images.size(), vector<KeyPoint>());
	all_descriptors.resize(images.size(), Mat());
	cout << "processed images" << endl;

	show = false;

	if (argc < 2) {
		cout << "wrong number of arguments" << endl;
		useage();
		return -1;
	} else if (argc == 4) {
		if (string(argv[3]) == "-V") {
			show = true;
		} 
		if (string(argv[2]) == "-A") {
			//auto-calibrate.
			autoCalibrate();
		} else {
			CameraCalib cc(argv[2], K, distanceCoeffs, rVectors, tVectors, show);	
		}
		
	} else if (argc == 3) {
		if (string(argv[2]) == "-V") {
			show = true;
		}
		CameraCalib cc("../src/calibInfo.yml", K, distanceCoeffs);
	} else if (argc == 2) {
		//read in calibration info from file.
		CameraCalib cc("../src/calibInfo.yml", K, distanceCoeffs);
		show = false;
	}

	for (int i = 0; i < images.size(); ++i) {
		findFeatures(images[i], i);
	}
	cout << "done detecting and computing features" << endl;

	//detect maching keypoints for different images.
	for (int i = 0; i < images.size() - 1; ++i) {
		for (int j = i + 1; j < images.size(); ++j) {
			bool matches_res = computeMatches(i, j);
			if (matches_res) {
				cout << "successful matching with image: [" << i << ", " << j << "]" << endl;
				//cout << "current cloud size: " << all_points.size() << "\n\n" << endl;
			} else {
				cout << "failed matching with image: [" << i << ", " << j << "]\n\n" << endl;
			}
		}
	}

	//get the baseline triangulation with 2 views
	triangulate2Views(first_view, second_view);

	//add more views
	// Matx34d P = all_pmats[second_view];
	// Mat_<double> t = (Mat_<double>(1, 3) << P(0, 3), P(1, 3), P(2, 3));
	// Mat_<double> R = (cv::Mat_<double>(3,3) << P(0,0), P(0,1), P(0,2), 
	// 										P(1,0), P(1,1), P(1,2), 
	// 										P(2,0), P(2,1), P(2,2));
	// Mat_<double> rvec(1, 3);
	// Rodrigues(R, rvec);
	// done_views.insert(first_view);
	// done_views.insert(second_view);
	// good_views.insert(first_view);
	// good_views.insert(second_view);

	addMoreViewsToReconstruction();

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
		// vector<Point3f> max_3d; 
		// vector<Point2f> max_2d;
		// for (int i = 0; i < images.size(); ++i) {
		// 	if (done_views.find(i) != done_views.end()) continue;
		// 	vector<Point3f> tmp3d; 
		// 	vector<Point2f> tmp2d;
		// 	Find2D3DCorrespondences(i, tmp3d, tmp2d);
		// 	if (tmp3d.size() > max_2d3d_count) {
		// 		max_2d3d_count = tmp3d.size();
		// 		max_2d3d_view = i;
		// 		max_3d = tmp3d;
		// 		max_2d = tmp2d;
		// 	}
		// }
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
		cout << "found good pose" << endl;
		all_pmats[i] = Pnew;

		bool success = false;
		for (const int good_view : good_views) {
			size_t idx1 = (good_view < i) ? good_view : i;
			size_t idx2 = (good_view < i) ? i : good_view;
			if (idx1 == idx2) continue;
			vector<Point3DInMap> cloud;
			bool sfm_res = computeSFM(idx1, idx2, cloud);
			if (sfm_res) {
				cout << "merge points for: " << idx1 << " and " << idx2 << endl;
				mergeClouds(cloud);
				success = true;
			} else {
				cout << "could not compute sfm" << endl;
			}

			if (success) {
				Mat cam_matrix = K;
				adjustBundle(cam_matrix);
				K = cam_matrix;
			}
			good_views.insert(i);

		}

	// 	all_pmats[i] = Matx34d (R(0,0),R(0,1),R(0,2),t(0),
	// 							R(1,0),R(1,1),R(1,2),t(1),
	// 							R(2,0),R(2,1),R(2,2),t(2));
	// 	for (auto done_view = good_views.begin(); done_view != good_views.end(); ++done_view) {
	// 		int view = *done_view;
	// 		cout << "view: " << view << " i: " << i << endl;
	// 		if (view == i) continue;
	// 		vector<CloudPoint> new_triangulated;
	// 		vector<int> add_to_cloud;
	// 		vector<KeyPoint> correspondingImg1Pt;
	// 		Matx34d P1 = all_pmats[view];
	// 		Matx34d P2 = all_pmats[i];
	// 		bool good_triangulation = triangulateBetweenViews(new_triangulated,
	// 			add_to_cloud, correspondingImg1Pt, view, i);
	// 		if (!good_triangulation) {
	// 			cout << "err: bad triangulation" << endl;
	// 			continue;	
	// 		} 
	// 		for (int j = 0; j < add_to_cloud.size(); ++j) {
	// 			if (add_to_cloud[j] == 1) {
	// 				global_pcloud.push_back(new_triangulated[j]);
	// 			}
	// 		}
	// 	}
	// 	good_views.insert(i);
	// 	Mat cam_matrix = K;
	// 	adjustBundle(cam_matrix);
	}
}

void triangulate2Views(int first_view, int second_view) {

	//getting baseline reconstruction from 2 view
	list<pair<int,pair<int,int>>> percent_matches;
	sortMatchesFromHomography(percent_matches);
	vector<Point3DInMap> cloud;

	cout << "\nGetting baseline triangulation......" << endl;
	for (auto it = percent_matches.begin(); it != percent_matches.end(); ++it) {
		int i = it->second.first;
		int j = it->second.second;
		first_view  = i;
		second_view = j;
		cout << "image [" << i << ", " << j << "] with " << it->first << "%" << endl;
		bool sfm_res = computeSFM(i, j, cloud);
		if (sfm_res) {
			cout << "successful sfm with image: [" << i << ", " << j << "]" << endl;
			break;
		} else {
			cout << "failed sfm with image: [" << i << ", " << j << "]\n\n" << endl;
		}
	}
	//cout << "\n...after baseline triangulation found: " << global_pcloud.size() << " points.\n" << endl;
	cout << "\n...after baseline triangulation found: " << cloud.size() << " points.\n" << endl;
	
	globalPoints = cloud;
	done_views.insert(first_view);
	done_views.insert(second_view);
	good_views.insert(first_view);
	good_views.insert(second_view);

	cout << "\n -- Starting Bundle Adjustment -- \n" << endl;
	Mat cam_matrix = K;
	adjustBundle(cam_matrix);
	cout << "...finished bundle adjustment\n" << endl;
	K = cam_matrix;
}

bool estimatePose(vector<Point3f>& points3d, vector<Point2f>& points2d, Matx34d& Pnew) {

	Mat rvec, tvec;
	Mat inliers;
	solvePnPRansac(points3d, points2d, K, distanceCoeffs, rvec, tvec, false, 100, 
		10.0f, 0.99, inliers);
	cout << "inliers size: " << inliers.rows << endl;
	cout << "points2d size: " << points2d.size() << endl;
	if (((float)countNonZero(inliers) / (float)points2d.size()) < 0.5) {
		cout << "error: inliers ratio is too small in pose estimation" << endl;
		cout << "ratio was: " << ((float)countNonZero(inliers) / (float)points2d.size()) << endl;
		return false;
	}
	Mat rot;
	Rodrigues(rvec, rot);
	rot.copyTo(Mat(3, 4, CV_32FC1, Pnew.val)(ROT));
	tvec.copyTo(Mat(3, 4, CV_32FC1, Pnew.val)(TRA));
	return true;

	// if (cloud.size() <= 7 || imgPoints.size() <= 7 || cloud.size() != imgPoints.size()) {
	// 	cout << "err: not enough points to find pose." << endl;
	// 	return false;
	// }
	// vector<int> inliers;
	// double min, max;
	// minMaxIdx(imgPoints, &min, &max);
	// solvePnPRansac(cloud, imgPoints, K, distanceCoeffs,	rvec, t, false, 
	// 	1000, 0.06 * max, 0.99, inliers, CV_P3P);
	// vector<Point2f> projected3D;
	// projectPoints(cloud, rvec, t, K, distanceCoeffs, projected3D);
	// cout << "inliers size: " << inliers.size() << endl;
	// if (inliers.size() == 0) {
	// 	for (int i = 0; i < projected3D.size(); ++i) {
	// 		if (norm(projected3D[i] - imgPoints[i]) < 10.0) {
	// 			inliers.push_back(i);
	// 		}
	// 	}
	// }
	// if (inliers.size() < (double)(imgPoints.size()) / 5.0) {
	// 	cout << "err: not enough inliers to find good pose." << endl;
	// 	return false;
	// }
	// if (norm(t) > 200.0) {
	// 	cout << "err: camera movement is to big. " << endl;
	// 	return false;
	// }
	// Rodrigues(rvec, R);
	// if (!checkRotationMat(R)) {
	// 	cout << "err: not a coherent rotation to find good pose." << endl;
	// 	return false;
	// }
	// return true;
}

/**
* this function finds the points that have a 3d coordinate in the global point cloud
* and an image coordinate in the keypoints that correspond.
* and further refines the cloud point.
*/ 
void Find2D3DCorrespondences() {

// int curr_view, vector<Point3f>& cloud, 
// 	vector<Point2f>& imgPoints

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


	// cloud.clear();
	// imgPoints.clear();
	// vector<int> cloud_status(global_pcloud.size(), 0);
	// for (auto done_view = good_views.begin(); done_view != good_views.end(); ++done_view) {
	// 	int old_view = *done_view;
	// 	vector<DMatch> matches = all_matches[make_pair(curr_view, old_view)];
	// 	for (int match = 0; match < matches.size(); ++match) {
	// 		int old_index = matches[match].queryIdx;
	// 		for (int point = 0; point < global_pcloud.size(); ++point) {
	// 			if (old_index == global_pcloud[point].imgpt_for_img[old_view]
	// 				&& cloud_status[point] == 0) {
	// 				cloud.push_back(global_pcloud[point].pt);
	// 				imgPoints.push_back(all_keypoints[curr_view][matches[match].trainIdx].pt);
	// 				cloud_status[point] = 1;
	// 				break;
	// 			}
	// 		}
	// 	}
	// }
}

void adjustBundle(Mat& cam_matrix) {
	int N = all_pmats.size();
	// int M = global_pcloud.size();
	// int K = get2DMeasurements(global_pcloud);
	int M = globalPoints.size();
	int K = get2DMeasurements(globalPoints);
	StdDistortionFunction distortion;

	Matrix3x3d KMat;
	makeIdentityMatrix(KMat);
	KMat[0][0] = cam_matrix.at<double>(0,0); //fx
	KMat[1][1] = cam_matrix.at<double>(1,1); //fy
	KMat[0][1] = cam_matrix.at<double>(0,1); //skew
	KMat[0][2] = cam_matrix.at<double>(0,2); //ppx
	KMat[1][2] = cam_matrix.at<double>(1,2); //ppy

	double const f0 = KMat[0][0];
	if (show) {
		cout << "before bundle adjustment: " << endl;
		displayMatrix(KMat);
	}

	Matrix3x3d Knorm = KMat;
	scaleMatrixIP(1.0/f0, Knorm);
	Knorm[2][2] = 1.0;
	
	vector<int> pointIdFwdMap(M);
	map<int, int> pointIdBwdMap;
	
	//conver 3D point cloud to BA datastructs
	vector<Vector3d > Xs(M);
	for (int i = 0; i < M; ++i) {
		int pointId = i;
		// Xs[i][0] = global_pcloud[i].pt.x;
		// Xs[i][1] = global_pcloud[i].pt.y;
		// Xs[i][2] = global_pcloud[i].pt.z;
		Xs[i][0] = globalPoints[i].p.x;
		Xs[i][1] = globalPoints[i].p.y;
		Xs[i][2] = globalPoints[i].p.z;
		pointIdFwdMap[i] = pointId;
		pointIdBwdMap.insert(make_pair(pointId, i));
	}
	vector<int> camIdFwdMap(N,-1);
	map<int, int> camIdBwdMap;
	
	//convert cameras to BA datastructs
	vector<CameraMatrix> cams(N);
	for (int i = 0; i < N; ++i) {
		int camId = i;
		Matrix3x3d R;
		Vector3d T;

		Matx34d& P = all_pmats[i];
		R[0][0] = P(0,0); R[0][1] = P(0,1); R[0][2] = P(0,2); T[0] = P(0,3);
		R[1][0] = P(1,0); R[1][1] = P(1,1); R[1][2] = P(1,2); T[1] = P(1,3);
		R[2][0] = P(2,0); R[2][1] = P(2,1); R[2][2] = P(2,2); T[2] = P(2,3);
		
		camIdFwdMap[i] = camId;
		camIdBwdMap.insert(make_pair(camId, i));
		
		cams[i].setIntrinsic(Knorm);
		cams[i].setRotation(R);
		cams[i].setTranslation(T);
	}

	vector<Vector2d > measurements;
	vector<int> correspondingView;
	vector<int> correspondingPoint;
	
	measurements.reserve(K);
	correspondingView.reserve(K);
	correspondingPoint.reserve(K);
	// for (int i = 0; i < global_pcloud.size(); ++i) {
	// 	for (int j = 0; j < global_pcloud[i].imgpt_for_img.size(); ++j) {
	// 		if (global_pcloud[i].imgpt_for_img[j] >= 0) {
	// 			int view = j;
	// 			int point = i;
	// 			Vector3d p;
	// 			Point cvp = all_keypoints[j][global_pcloud[i].imgpt_for_img[j]].pt;
	// 			p[0] = cvp.x;
	// 			p[1] = cvp.y;
	// 			p[2] = 1.0;

	// 			if (camIdBwdMap.find(view) != camIdBwdMap.end() &&
	// 				pointIdBwdMap.find(point) != pointIdBwdMap.end()) {
	// 				// Normalize the measurements to match the unit focal length.
	// 				scaleVectorIP(1.0/f0, p);
	// 				measurements.push_back(Vector2d(p[0], p[1]));
	// 				correspondingView.push_back(camIdBwdMap[view]);
	// 				correspondingPoint.push_back(pointIdBwdMap[point]);
	// 			}
	// 		}
	// 	}
	// }

	for (int i = 0; i < globalPoints.size(); ++i) {
		for (int j = 0; j < globalPoints[i].originatingViews.size(); ++j) {
			int view = j;
			int point = i;
			Vector3d p;
			Point cvp = all_keypoints[j][globalPoints[i].originatingViews[j]].pt;
			p[0] = cvp.x;
			p[1] = cvp.y;
			p[2] = 1.0;

			if (camIdBwdMap.find(view) != camIdBwdMap.end() &&
				pointIdBwdMap.find(point) != pointIdBwdMap.end()) {
				// Normalize the measurements to match the unit focal length.
				scaleVectorIP(1.0/f0, p);
				measurements.push_back(Vector2d(p[0], p[1]));
				correspondingView.push_back(camIdBwdMap[view]);
				correspondingPoint.push_back(pointIdBwdMap[point]);
			}
		}
	}

	K = measurements.size();
	double const inlierThreshold = 2.0 / fabs(f0);
	Matrix3x3d K0 = cams[0].getIntrinsic();
	if (show) {
		cout << "K0 before: " << endl;
		displayMatrix(K0);
	}

	bool  good_adjustment = false;
	{
		ScopedBundleExtrinsicNormalizer extNorm(cams, Xs);
		ScopedBundleIntrinsicNormalizer intNorm(cams,measurements,correspondingView);
		CommonInternalsMetricBundleOptimizer opt(V3D::FULL_BUNDLE_FOCAL_LENGTH_PP, inlierThreshold, K0, distortion, cams, Xs,
												 measurements, correspondingView, correspondingPoint);
		
		opt.tau = 1e-3;
		opt.maxIterations = 50;
		opt.minimize();
		
		cout << "optimizer status = " << opt.status << endl;
		
		good_adjustment = (opt.status != 2);
	}
	if (show) {
		cout << "K0 after: " << endl;
		displayMatrix(K0);
	}
	for (int i = 0; i < N; ++i) {
		cams[i].setIntrinsic(K0);
	}

	Matrix3x3d Knew = K0;
	scaleMatrixIP(f0, Knew);
	Knew[2][2] = 1.0;
	if (show) {
		cout << "Knew: " << endl;
		displayMatrix(Knew);
	}

	if (good_adjustment) {
		// for (int i = 0; i < Xs.size(); ++i) {
		// 	global_pcloud[i].pt.x = Xs[i][0];
		// 	global_pcloud[i].pt.y = Xs[i][1];
		// 	global_pcloud[i].pt.z = Xs[i][2];
		// }
		for (int i = 0; i < Xs.size(); ++i) {
			globalPoints[i].p.x = Xs[i][0];
			globalPoints[i].p.y = Xs[i][1];
			globalPoints[i].p.z = Xs[i][2];
		}
		for (int i = 0; i < N; ++i) {
			Matrix3x3d R = cams[i].getRotation();
			Vector3d T = cams[i].getTranslation();

			Matx34d P;
			P(0,0) = R[0][0]; P(0,1) = R[0][1]; P(0,2) = R[0][2]; P(0,3) = T[0];
			P(1,0) = R[1][0]; P(1,1) = R[1][1]; P(1,2) = R[1][2]; P(1,3) = T[1];
			P(2,0) = R[2][0]; P(2,1) = R[2][1]; P(2,2) = R[2][2]; P(2,3) = T[2];
			
			all_pmats[i] = P;
		}
		cam_matrix.at<double>(0,0) = Knew[0][0];
		cam_matrix.at<double>(0,1) = Knew[0][1];
		cam_matrix.at<double>(0,2) = Knew[0][2];
		cam_matrix.at<double>(1,1) = Knew[1][1];
		cam_matrix.at<double>(1,2) = Knew[1][2];
	}
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

int get2DMeasurements(const vector<CloudPoint>& global_pcloud) {
	int count = 0;
	for (int i = 0; i < global_pcloud.size(); ++i) {
		for (int j = 0; j < global_pcloud[i].imgpt_for_img.size(); ++j) {
			if (global_pcloud[i].imgpt_for_img[j] >= 0) {
				count++;
			}	
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

		all_points.push_back(p);
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
			// if (global_pcloud[i].imgpt_for_img[j] != -1) {
			// 	int pt_idx = global_pcloud[i].imgpt_for_img[j];
			// 	if (pt_idx >= all_keypoints[j].size()) {
			// 		cout << "err: index out of bounds while getting point RGB" << endl;
			// 		continue;
			// 	}
			// 	Point p = all_keypoints[j][pt_idx].pt;
			// 	point_colors.push_back(imagesColored[j].at<Vec3b>(p));
			// }
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
bool computeSFM(int idx1, int idx2, vector<Point3DInMap>& cloud) {

	vector<KeyPoint> kpts1 = all_keypoints[idx1];
	vector<KeyPoint> kpts2 = all_keypoints[idx2];
	int image_size = images.size();

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
	Mat E, R, t;
	Mat mask;
	double f0 = K.at<double>(0, 0);
	Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));
	E = findEssentialMat(points1, points2, f0, pp, CV_RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points1, points2, R, t, f0, pp, mask);
	cout << "done recover pose " << endl;
	Matx34d P1 = Matx34d::eye();
	Matx34d P2 = Matx34d(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                    R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
					R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
	all_pmats[idx1] = P1;
	all_pmats[idx2] = P2;

	//triangulate the points and create point cloud.
	vector<CloudPoint> tri_pts;
	vector<int> add_to_cloud;
	vector<KeyPoint> correspondingImg1Pt;


	cout << "before triangulation: " << cloud.size() << endl;
	bool tri_res = triangulateBetweenViews(tri_pts, add_to_cloud, correspondingImg1Pt,
		P1, P2, cloud, idx1, idx2);
		// if (tri_res && countNonZero(add_to_cloud) >= 20) {
		// 	// for (int j = 0; j < add_to_cloud.size(); ++j) {
		// 	// 	if (add_to_cloud[j] == 1) {
		// 	// 		global_pcloud.push_back(tri_pts[j]);
		// 	// 	}
		// 	// }
		// 	cout << "after triangulation: " << globalPoints.size() << endl;
		// } else {
		// 	cout << "triangulation failed or not enough points were added" << endl;
		// 	return false;
		// }
	cout << "after triangulation: " << cloud.size() << endl;		

	//prune the matches based on the inliers.
	vector<DMatch> new_matches;
	for (int i = 0; i < mask.rows; ++i) {
		if (mask.at<uchar>(i)) {
			new_matches.push_back(matches[i]);
		}
	}
	all_matches[make_pair(idx1, idx2)] = new_matches;
	vector<DMatch> flipedMatches;
	flipMatches(new_matches, flipedMatches);
	all_matches[make_pair(idx2, idx1)] = flipedMatches;
	cout << "new matches size: " << new_matches.size() << endl;

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


	if (show) {
		cout << "found : " << matches.size() << " matches" << endl;
	}

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

	// now we allign the matched points.
	vector<KeyPoint> pts1_temp, pts2_temp;
	allignPoints(kpts1, kpts2, good_matches, pts1_temp, pts2_temp);
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
	if (show) {
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

	
	if (show) {
		string name = "matches" + to_string(idx1) + "_" + to_string(idx2) + ".jpg";
 		imwrite(name, img_matches);
		cout << "image saved" << endl;
	}
	all_matches[make_pair(idx1, idx2)] = good_matches;
	vector<DMatch> flipedMatches;
	flipMatches(good_matches, flipedMatches);
	all_matches[make_pair(idx2, idx1)] = flipedMatches;
	cout << "matches size: " << matches.size() << endl;

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
double TriangulatePoints(const vector<KeyPoint>& kpts_good, const Matx34d& P1, 
	const Matx34d& P2, vector<CloudPoint>& pointCloud,	
	vector<KeyPoint>& correspondingImg1Pt) {

	if (show) {
		cout << "starting TriangulatePoints.. " << endl;
	}

	Mat Kinv = K.inv();
	vector<double> reprojectionError;
	int size = kpts_good.size();
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
		Point2f kp1 = kpts_good[i].pt;
		Point3d u1(kp1.x, kp1.y, 1.0);

		//point from image 2
		Point2f kp2 = kpts_good[i].pt;
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
		correspondingImg1Pt.push_back(kpts_good[i]);
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
* this function alligns keypoints between two images.
* parameters: takes in the keypoints for the first and second image.
* 	a vector of matches.
* output: 2 vectors of alligned keypoints between the images.
*/
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

/**
* this function alligns keypoints between two images.
* parameters: takes in the keypoints for the first and second image.
* 	a vector of matches.
* output: 2 vectors of alligned keypoints between the images.
*/
void allignPoints(const vector<KeyPoint>& imgpts1, const vector<KeyPoint>& imgpts2,
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
bool triangulateBetweenViews(vector<CloudPoint>& tri_pts, vector<int>& add_to_cloud,
	vector<KeyPoint>& correspondingImg1Pt, const Matx34d& P1, const Matx34d& P2,
	vector<Point3DInMap>& cloud, int idx1, int idx2) {

	const float MIN_REPROJECTION_ERROR = 10.0;

	vector<int> leftBackReference;
	vector<int> rightBackReference;
	vector<KeyPoint> kpts1 = all_keypoints[idx1];
	vector<KeyPoint> kpts2 = all_keypoints[idx2];
	vector<KeyPoint> kpts1_temp, kpts2_temp;
	vector<DMatch> matches = all_matches[make_pair(idx1, idx2)];
	vector<Point2f> points1, points2;

	allignPoints(kpts1, kpts2, matches, kpts1_temp, kpts2_temp, 
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

	// Mat Kinv = K.inv();
 // 	vector<DMatch> good_matches = all_matches[make_pair(idx1, idx2)];
 // 	Matx34d P1 = all_pmats[idx1];
 // 	Matx34d P2 = all_pmats[idx2];
 // 	vector<KeyPoint> kpts_good = all_good_keypoints[make_pair(idx1, idx2)];
 // 	int image_size = images.size();
	
	// double err = TriangulatePoints(kpts_good, P1, P2, tri_pts, correspondingImg1Pt);
	// if (show) {
	// 	cout << "reprojection error was: " << err << endl;
	// 	cout << "\ntri_pts size: " << tri_pts.size() << endl;
	// }
	// vector<uchar> trig_status;
	// if (!testTriangulation(tri_pts, P1, trig_status) 
	// 	|| !testTriangulation(tri_pts, P2, trig_status)) {
	// 	cout << "err: triangulation failed" << endl;
	// 	return false;
	// }

	// vector<double> reproj_error;
	// for (int i = 0; i < tri_pts.size(); ++i) {
	// 	reproj_error.push_back(tri_pts[i].reprojection_error);
	// }
	// sort(reproj_error.begin(), reproj_error.end());
	// double err_cutoff = reproj_error[4 * reproj_error.size() / 5] * 2.4;

	// vector<CloudPoint> new_triangulated_filtered;
	// vector<DMatch> new_matches;
	// for (int i = 0; i < tri_pts.size(); ++i) {
	// 	if (trig_status[i] == 0) {
	// 		continue;
	// 	}
	// 	if (tri_pts[i].reprojection_error > 16.0) {
	// 		continue;
	// 	}
	// 	if (tri_pts[i].reprojection_error < 4.0 ||
	// 		tri_pts[i].reprojection_error < err_cutoff) {
	// 		new_triangulated_filtered.push_back(tri_pts[i]);
	// 		new_matches.push_back(good_matches[i]);
	// 	} else {
	// 		continue;
	// 	}
	// }

	// if (show) {
	// 	cout << "# points that are triangulated: " << new_triangulated_filtered.size() << endl;
	// }
	// if (new_triangulated_filtered.size() <= 0) {
	// 	cout << "err: could not find enough points" << endl;
	// 	return false;
	// }
	// tri_pts = new_triangulated_filtered;
	// good_matches = new_matches;
	// add_to_cloud.clear();
	// add_to_cloud.resize(tri_pts.size(), 1);
	// int found_in_other_views = 0;
	// for (int j = 0; j < tri_pts.size(); ++j) {

	// 	tri_pts[j].imgpt_for_img = vector<int>(image_size, -1);
	// 	tri_pts[j].imgpt_for_img[idx1] = good_matches[j].queryIdx;
	// 	tri_pts[j].imgpt_for_img[idx2] = good_matches[j].trainIdx;
	// 	bool found = false;
	// 	for (unsigned int image = 0; image < image_size; ++image) {
	// 		if (image != idx1) {
	// 			//get the matches of all other points.
	// 			vector<DMatch> submatches = all_matches[make_pair(image, idx2)];
	// 			for (int k = 0; k < submatches.size(); ++k) {
	// 				if (submatches[k].trainIdx == good_matches[j].trainIdx &&
	// 					!found) {
	// 					for (unsigned int pt3d = 0; pt3d < global_pcloud.size(); ++pt3d) {
	// 						if (global_pcloud[pt3d].imgpt_for_img[image] == submatches[k].queryIdx) {
	// 							global_pcloud[pt3d].imgpt_for_img[idx1] = good_matches[j].trainIdx;
	// 							global_pcloud[pt3d].imgpt_for_img[idx2] = good_matches[j].queryIdx;
	// 							found = true;
	// 							add_to_cloud[j] = 0;
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}

	// 	if (found) {
	// 		found_in_other_views++;
	// 	} else {
	// 		add_to_cloud[j] = 1;		
	// 	}
	// }
	// cout << "add_to_cloud size: " << countNonZero(add_to_cloud) << endl;
	// return true;
}

void sortMatchesFromHomography(list<pair<int, pair<int,int>>>& percent_matches) {

	for (auto it = all_matches.begin(); it != all_matches.end(); ++it) {
		if ((*it).second.size() < 100) {
			percent_matches.push_back(make_pair(100, (*it).first));
		} else {
			vector<KeyPoint> keypts1, keypts2;
			vector<Point2f> pts1, pts2;
			int idx1 = (*it).first.first;
			int idx2 = (*it).first.second;

			allignPoints(all_keypoints[idx1], all_keypoints[idx2], all_matches[make_pair(idx1, idx2)],
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
			if (show) {
				cout << "percentage inliers for: " << idx1 << ", " << idx2 << ": " 
				<< percent <<  endl;
			}
		}
	}
	percent_matches.sort(sortFromPercentage);
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

void autoCalibrate() {
	cout << "--auto calibrating--\n" << endl;
	K = (Mat_<float>(3,3) << 2500,   0, images[0].cols / 2,
            				 0, 2500, images[0].rows / 2,
							 0, 0, 1);
}