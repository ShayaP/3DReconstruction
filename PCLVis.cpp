#include "PCLVis.hpp"

using namespace std;
using namespace pcl;
using namespace pcl::visualization;

void simpleVis(vector<CloudPoint>& cloud) {
	PointCloud<PointXYZ>::Ptr cloud_ptr (new PointCloud<PointXYZ>);
	for (int i = 0; i < cloud.size(); ++i) {
		PointXYZ p;
		p.x = cloud[i].pt.x;
		p.y = cloud[i].pt.y;
		p.z = cloud[i].pt.z;
		cloud_ptr->points.push_back(p); 
	}
	cloud_ptr->width = (int) cloud_ptr->points.size();
	cloud_ptr->height = 1;
	CloudViewer viewer("CloudViewer");
	viewer.showCloud(cloud_ptr);
}