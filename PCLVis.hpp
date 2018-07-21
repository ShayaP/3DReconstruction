#ifndef PCLVIS_HPP
#define PCLVIS_HPP

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>


#include <iostream>
#include <vector>

#include "3dReconstruction.hpp"
#include "CameraCalib.hpp"

void simpleVis(vector<CloudPoint>& cloud);

#endif