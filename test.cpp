#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {

	//vector used to hold the images
	vector<Mat> images;

	if (argc != 2) {
		cout << "wrong number of arguments" << endl;
		return -1;
	}

	//access the image folder, and read each image into images.
	DIR *dir;
	struct dirent *entity;
	if (dir=opendir(argv[1])) {
		while(entity=readdir(dir)) {
			if (entity == NULL) {
				cout << "Could not read the directory." << endl;
				break;
			} else if (((string)entity->d_name).find(".jpg")) {
				//insert the images into the vector
				string path;
				path.append(argv[1]);
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

	cout << images.size() << endl;

	return 0;
}
