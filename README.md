                                            Description

This project recreates a 3d scene given a series of images from a calibrated camera using Structure From Motion techniques. The camera's intrinsics must be included in the calibinfo.yml, otherwise a folder with calibration images must be given for manual calibration. if nothing is provided auto calibration will take place but the intrinsics will only be an estimate.


                                            Building/Usage

Note: this project requires OpenCV, PCL, and SSBA libraries to build.                                            

mkdir build
cd build
cmake ..
make

then either:
./3drecon <image directory>
./3drecon <image directory> <1 for debug info>
./3drecon <image directory> <calibration image path> <1 or 0 for debug info>

											Examples
Some examples of reconstructions from the res folder:

1- testPics (contains 3 images of a russian doll):
![russianDoll](https://raw.githubusercontent.com/ShayaP/3DReconstruction/master/result1.png)
![russianDoll](https://raw.githubusercontent.com/ShayaP/3DReconstruction/master/result2.png)
                                            Credit

Multiple View Geometry by Hartley and Zisserman<br/>
Mastering OpenCV with Practical Computer Vision Projects

