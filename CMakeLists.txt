set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0015 NEW)

project(myproject)
add_executable(3drecon src/3dReconstruction.hpp src/3dReconstruction.cpp src/CameraCalib.hpp)

find_package(PCL 1.7.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

#PCL
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

#OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

#ceres-solver
include_directories(${CERES_INCLUDE_DIRS})
link_directories(${CERES_LIB_DIR})

target_link_libraries (3drecon ${PCL_LIBRARIES})
target_link_libraries (3drecon ${OpenCV_LIBS})
target_link_libraries (3drecon ${CERES_LIBRARIES})