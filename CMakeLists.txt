set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(myproject)
add_executable(3drecon 3dReconstruction.hpp 3dReconstruction.cpp CameraCalib.hpp)

find_package(PCL 1.7.2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/SSBA-3.0")
add_subdirectory (SSBA-3.0)
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIB_DIR})

set (EXTRA_LIBS ${EXTRA_LIBS} SSBA-3.0)

target_link_libraries (3drecon ${PCL_LIBRARIES})
target_link_libraries (3drecon ${OpenCV_LIBS})
target_link_libraries (3drecon ${SSBA-3.0})