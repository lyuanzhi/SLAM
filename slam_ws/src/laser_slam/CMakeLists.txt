cmake_minimum_required(VERSION 2.8.3)
project(laser_slam)

SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -std=c++0x -std=c++14 -fexceptions -Wno-unused-local-typedefs")

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  visualization_msgs
  )

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(octomap REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)

include_directories(
	${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs visualization_msgs
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS
)

add_executable(tracking1 src/tracking1.cpp)
target_link_libraries(tracking1 ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES} ${Pangolin_LIBRARIES} yaml-cpp)

add_executable(mapping1 src/mapping1.cpp)
target_link_libraries(mapping1 ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES} ${Pangolin_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp gtsam)

