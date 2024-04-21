// TODO: scan-context
// TODO: 处理 回环检测时多个当前帧对应到同一个历史帧的情况

#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include <opencv/cv.h>
#include <iostream>
#include <chrono>
#include <iostream>
#include <octomap/octomap.h>
#include <queue>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <pangolin/pangolin.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <pcl/registration/icp.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
using namespace std;
using namespace gtsam;
typedef pcl::PointXYZI PointType;
ros::Publisher pub_global_map;
ros::Publisher pub_optimized_pose;
ros::Subscriber sub_local_map;
ros::Subscriber sub_feature;
ros::Subscriber sub_unoptimized_pose;

// debug pub
ros::Publisher pub_closure_his_map;
ros::Publisher pub_closure_his_pose;
ros::Publisher pub_closure_cur_map;
ros::Publisher pub_closure_cur_pose;

vector<pcl::PointCloud<PointType>::Ptr> localMaps;
vector<pcl::PointCloud<PointType>::Ptr> features;
pcl::PointCloud<PointType>::Ptr globalMaps(new pcl::PointCloud<PointType>());
// 2 0 1 3 4 5
vector<vector<double>> unoptPoses; // Pitch Yaw Roll x y z time
vector<vector<double>> optPoses;
pcl::PointCloud<PointType>::Ptr unoptPosesMap(new pcl::PointCloud<PointType>()); // x y z
pcl::KdTreeFLANN<PointType>::Ptr kdtreeFLANN(new pcl::KdTreeFLANN<PointType>());
pcl::VoxelGrid<PointType> downSizeFilter;
double localMapsTime = 0;
double unoptPosesTime = 0;
double featuresTime = 0;
mutex mtx;
int latestIndex = -1;

double closure_search_radius = 0;
double sec_difference = 0;
int closure_search_num = 0;
double resolution = 0;
double resolutionICP = 0;
double score_thre = 0;
int show_size = 0;
int show_score = 0;
int show_closure_points_compare = 0;
int test_num = 0;

void getYamlParamters(string path);
void main_thread();
void localMapHandler(const sensor_msgs::PointCloud2ConstPtr &msg);
void unoptPoseHandler(const nav_msgs::Odometry::ConstPtr &msg);
void featureHandler(const sensor_msgs::PointCloud2ConstPtr &msg);
void showDoubleVector(vector<double> &v);
void debugTopicPub(int curID, int hisID);
void camToWorld(PointType const *const in, PointType *const out, vector<double> &pose);

int main(int argc, char **argv)
{
    getYamlParamters(ros::package::getPath("laser_slam") + "/config/mapping1.yaml");
    ros::init(argc, argv, "mapping1");
    ros::NodeHandle handle;
    pub_global_map = handle.advertise<sensor_msgs::PointCloud2>("/globalmap", 1);
    pub_optimized_pose = handle.advertise<geometry_msgs::PoseArray>("/optpose", 1);

    // debug pub
    pub_closure_his_map = handle.advertise<sensor_msgs::PointCloud2>("/closure_his_map", 1);
    pub_closure_his_pose = handle.advertise<nav_msgs::Odometry>("/closure_his_pose", 1);
    pub_closure_cur_map = handle.advertise<sensor_msgs::PointCloud2>("/closure_cur_map", 1);
    pub_closure_cur_pose = handle.advertise<nav_msgs::Odometry>("/closure_cur_pose", 1);

    sub_local_map = handle.subscribe<sensor_msgs::PointCloud2>("/cloud", 100, localMapHandler);
    sub_feature = handle.subscribe<sensor_msgs::PointCloud2>("/feature", 100, featureHandler);
    sub_unoptimized_pose = handle.subscribe<nav_msgs::Odometry>("/pose", 100, unoptPoseHandler);
    boost::thread server(main_thread);
    ros::spin();
    return 0;
}

void getYamlParamters(string path)
{
    YAML::Node config = YAML::LoadFile(path);
    closure_search_radius = config["closure_search_radius"].as<double>();
    sec_difference = config["sec_difference"].as<double>();
    closure_search_num = config["closure_search_num"].as<int>();
    resolution = config["resolution"].as<double>();
    resolutionICP = config["resolutionICP"].as<double>();
    score_thre = config["score_thre"].as<double>();
    show_size = config["show_size"].as<int>();
    show_score = config["show_score"].as<int>();
    show_closure_points_compare = config["show_closure_points_compare"].as<int>();
    test_num = config["test_num"].as<int>();
    cout << "closure_search_radius:" << closure_search_radius << endl;
    cout << "sec_difference:" << sec_difference << endl;
    cout << "closure_search_num:" << closure_search_num << endl;
    cout << "resolution:" << resolution << endl;
    cout << "resolutionICP:" << resolutionICP << endl;
    cout << "score_thre:" << score_thre << endl;
    cout << "show_size:" << show_size << endl;
    cout << "show_score:" << show_score << endl;
    cout << "show_closure_points_compare:" << show_closure_points_compare << endl;
    cout << "test_num:" << test_num << endl;
}

void main_thread()
{
    octomap::OcTree tree(resolution);
    ISAM2Params gtsam_parameters;
    gtsam_parameters.relinearizeThreshold = 0.01;
    gtsam_parameters.relinearizeSkip = 1;
    ISAM2 isam(gtsam_parameters);
    NonlinearFactorGraph graph;
    Values initialEstimate;
    Values currentEstimate;
    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    priorNoise = noiseModel::Diagonal::Variances(Vector6);
    odometryNoise = noiseModel::Diagonal::Variances(Vector6);

    ros::Rate rate(20);
    vector<int> pointSearchIndLoop;
    vector<float> pointSearchSqDisLoop;
    while (ros::ok())
    {
        rate.sleep();
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        mtx.lock();
        if (localMaps.size() == 0 || unoptPoses.size() == 0 || localMapsTime != unoptPosesTime || localMapsTime != featuresTime || featuresTime != unoptPosesTime)
        {
            mtx.unlock();
            // cout << "no data!" << endl;
            continue;
        }
        latestIndex++;
        if (latestIndex > unoptPoses.size() - 1)
            latestIndex = unoptPoses.size() - 1;
        mtx.unlock();

        // build pose graph
        if (unoptPoses[latestIndex][7] == 0)
        {
            if (latestIndex == 0)
            {
                graph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(unoptPoses[latestIndex][2], unoptPoses[latestIndex][0], unoptPoses[latestIndex][1]), Point3(unoptPoses[latestIndex][3], unoptPoses[latestIndex][4], unoptPoses[latestIndex][5])), priorNoise));
                initialEstimate.insert(0, Pose3(Rot3::RzRyRx(unoptPoses[latestIndex][2], unoptPoses[latestIndex][0], unoptPoses[latestIndex][1]), Point3(unoptPoses[latestIndex][3], unoptPoses[latestIndex][4], unoptPoses[latestIndex][5])));
            }
            else
            {
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(unoptPoses[latestIndex - 1][2], unoptPoses[latestIndex - 1][0], unoptPoses[latestIndex - 1][1]), Point3(unoptPoses[latestIndex - 1][3], unoptPoses[latestIndex - 1][4], unoptPoses[latestIndex - 1][5]));
                gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(unoptPoses[latestIndex][2], unoptPoses[latestIndex][0], unoptPoses[latestIndex][1]), Point3(unoptPoses[latestIndex][3], unoptPoses[latestIndex][4], unoptPoses[latestIndex][5]));
                graph.add(BetweenFactor<Pose3>(latestIndex - 1, latestIndex, poseFrom.between(poseTo), odometryNoise));
                initialEstimate.insert(latestIndex, Pose3(Rot3::RzRyRx(unoptPoses[latestIndex][2], unoptPoses[latestIndex][0], unoptPoses[latestIndex][1]), Point3(unoptPoses[latestIndex][3], unoptPoses[latestIndex][4], unoptPoses[latestIndex][5])));
            }
            isam.update(graph, initialEstimate);
            isam.update();
            graph.resize(0);
            initialEstimate.clear();
            unoptPoses[latestIndex][7] = 1;
        }

        // detect loop closure
        kdtreeFLANN->setInputCloud(unoptPosesMap);
        kdtreeFLANN->radiusSearch(unoptPosesMap->points[latestIndex], closure_search_radius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        int closestHistoryFrameID = -1;
        for (int i = 0; i < pointSearchIndLoop.size(); ++i)
        {
            if (unoptPoses[latestIndex][6] - unoptPoses[pointSearchIndLoop[i]][6] > sec_difference)
            {
                closestHistoryFrameID = pointSearchIndLoop[i];
                break;
            }
        }

        if (closestHistoryFrameID != -1)
        {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);
            int isReverse = 0;

            pcl::PointCloud<PointType>::Ptr temp1(new pcl::PointCloud<PointType>());
            downSizeFilter.setLeafSize(resolutionICP, resolutionICP, resolutionICP);
            downSizeFilter.setInputCloud(localMaps[closestHistoryFrameID]);
            downSizeFilter.filter(*temp1);
            localMaps[closestHistoryFrameID] = temp1;

            pcl::PointCloud<PointType>::Ptr temp2(new pcl::PointCloud<PointType>());
            downSizeFilter.setLeafSize(resolutionICP, resolutionICP, resolutionICP);
            downSizeFilter.setInputCloud(localMaps[latestIndex]);
            downSizeFilter.filter(*temp2);
            localMaps[latestIndex] = temp2;

            if (show_size)
            {
                cout << "index: " << latestIndex << "; Source: " << localMaps[latestIndex]->size() << endl;
                cout << "index: " << closestHistoryFrameID << "; Target: " << localMaps[closestHistoryFrameID]->size() << endl;
            }
            if (localMaps[latestIndex]->size() > localMaps[closestHistoryFrameID]->size())
            {
                icp.setInputSource(localMaps[closestHistoryFrameID]);
                icp.setInputTarget(localMaps[latestIndex]);
                isReverse = 1;
            }
            else
            {
                icp.setInputSource(localMaps[latestIndex]);
                icp.setInputTarget(localMaps[closestHistoryFrameID]);
                isReverse = 0;
            }
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            if (icp.hasConverged() == true && icp.getFitnessScore() < score_thre)
            {
                if (show_score)
                {
                    cout << "index: " << latestIndex << ", " << closestHistoryFrameID << "; score: " << icp.getFitnessScore() << endl;
                }
                if (show_closure_points_compare)
                {
                    debugTopicPub(latestIndex, closestHistoryFrameID);
                }

                float x, y, z, roll, pitch, yaw;
                Eigen::Affine3f correction;
                correction = icp.getFinalTransformation();
                if (isReverse)
                {
                    pcl::getTranslationAndEulerAngles(correction, x, y, z, roll, pitch, yaw);
                    correction = pcl::getTransformation(-x, -y, -z, -roll, -pitch, -yaw);
                }
                Eigen::Affine3f tWrong = pcl::getTransformation(unoptPoses[latestIndex][3], unoptPoses[latestIndex][4], unoptPoses[latestIndex][5], unoptPoses[latestIndex][2], unoptPoses[latestIndex][0], unoptPoses[latestIndex][1]);
                Eigen::Affine3f tCorrect = correction * tWrong;
                pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(unoptPoses[closestHistoryFrameID][2], unoptPoses[closestHistoryFrameID][0], unoptPoses[closestHistoryFrameID][1]), Point3(unoptPoses[closestHistoryFrameID][3], unoptPoses[closestHistoryFrameID][4], unoptPoses[closestHistoryFrameID][5]));
                gtsam::Vector Vector6_1(6);
                float noiseScore = icp.getFitnessScore();
                Vector6_1 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
                constraintNoise = noiseModel::Diagonal::Variances(Vector6_1);
                graph.add(BetweenFactor<Pose3>(latestIndex, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
                isam.update(graph);
                isam.update();
                graph.resize(0);
            }
        }

        optPoses.clear();
        currentEstimate = isam.calculateBestEstimate();
        geometry_msgs::PoseArray poseArray;
        poseArray.header.frame_id = "world";
        for (int i = 0; i < currentEstimate.size(); ++i)
        {
            vector<double> po;
            po.push_back(currentEstimate.at<Pose3>(i).rotation().pitch()); // pitch
            po.push_back(currentEstimate.at<Pose3>(i).rotation().yaw());   // yaw
            po.push_back(currentEstimate.at<Pose3>(i).rotation().roll());  // roll
            po.push_back(currentEstimate.at<Pose3>(i).translation().x());  // x
            po.push_back(currentEstimate.at<Pose3>(i).translation().y());  // y
            po.push_back(currentEstimate.at<Pose3>(i).translation().z());  // z
            optPoses.push_back(po);

            geometry_msgs::Pose p;
            geometry_msgs::Quaternion Q = tf::createQuaternionMsgFromRollPitchYaw(po[2], po[0], po[1]);
            p.orientation.x = Q.x;
            p.orientation.y = Q.y;
            p.orientation.z = Q.z;
            p.orientation.w = Q.w;
            p.position.x = po[3];
            p.position.y = po[4];
            p.position.z = po[5];
            poseArray.poses.push_back(p);
        }
        pub_optimized_pose.publish(poseArray);

        // cout << optPoses.size() << endl;
        if (optPoses.size() == test_num) // just for testing
        {
            globalMaps->clear();
            PointType p;
            for (int i = 0; i < optPoses.size(); ++i)
            {
                for (int j = 0; j < features[i]->size(); ++j)
                {
                    camToWorld(&features[i]->points[j], &p, optPoses[i]);
                    tree.updateNode(octomap::point3d(p.x, p.y, p.z), true);
                    // globalMaps->push_back(p);
                }
                // pcl::PointCloud<PointType>::Ptr globalMapsTemp(new pcl::PointCloud<PointType>());
                // downSizeFilter.setLeafSize(resolution, resolution, resolution);
                // downSizeFilter.setInputCloud(globalMaps);
                // downSizeFilter.filter(*globalMapsTemp);
                // globalMaps = globalMapsTemp;
            }

            for (auto it = tree.begin_leafs(); it != tree.end_leafs(); it++)
            {
                p.x = it.getCoordinate().x();
                p.y = it.getCoordinate().y();
                p.z = it.getCoordinate().z();
                p.intensity = 255;
                globalMaps->push_back(p);
            }
            cout << "mapping:" << globalMaps->size() << endl;
            sensor_msgs::PointCloud2 cloudOutMsg;
            pcl::toROSMsg(*globalMaps, cloudOutMsg);
            cloudOutMsg.header.frame_id = "world";
            pub_global_map.publish(cloudOutMsg);
        }
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        // std::cout << "backend time cost = " << time_used.count() << " seconds. " << std::endl;
    }
}

void localMapHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<PointType>::Ptr incloud(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*msg, *incloud);
    mtx.lock();
    localMapsTime = msg->header.stamp.toSec();
    // cout << 1 << endl;
    localMaps.push_back(incloud);
    mtx.unlock();
}

void featureHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<PointType>::Ptr incloud(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*msg, *incloud);
    mtx.lock();
    featuresTime = msg->header.stamp.toSec();
    // cout << 2 << endl;
    features.push_back(incloud);
    // cout << features.size() << endl;
    mtx.unlock();
}

void unoptPoseHandler(const nav_msgs::Odometry::ConstPtr &msg)
{
    PointType p;
    p.x = msg->pose.pose.position.x;
    p.y = msg->pose.pose.position.y;
    p.z = msg->pose.pose.position.z;
    p.intensity = 0;
    double pitch, yaw, roll;
    geometry_msgs::Quaternion Q = msg->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(Q.x, Q.y, Q.z, Q.w)).getRPY(roll, pitch, yaw);
    vector<double> pose;
    pose.push_back(pitch);
    pose.push_back(yaw);
    pose.push_back(roll);
    pose.push_back(p.x);
    pose.push_back(p.y);
    pose.push_back(p.z);
    pose.push_back(msg->header.stamp.toSec());
    pose.push_back(0);
    // showDoubleVector(pose);
    mtx.lock();
    unoptPosesTime = msg->header.stamp.toSec();
    // cout << 3 << endl;
    unoptPosesMap->push_back(p);
    unoptPoses.push_back(pose);
    mtx.unlock();
}

void showDoubleVector(vector<double> &v)
{
    cout << "mapping:";
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << setprecision(12) << v[i] << " ";
    }
    cout << endl;
}

void debugTopicPub(int curID, int hisID)
{
    nav_msgs::Odometry poseData;
    poseData.header.frame_id = "world";
    geometry_msgs::Quaternion Q = tf::createQuaternionMsgFromRollPitchYaw(unoptPoses[curID][2], unoptPoses[curID][0], unoptPoses[curID][1]);
    poseData.pose.pose.orientation.x = Q.x;
    poseData.pose.pose.orientation.y = Q.y;
    poseData.pose.pose.orientation.z = Q.z;
    poseData.pose.pose.orientation.w = Q.w;
    poseData.pose.pose.position.x = unoptPoses[curID][3];
    poseData.pose.pose.position.y = unoptPoses[curID][4];
    poseData.pose.pose.position.z = unoptPoses[curID][5];
    pub_closure_cur_pose.publish(poseData);

    nav_msgs::Odometry poseData1;
    poseData1.header.frame_id = "world";
    geometry_msgs::Quaternion Q1 = tf::createQuaternionMsgFromRollPitchYaw(unoptPoses[hisID][2], unoptPoses[hisID][0], unoptPoses[hisID][1]);
    poseData1.pose.pose.orientation.x = Q1.x;
    poseData1.pose.pose.orientation.y = Q1.y;
    poseData1.pose.pose.orientation.z = Q1.z;
    poseData1.pose.pose.orientation.w = Q1.w;
    poseData1.pose.pose.position.x = unoptPoses[hisID][3];
    poseData1.pose.pose.position.y = unoptPoses[hisID][4];
    poseData1.pose.pose.position.z = unoptPoses[hisID][5];
    pub_closure_his_pose.publish(poseData1);

    sensor_msgs::PointCloud2 cloudOutMsg;
    pcl::toROSMsg(*(localMaps[curID]), cloudOutMsg);
    cloudOutMsg.header.frame_id = "world";
    pub_closure_cur_map.publish(cloudOutMsg);

    sensor_msgs::PointCloud2 cloudOutMsg1;
    pcl::toROSMsg(*(localMaps[hisID]), cloudOutMsg1);
    cloudOutMsg1.header.frame_id = "world";
    pub_closure_his_map.publish(cloudOutMsg1);
}

void camToWorld(PointType const *const in, PointType *const out, vector<double> &pose)
{
    // 转Roll
    float x1 = cos(pose[2]) * in->x - sin(pose[2]) * in->y;
    float y1 = sin(pose[2]) * in->x + cos(pose[2]) * in->y;
    float z1 = in->z;
    // 转Pitch
    float x2 = x1;
    float y2 = cos(pose[0]) * y1 - sin(pose[0]) * z1;
    float z2 = sin(pose[0]) * y1 + cos(pose[0]) * z1;
    // 转Yaw并加平移
    out->x = cos(pose[1]) * x2 + sin(pose[1]) * z2 + pose[3];
    out->y = y2 + pose[4];
    out->z = -sin(pose[1]) * x2 + cos(pose[1]) * z2 + pose[5];
    out->intensity = in->intensity;
}
