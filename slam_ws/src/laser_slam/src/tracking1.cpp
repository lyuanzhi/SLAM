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
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <pangolin/pangolin.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
using namespace std;
typedef pcl::PointXYZI PointType;
ros::Publisher pub_pose;
ros::Publisher pub_cloud;
ros::Publisher pub_feature;
ros::Publisher pub_cloud_disp;
ros::Publisher pub_GT;
ros::Subscriber sub;
ifstream fin;
// ofstream fout;

float resolution = 0;
int iterCount_num = 0;
int nearestKSearch_num = 0;
float max_d = 0;
int show_goodFeatureNum_size = 0;
string topic = "";
string poseGT_path = "";
string pose_path = "";
int cal_curva_rate = 0;
int show_pose_err = 0;
float lim = 0;

vector<float> pose(6, 0); // Pitch Yaw Roll x y z
pcl::PointCloud<PointType>::Ptr localMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr features(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilter;
pcl::KdTreeFLANN<PointType>::Ptr kdtreeFLANN(new pcl::KdTreeFLANN<PointType>());
pcl::PointCloud<PointType>::Ptr validCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr localMap_disp(new pcl::PointCloud<PointType>());

void getYamlParamters(string path);
void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg);
void getValidCloud(const sensor_msgs::PointCloud2ConstPtr &msg, pcl::PointCloud<PointType>::Ptr outcloud);
void frameMatching(pcl::PointCloud<PointType>::Ptr incloud);
void topicPub();
double rad2deg(double rad);
void camToWorld(PointType const *const in, PointType *const out); // 雷达坐标系到世界坐标系(欧拉角坐标转换)
void filterThroughXYZ(pcl::PointCloud<PointType>::Ptr &inCloud, pcl::PointCloud<PointType>::Ptr &outCloud,
                      float x_down, float x_up, float y_down, float y_up, float z_down, float z_up);

int main(int argc, char **argv)
{
    getYamlParamters(ros::package::getPath("laser_slam") + "/config/tracking1.yaml");
    ros::init(argc, argv, "tracking1");
    ros::NodeHandle handle;
    pub_pose = handle.advertise<nav_msgs::Odometry>("/pose", 1);
    pub_cloud = handle.advertise<sensor_msgs::PointCloud2>("/cloud", 1);
    pub_feature = handle.advertise<sensor_msgs::PointCloud2>("/feature", 1);
    pub_cloud_disp = handle.advertise<sensor_msgs::PointCloud2>("/clouddisp", 1);
    pub_GT = handle.advertise<nav_msgs::Odometry>("/posegt", 1);
    sub = handle.subscribe<sensor_msgs::PointCloud2>(topic, 100, cloudHandler);
    fin.open(ros::package::getPath("laser_slam") + poseGT_path);
    if (!fin.is_open())
        return -1;
    // fout.open(ros::package::getPath("laser_slam") + pose_path);
    // if (!fout.is_open())
    //     return -1;
    ros::spin();
    return 0;
}

void getYamlParamters(string path)
{
    YAML::Node config = YAML::LoadFile(path);
    topic = config["topic"].as<string>();
    poseGT_path = config["poseGT_path"].as<string>();
    pose_path = config["pose_path"].as<string>();
    resolution = config["resolution"].as<float>();
    iterCount_num = config["iterCount_num"].as<int>();
    nearestKSearch_num = config["nearestKSearch_num"].as<int>();
    max_d = config["max_d"].as<float>();
    show_goodFeatureNum_size = config["show_goodFeatureNum_size"].as<int>();
    cal_curva_rate = config["cal_curva_rate"].as<int>();
    show_pose_err = config["show_pose_err"].as<int>();
    lim = config["lim"].as<float>();
    cout << "topic:" << topic << endl;
    cout << "poseGT_path:" << poseGT_path << endl;
    cout << "pose_path:" << poseGT_path << endl;
    cout << "resolution:" << resolution << endl;
    cout << "iterCount_num:" << iterCount_num << endl;
    cout << "nearestKSearch_num:" << nearestKSearch_num << endl;
    cout << "max_d:" << max_d << endl;
    cout << "show_goodFeatureNum_size:" << show_goodFeatureNum_size << endl;
    cout << "cal_curva_rate:" << cal_curva_rate << endl;
    cout << "show_pose_err:" << show_pose_err << endl;
    cout << "lim:" << lim << endl;
}

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    validCloud->clear();
    getValidCloud(msg, validCloud);
    frameMatching(validCloud);
    topicPub();
    // exit(0);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // std::cout << "frontend time cost = " << time_used.count() << " seconds. " << std::endl;
}

void getValidCloud(const sensor_msgs::PointCloud2ConstPtr &msg, pcl::PointCloud<PointType>::Ptr outcloud)
{
    pcl::PointCloud<PointType>::Ptr incloud(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*msg, *incloud);
    // cout << incloud->size() << endl;
    for (auto it = incloud->begin(); it != incloud->end(); ++it)
    {
        if (!pcl_isfinite(it->x) || !pcl_isfinite(it->y) || !pcl_isfinite(it->z))
            continue;
        if (it->x == 0 && it->y == 0 && it->z == 0)
            continue;
        outcloud->push_back(*it);
    }
}

void frameMatching(pcl::PointCloud<PointType>::Ptr incloud)
{
    if (incloud->size() < 100)
        return;
    // LM非线性位姿优化
    features->clear();
    float diffX, diffY, diffZ, curvature;
    for (int i = 2; i < incloud->size() - 2; i += cal_curva_rate)
    {
        // 计算曲率
        diffX = incloud->points[i - 2].x + incloud->points[i - 1].x - 4 * incloud->points[i].x + incloud->points[i + 1].x + incloud->points[i + 2].x;
        diffY = incloud->points[i - 2].y + incloud->points[i - 1].y - 4 * incloud->points[i].y + incloud->points[i + 1].y + incloud->points[i + 2].y;
        diffZ = incloud->points[i - 2].z + incloud->points[i - 1].z - 4 * incloud->points[i].z + incloud->points[i + 1].z + incloud->points[i + 2].z;
        curvature = diffX * diffX + diffY * diffY + diffZ * diffZ;
        if (curvature < 0.001)
            features->push_back(incloud->points[i]);
    }
    if (features->size() == 0)
        return;
    // cout << features->size() << endl;
    pcl::PointCloud<PointType>::Ptr featuresTemp(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(resolution, resolution, resolution);
    downSizeFilter.setInputCloud(features);
    downSizeFilter.filter(*featuresTemp);
    features = featuresTemp;
    if (features->size() == 0)
        return;

    // cout << features->size() << endl;

    PointType pointTemp, desc;
    if (localMap->size() > 100)
    {
        vector<int> pointSearchInd;
        vector<float> pointSearchDis;
        cv::Mat matA0(8, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB0(8, 1, CV_32F, cv::Scalar::all(-1));
        cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));
        kdtreeFLANN->setInputCloud(localMap);
        for (int iterCount = 0; iterCount < iterCount_num; iterCount++)
        {
            // 构建Jaccobian矩阵
            pcl::PointCloud<PointType>::Ptr goodFeatures(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr descs(new pcl::PointCloud<PointType>()); // Jaccobian矩阵的平移部分元素
            for (int i = 0; i < features->size(); i++)
            {
                camToWorld(&features->points[i], &pointTemp);
                kdtreeFLANN->nearestKSearch(pointTemp, nearestKSearch_num, pointSearchInd, pointSearchDis); // 这8个点为特征点在上一时刻的对应点
                if (pointSearchDis[7] < 5.0)
                {
                    for (int j = 0; j < nearestKSearch_num; j++)
                    {
                        matA0.at<float>(j, 0) = localMap->points[pointSearchInd[j]].x;
                        matA0.at<float>(j, 1) = localMap->points[pointSearchInd[j]].y;
                        matA0.at<float>(j, 2) = localMap->points[pointSearchInd[j]].z;
                    }

                    // 获取这8个点构成的平面方程
                    // AX+BY+CZ+D=0 => (A/D)X+(B/D)Y+(C/D)Z=-1 => matA0 * matX0 = matB0
                    cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);
                    float pa = matX0.at<float>(0, 0);
                    float pb = matX0.at<float>(1, 0);
                    float pc = matX0.at<float>(2, 0);
                    float pd = 1;

                    // 归一化
                    float ps = sqrt(pa * pa + pb * pb + pc * pc);
                    pa /= ps;
                    pb /= ps;
                    pc /= ps;
                    pd /= ps;

                    // 删掉不在同一平面的点
                    bool planeValid = true;
                    for (int j = 0; j < nearestKSearch_num; j++)
                    {
                        // 归一化后点到平面的距离 > max_d
                        if (fabs(pa * localMap->points[pointSearchInd[j]].x + pb * localMap->points[pointSearchInd[j]].y + pc * localMap->points[pointSearchInd[j]].z + pd) > max_d)
                        {
                            planeValid = false;
                            break;
                        }
                    }
                    if (planeValid)
                    {
                        // d=|Ax+By+Cz+D|/√(A²+B²+C²)
                        float pd2 = pa * pointTemp.x + pb * pointTemp.y + pc * pointTemp.z + pd;
                        // 阻尼因子, 点到平面距离越小阻尼因子越大
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointTemp.x * pointTemp.x + pointTemp.y * pointTemp.y + pointTemp.z * pointTemp.z));
                        desc.x = s * pa;
                        desc.y = s * pb;
                        desc.z = s * pc;
                        desc.intensity = s * pd2;
                        if (s > 0.1) // 确保点到平面的距离足够小
                        {
                            goodFeatures->push_back(features->points[i]);
                            descs->push_back(desc);
                        }
                    }
                }
            }
            int goodFeatureNum = goodFeatures->size();
            if (show_goodFeatureNum_size)
                cout << goodFeatureNum << endl;
            if (goodFeatureNum < 50)
                continue;

            // 欧拉角转旋转矩阵 R
            // |cry*crz+sry*srx*srz crz*sry*srx-cry*srz crx*sry|
            // |      crx*srz             crx*crz        -srx  |
            // |cry*srx*srz-crz*sry cry*crz*srx+sry*srz cry*crx|
            // JT*J*x = JT*f
            cv::Mat matJ(goodFeatureNum, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matJT(6, goodFeatureNum, CV_32F, cv::Scalar::all(0));
            cv::Mat matJTJ(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matF(goodFeatureNum, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matJTF(6, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
            float srx = sin(pose[0]);
            float crx = cos(pose[0]);
            float sry = sin(pose[1]);
            float cry = cos(pose[1]);
            float srz = sin(pose[2]);
            float crz = cos(pose[2]);
            for (int i = 0; i < goodFeatureNum; i++)
            {
                pointTemp = goodFeatures->points[i];
                desc = descs->points[i];
                // (dt/dx, dt/dy, dt/dz) * (dR/dx) * PT
                float arx = (crx * sry * srz * pointTemp.x + crx * crz * sry * pointTemp.y - srx * sry * pointTemp.z) * desc.x + (-srx * srz * pointTemp.x - crz * srx * pointTemp.y - crx * pointTemp.z) * desc.y + (crx * cry * srz * pointTemp.x + crx * cry * crz * pointTemp.y - cry * srx * pointTemp.z) * desc.z;
                // (dt/dx, dt/dy, dt/dz) * (dR/dy) * PT
                float ary = ((cry * srx * srz - crz * sry) * pointTemp.x + (sry * srz + cry * crz * srx) * pointTemp.y + crx * cry * pointTemp.z) * desc.x + ((-cry * crz - srx * sry * srz) * pointTemp.x + (cry * srz - crz * srx * sry) * pointTemp.y - crx * sry * pointTemp.z) * desc.z;
                // (dt/dx, dt/dy, dt/dz) * (dR/dz) * PT
                float arz = ((crz * srx * sry - cry * srz) * pointTemp.x + (-cry * crz - srx * sry * srz) * pointTemp.y) * desc.x + (crx * crz * pointTemp.x - crx * srz * pointTemp.y) * desc.y + ((sry * srz + cry * crz * srx) * pointTemp.x + (crz * sry - cry * srx * srz) * pointTemp.y) * desc.z;
                // 欧拉角的偏导
                matJ.at<float>(i, 0) = arx;
                matJ.at<float>(i, 1) = ary;
                matJ.at<float>(i, 2) = arz;
                // 平移的偏导
                matJ.at<float>(i, 3) = desc.x;
                matJ.at<float>(i, 4) = desc.y;
                matJ.at<float>(i, 5) = desc.z;
                // distance
                matF.at<float>(i, 0) = -desc.intensity;
            }
            cv::transpose(matJ, matJT);
            matJTJ = matJT * matJ;
            matJTF = matJT * matF;
            cv::solve(matJTJ, matJTF, matX, cv::DECOMP_QR);

            // 退化检测
            cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));
            bool isDegenerate = false;
            if (iterCount == 0)
            {
                cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
                cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
                cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

                // 计算JTJ的特征值matE和特征向量矩阵matV
                cv::eigen(matJTJ, matE, matV);
                matV.copyTo(matV2);

                isDegenerate = false;
                float eignThre[6] = {1, 1, 1, 1, 1, 1};
                for (int i = 5; i >= 0; i--)
                {
                    if (matE.at<float>(0, i) < eignThre[i])
                    {
                        for (int j = 0; j < 6; j++)
                            matV2.at<float>(i, j) = 0; // 抹去退化方向
                        isDegenerate = true;
                    }
                    else
                        break;
                }
                matP = matV.inv() * matV2;
            }
            if (isDegenerate)
            {
                cout << "退化了！" << endl;
                cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
                matX.copyTo(matX2);
                matX = matP * matX2;
            }

            pose[0] += matX.at<float>(0, 0);
            pose[1] += matX.at<float>(1, 0);
            pose[2] += matX.at<float>(2, 0);
            pose[3] += matX.at<float>(3, 0);
            pose[4] += matX.at<float>(4, 0);
            pose[5] += matX.at<float>(5, 0);

            float deltaR = sqrt(pow(rad2deg(matX.at<float>(0, 0)), 2) + pow(rad2deg(matX.at<float>(1, 0)), 2) + pow(rad2deg(matX.at<float>(2, 0)), 2));
            float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) + pow(matX.at<float>(5, 0) * 100, 2));
            if (deltaR < 0.05 && deltaT < 0.05) // 收敛
                break;
        }
    }

    // Pitch Yaw Roll x y z
    // cout << "Pitch: " << pose[0] << " Yaw: " << pose[1] << " Roll: " << pose[3] << endl;

    for (int i = 0; i < features->size(); i++)
    {
        camToWorld(&features->points[i], &pointTemp);
        localMap->push_back(pointTemp);
    }

    pcl::PointCloud<PointType>::Ptr localMapTemp(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(resolution, resolution, resolution);
    downSizeFilter.setInputCloud(localMap);
    downSizeFilter.filter(*localMapTemp);
    // localMap = localMapTemp;

    filterThroughXYZ(localMapTemp, localMap, pose[3] - lim, pose[3] + lim, pose[4] - lim, pose[4] + lim, pose[5] - lim, pose[5] + lim);

    localMap_disp->clear();
    for (int i = 0; i < validCloud->size(); i++)
    {
        camToWorld(&validCloud->points[i], &pointTemp);
        localMap_disp->push_back(pointTemp);
    }
}

void topicPub()
{
    ros::Time time = ros::Time::now();

    nav_msgs::Odometry poseData;
    poseData.header.frame_id = "world";
    poseData.header.stamp = time;
    geometry_msgs::Quaternion Q = tf::createQuaternionMsgFromRollPitchYaw(pose[2], pose[0], pose[1]);
    poseData.pose.pose.orientation.x = Q.x;
    poseData.pose.pose.orientation.y = Q.y;
    poseData.pose.pose.orientation.z = Q.z;
    poseData.pose.pose.orientation.w = Q.w;
    poseData.pose.pose.position.x = pose[3];
    poseData.pose.pose.position.y = pose[4];
    poseData.pose.pose.position.z = pose[5];
    pub_pose.publish(poseData);
    // cout << "tracking:" << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3] << " " << pose[4] << " " << pose[5] << " " << time.toSec() << endl;

    sensor_msgs::PointCloud2 cloudOutMsg;
    pcl::toROSMsg(*localMap, cloudOutMsg);
    cloudOutMsg.header.frame_id = "world";
    cloudOutMsg.header.stamp = time;
    pub_cloud.publish(cloudOutMsg);

    // cout << "tracking:" << localMap->size() << endl;

    sensor_msgs::PointCloud2 cloudOutMsg1;
    pcl::toROSMsg(*features, cloudOutMsg1); // for better global map display (longer time cost)
    cloudOutMsg1.header.frame_id = "world";
    cloudOutMsg1.header.stamp = time;
    pub_feature.publish(cloudOutMsg1);

    sensor_msgs::PointCloud2 cloudOutMsg2;
    pcl::toROSMsg(*localMap_disp, cloudOutMsg2);
    cloudOutMsg2.header.frame_id = "world";
    cloudOutMsg2.header.stamp = time;
    pub_cloud_disp.publish(cloudOutMsg2);

    double data[12] = {0};
    int cnt = 0;
    for (auto &d : data)
    {
        fin >> d;
        if (d == 0)
            cnt++;
    }
    if (cnt > 5)
        return;
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    rotation_matrix << data[0], data[1], data[2], data[4], data[5], data[6], data[8], data[9], data[10];
    Eigen::Quaterniond Q1 = Eigen::Quaterniond(rotation_matrix);
    nav_msgs::Odometry poseData1;
    poseData1.header.frame_id = "world";
    poseData1.header.stamp = time;
    poseData1.pose.pose.orientation.x = Q1.x();
    poseData1.pose.pose.orientation.y = Q1.y();
    poseData1.pose.pose.orientation.z = Q1.z();
    poseData1.pose.pose.orientation.w = Q1.w();
    poseData1.pose.pose.position.x = data[11];
    poseData1.pose.pose.position.y = -data[3];
    poseData1.pose.pose.position.z = data[7];
    pub_GT.publish(poseData1);

    if (show_pose_err)
    {
        cout << "x:" << poseData1.pose.pose.position.x - poseData.pose.pose.position.x
             << " y:" << poseData1.pose.pose.position.y - poseData.pose.pose.position.y
             << " z:" << poseData1.pose.pose.position.z - poseData.pose.pose.position.z << endl;
    }

    // //这里可能有问题
    // Eigen::Vector3d eulerAngle(pose[1], pose[0], pose[2]);
    // Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitX()));
    // Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
    // Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(0), Eigen::Vector3d::UnitZ()));
    // Eigen::Matrix3d m;
    // m = yawAngle * pitchAngle * rollAngle;
    // fout << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << " " << -pose[4] << " " << m(1, 0) << " " << m(1, 1) << " " << m(1, 2)
    //      << " " << pose[5] << " " << m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << " " << pose[3] << endl;
}

double rad2deg(double rad)
{
    return rad * 180.0 / M_PI;
}

void camToWorld(PointType const *const in, PointType *const out)
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

void filterThroughXYZ(pcl::PointCloud<PointType>::Ptr &inCloud, pcl::PointCloud<PointType>::Ptr &outCloud,
                      float x_down, float x_up, float y_down, float y_up, float z_down, float z_up)
{
    pcl::PassThrough<PointType> ptfilter(true);
    pcl::PointCloud<PointType>::Ptr filteredCloud1(new pcl::PointCloud<PointType>());
    ptfilter.setInputCloud(inCloud);
    ptfilter.setFilterFieldName("x");
    ptfilter.setFilterLimits(x_down, x_up);
    ptfilter.setNegative(false);
    ptfilter.filter(*filteredCloud1);

    pcl::PointCloud<PointType>::Ptr filteredCloud2(new pcl::PointCloud<PointType>());
    ptfilter.setInputCloud(filteredCloud1);
    ptfilter.setFilterFieldName("y");
    ptfilter.setFilterLimits(y_down, y_up);
    ptfilter.setNegative(false);
    ptfilter.filter(*filteredCloud2);

    ptfilter.setInputCloud(filteredCloud2);
    ptfilter.setFilterFieldName("z");
    ptfilter.setFilterLimits(z_down, z_up);
    ptfilter.setNegative(false);
    ptfilter.filter(*outCloud);
}
