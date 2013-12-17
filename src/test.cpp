#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include <iostream>
#include "std_msgs/UInt8.h"
#include <boost/thread.hpp>
#include "pcl/pcl_base.h"
#include "pcl/point_cloud.h"
#include "pcl_ros/impl/transforms.hpp"
#include "pcl/visualization/cloud_viewer.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/search/kdtree.h"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <angles/angles.h>
using namespace std;

bool firstCapture = false;
bool capturePointCloud = false;
pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);

void imgReceived(const sensor_msgs::PointCloud2Ptr& msg){
    if(firstCapture){
        cout << "first capture !!" << endl;
        pcl::fromROSMsg(*msg,*source_cloud);
        firstCapture = false;
        //        pcl::visualization::CloudViewer viewer("Cloud");
        //        viewer.showCloud((*source_cloud).makeShared());
        //        while (!viewer.wasStopped())
        //        {
        //        }
    }
    if(capturePointCloud){
        cout << "second capture !!" << endl;
        pcl::fromROSMsg(*msg,*target_cloud);
        capturePointCloud = false;
    }
    //    pcl::visualization::CloudViewer viewer("Cloud");
    //    viewer.showCloud(pcl_cloud.makeShared());
    //      while (!viewer.wasStopped())
    //      {
    //      }
}

void generateCustomClouds(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_target(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_source(new pcl::PointCloud<pcl::PointXYZ>);
    for (float z(-1.0); z <= 1.0; z += 0.05)
      {
        for (float angle(0.0); angle <= 360.0; angle += 5.0)
        {
          pcl::PointXYZ basic_point;
          basic_point.x = 0.5 * cosf (angles::from_degrees(angle));
          basic_point.y = sinf (angles::from_degrees(angle));
          basic_point.z = z;
          tmp_target->points.push_back(basic_point);

          basic_point.z = basic_point.z + 5;
          tmp_source->points.push_back (basic_point);
        }
      }

    cout << "Taille du PointCloud = " << tmp_target->size() << endl;
    target_cloud = tmp_target;
    source_cloud = tmp_source;
}


//==============================================FPFH=========================================================

pcl::PointCloud<pcl::PointXYZ>::Ptr downSample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    const float voxel_grid_size = 0.005;
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud (cloud);
    vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);
    vox_grid.filter (*tempCloud);
    return tempCloud;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr calculateFPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    //Normal estimation
    cout << "Starting normals estimation!" << endl;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); //search method
    ne.setInputCloud (cloud);
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.02); //neighbours in 3cm sphere
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>); //output
    ne.compute (*cloud_normals);
    //bool sizeComp = cloud_normals->points.size() == cloud_to_merge->points.size();
    cout << "Taille Normales = " << cloud_normals->points.size() << "   Taille PointCloud = " << target_cloud->points.size() << endl;
    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*

    //FPFH Calculations
    cout << "Starting FPFH calculation!" << endl;
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (cloud_normals);
    fpfh.setSearchMethod (tree2);
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch (0.02);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ()); //Output
    fpfh.compute (*fpfhs);
    cout << "Taille FPFH = " << fpfhs->points.size() << endl;
    return fpfhs;
}


void mergePointClouds(pcl::PointCloud<pcl::FPFHSignature33>::Ptr f_src,pcl::PointCloud<pcl::FPFHSignature33>::Ptr f_target){
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
    sac_ia_.setMinSampleDistance(0.05);
    sac_ia_.setMaxCorrespondenceDistance(0.01*0.01);
    sac_ia_.setMaximumIterations(1000);

    //Set target
    sac_ia_.setInputTarget(target_cloud);
    sac_ia_.setTargetFeatures(f_target);

    //Set source
    sac_ia_.setInputCloud(source_cloud);
    sac_ia_.setSourceFeatures(f_src);

    cout << "Starting Alignment" << endl;
    //Output
    pcl::PointCloud<pcl::PointXYZ> registration_output;
    sac_ia_.align (registration_output);
    merged_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(registration_output));
    cout << "Alignment Done" << endl;
}


//===========================================================================================================
void pressAnyKey(){
    cout << "Press any key to take a snapshot" << endl;
    //getchar();
    sleep(6);
}

void spinFunction(){
    ros::Rate r(30);
    while(ros::ok()){
        ros::spinOnce();
        r.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    ros::Rate loop_rate(5);
    //    ros::Subscriber sub_image = n.subscribe("/camera/points",1,imgReceived);
    //    boost::thread spinThread(spinFunction);

    //===========================================LOAD PCD FILES===============================================//
    pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/jp/Devel/masterWS/src/fun/scans/jp1.pcd", *source_cloud);
    pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/jp/Devel/masterWS/src/fun/scans/jp2.pcd", *target_cloud);

    //generateCustomClouds();

    //===========================================TAKE SNAPSHOTS===============================================//
    //    //Take first scan
    //    cout << "First Scan : " << endl;
    //    pressAnyKey();
    //    firstCapture = true;
    //    while(true){
    //        if(firstCapture) loop_rate.sleep();
    //        else break;
    //    }
    //    pcl::io::savePCDFileASCII("/home/jp/Devel/masterWS/src/pmd_camboard_nano/scans/jp.pcd", *source_cloud);


    //Take second scan
    //    cout << "Second Scan : " << endl;
    //    //pressAnyKey();
    //    capturePointCloud = true;
    //    while(true){
    //        if(capturePointCloud) loop_rate.sleep();
    //        else break;
    //    }

    //===========================================FPFH===============================================//

    //cout << "Before sampling: TARGET= " << target_cloud->points.size() << "  SOURCE= " << source_cloud->points.size() << endl;
    target_cloud = downSample(target_cloud);
    source_cloud = downSample(source_cloud);
    //cout << "After sampling: TARGET= " << target_cloud->points.size() << "  SOURCE= " << source_cloud->points.size() << endl;

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_src;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target;
    features_target = calculateFPFH(target_cloud);
    features_src = calculateFPFH(source_cloud);

    mergePointClouds(features_src,features_target);


    //===========================================PCL VIEWER===============================================//

    boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->addCoordinateSystem (1.0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green_color(source_cloud, 0, 255, 0);
    pclViewer->addPointCloud<pcl::PointXYZ>(source_cloud,green_color,"source cloud");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue_color(target_cloud, 0, 0, 255);
    pclViewer->addPointCloud<pcl::PointXYZ>(target_cloud,blue_color,"target cloud");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color(merged_cloud, 255, 0, 0);
    pclViewer->addPointCloud<pcl::PointXYZ>(merged_cloud,red_color,"merged cloud");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged cloud");

    while (!pclViewer->wasStopped())
    {
        pclViewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }




    //spinThread.~thread();
    return 0;
}
