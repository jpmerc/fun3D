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
#include <vector>
#include <pcl/registration/icp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/filter.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/multiscale_feature_persistence.h>
using namespace std;

bool firstCapture = false;
bool capturePointCloud = false;

struct CloudAndNormals{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr features_cloud;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    boost::shared_ptr<pcl::RangeImage> range_image;
};

CloudAndNormals target;
CloudAndNormals source;

pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::Normal>::Ptr tmpNormals(new pcl::PointCloud<pcl::Normal>);
vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_vector;

bool sData = true;
bool tData = true;
bool mData = true;
int l_count = 0;

void pressAnyKey();
string convertInt(int number);

void imgReceived(const sensor_msgs::PointCloud2Ptr& msg){
    if(firstCapture){
        cout << "first capture !!" << endl;
        pcl::fromROSMsg(*msg,*source.cloud);
        firstCapture = false;
        //        pcl::visualization::CloudViewer viewer("Cloud");
        //        viewer.showCloud((*source.cloud).makeShared());
        //        while (!viewer.wasStopped())
        //        {
        //        }
    }
    if(capturePointCloud){
        cout << "second capture !!" << endl;
        pcl::fromROSMsg(*msg,*target.cloud);
        capturePointCloud = false;
    }
    //    pcl::visualization::CloudViewer viewer("Cloud");
    //    viewer.showCloud(pcl_cloud.makeShared());
    //      while (!viewer.wasStopped())
    //      {
    //      }
}

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    l_count = l_count + 1;
    if(l_count < 2){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
        if (event.getKeySym () == "t" && tData){
            viewer->removePointCloud("target cloud");
            tData = false;
        }
        else if (event.getKeySym () == "t" && !tData){
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue_color(target.cloud, 0, 0, 255);
            viewer->addPointCloud<pcl::PointXYZ>(target.cloud,blue_color,"target cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
            tData = true;
        }

        else if(event.getKeySym () == "s" && sData){
            viewer->removePointCloud("source cloud");
            sData = false;
        }
        else if(event.getKeySym () == "s" && !sData){
            viewer->removePointCloud("source cloud");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green_color(source.cloud, 0, 255, 0);
            viewer->addPointCloud<pcl::PointXYZ>(source.cloud,green_color,"source cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");
            sData = true;
        }
        else if(event.getKeySym () == "m" && mData){
            viewer->removePointCloud("merged cloud");
            mData = false;
        }
        else if(event.getKeySym () == "m" && !mData){
            viewer->removePointCloud("merged cloud");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color(merged_cloud, 255, 0, 0);
            viewer->addPointCloud<pcl::PointXYZ>(merged_cloud,red_color,"merged cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged cloud");
            mData = true;
        }

    }
    else{
        l_count = 0;
    }

}


//===============================GENERATE POINT CLOUDS=========================================
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
    target.cloud = tmp_target;
    source.cloud = tmp_source;
}


void takeSnapshots(){
    //=========Need to uncomment the subscription to /camera/points to work=========
    ros::Rate loop_rate(5);

    //Take first scan
    cout << "First Scan : " << endl;
    pressAnyKey();
    firstCapture = true;
    while(true){
        if(firstCapture) loop_rate.sleep();
        else break;
    }

    //Take second scan
    cout << "Second Scan : " << endl;
    pressAnyKey();
    capturePointCloud = true;
    while(true){
        if(capturePointCloud) loop_rate.sleep();
        else break;
    }
}

void saveModels(){
    //=========Need to uncomment the subscription to /camera/points to work=========
    ros::Rate loop_rate(5);
    string pcdBaseName = "/home/jp/Devel/masterWS/src/fun/scans/unwantedScan";
    for(int i=1; i<10; i++){
        string pcdPath = pcdBaseName + convertInt(i) + ".pcd";
        //Take first scan
        cout << "First Scan : " << endl;
        pressAnyKey();
        firstCapture = true;
        while(true){
            if(firstCapture) loop_rate.sleep();
            else break;
        }
        pcl::io::savePCDFileASCII(pcdPath, *source.cloud);
    }

}

void loadPCDFiles(string pcdBaseName,int Begin, int End){
    for(int i=Begin; i<=End; i++){
        string pcdPath = pcdBaseName + convertInt(i) + ".pcd";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ> (pcdPath, *cloud);
        cloud_vector.push_back(cloud);
    }


}

void setRangeImage(CloudAndNormals& structure){

    float angularResolution = (float) (  0.1f * (M_PI/180.0f));  //   1.0 degree in radians
    float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
    float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noiseLevel=0.00;
    float minRange = 0.0f;
    int borderSize = 1;

    pcl::RangeImage rangeImage;
    rangeImage.createFromPointCloud(*(structure.cloud), angularResolution, maxAngleWidth, maxAngleHeight,
                                    sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);

    boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage(rangeImage));
    structure.range_image = range_image_ptr;

}

void setViewerPose (void* viewer_void, const Eigen::Affine3f& viewer_pose){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
    Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
    Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
    viewer->setCameraPosition (double(pos_vector[0]), double(pos_vector[1]), double(pos_vector[2]),
                               double(look_at_vector[0]), double(look_at_vector[1]), double(look_at_vector[2]));
}


//==============================================FPFH=========================================================

pcl::PointCloud<pcl::PointXYZ>::Ptr downSample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    const float voxel_grid_size = 0.005;
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud (cloud);
    vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);
    vox_grid.filter (*tempCloud);

    //    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    //    outrem.setInputCloud(tempCloud);
    //    outrem.setRadiusSearch(0.05);
    //    outrem.setMinNeighborsInRadius(100);
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    //    outrem.filter (*tempCloud2);
    return tempCloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr segment(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals){

    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight (0.1);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (20);
    seg.setDistanceThreshold (0.02);
    seg.setInputCloud (cloud);
    seg.setInputNormals (normals);

    seg.segment (*inliers_plane, *coefficients_plane);
    //cout << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers_plane);
    extract.setNegative (true);
    extract.filter (*tempCloud);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    extract_normals.setNegative (true);
    extract_normals.setInputCloud (normals);
    extract_normals.setIndices (inliers_plane);
    extract_normals.filter (*cloud_normals2);

    tmpNormals = cloud_normals2;


    //    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    //    tree->setInputCloud (cloud);
    //    std::vector<pcl::PointIndices> cluster_indices;
    //    pcl::EuclideanClusterExtraction<pcl::PointXYZ> p;
    //    p.setInputCloud (cloud);
    //    p.setClusterTolerance (0.05);
    //    p.setMinClusterSize (100);
    //    p.setSearchMethod (tree);
    //    p.extract (cluster_indices);


    return tempCloud;
}

pcl::PointCloud<pcl::Normal>::Ptr calculateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    //Normal estimation
    cout << "Starting normals estimation!" << endl;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); //search method
    ne.setInputCloud (cloud);
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.03); //neighbours in 3cm sphere
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>); //output
    ne.compute (*cloud_normals);
    //bool sizeComp = cloud_normals->points.size() == cloud_to_merge->points.size();
    //cout << "Taille Normales = " << cloud_normals->points.size() << "   Taille PointCloud = " << cloud->points.size() << endl;
    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
    return cloud_normals;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr calculateFPFH(CloudAndNormals& data){
    //FPFH Calculations
    cout << "Starting FPFH calculation!" << endl;
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr fpfh_estimation(new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh_estimation->setInputCloud (data.cloud);
    fpfh_estimation->setInputNormals (data.normals);
    fpfh_estimation->setSearchMethod (tree);
    fpfh_estimation->setRadiusSearch (0.01);

    std::vector<float> scale_values; //set the multi scales
    //scale_values.push_back(0.005);
    scale_values.push_back(0.01);
    scale_values.push_back(0.03);
    pcl::MultiscaleFeaturePersistence<pcl::PointXYZ, pcl::FPFHSignature33> feature_persistence;
    feature_persistence.setScalesVector (scale_values);
    feature_persistence.setAlpha (1.2f);
    feature_persistence.setFeatureEstimator (fpfh_estimation);
    feature_persistence.setDistanceMetric (pcl::CS);


    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr output_features (new pcl::PointCloud<pcl::FPFHSignature33> ()); //Output
    boost::shared_ptr<std::vector<int> > output_indices (new std::vector<int> ());
    feature_persistence.determinePersistentFeatures (*output_features, output_indices);


    //    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    //    pcl::PointCloud<pcl::Normal>::Ptr tempNormals(new pcl::PointCloud<pcl::Normal>);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (data.cloud);
    extract.setIndices (output_indices);
    extract.setNegative (false);
    extract.filter (*data.features_cloud);

    //    pcl::ExtractIndices<pcl::Normal> extract2;
    //    extract2.setInputCloud (data.normals);
    //    extract2.setIndices (output_indices);
    //    extract2.setNegative (false);
    //    extract2.filter (*tempNormals);

    //    data.cloud = tempCloud;
    //    data.normals = tempNormals;

    //fpfh.compute (*output_features);
    cout << "Taille FPFH = " << output_features->points.size() << endl;
    return output_features;
}


Eigen::Matrix4f mergePointClouds(pcl::PointCloud<pcl::FPFHSignature33>::Ptr f_src,pcl::PointCloud<pcl::FPFHSignature33>::Ptr f_target){
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
    float maxDistanceSACIA = 1.0;
    sac_ia_.setMinSampleDistance(0.05);
    sac_ia_.setMaxCorrespondenceDistance(maxDistanceSACIA);
    sac_ia_.setMaximumIterations(1000);//default value
    sac_ia_.setNumberOfSamples(2); //default value = 3
    sac_ia_.setCorrespondenceRandomness(10); //default value

    //Set target
    sac_ia_.setInputTarget(target.features_cloud);
    sac_ia_.setTargetFeatures(f_target);

    //Set source
    sac_ia_.setInputCloud(source.features_cloud);
    sac_ia_.setSourceFeatures(f_src);

    cout << "Starting Alignment" << endl;
    //Output
    pcl::PointCloud<pcl::PointXYZ> registration_output;
    sac_ia_.align (registration_output);
    cout << "Alignment Done" << endl;
    //   pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>(registration_output));

    float sac_score = sac_ia_.getFitnessScore(maxDistanceSACIA);
    Eigen::Matrix4f sac_transformation = sac_ia_.getFinalTransformation();

    cout << "SAC-IA Transformation Score = " << sac_score << endl;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    float maxDistanceICP = 0.2;
    icp.setInputCloud(source.features_cloud);
    icp.setInputTarget(target.features_cloud);
    icp.setMaxCorrespondenceDistance(maxDistanceICP);
    icp.setMaximumIterations(20);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final,sac_transformation);
    cout << "ICP Transformation Score = " << icp.getFitnessScore(maxDistanceICP) << endl;


    Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();
    //pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>(Final));




    return icp_transformation;
}


//===========================================================================================================
void pressAnyKey(){
    cout << "Press any key to take a snapshot" << endl;
    //getchar();
    sleep(10);
}

void spinFunction(){
    ros::Rate r(30);
    while(ros::ok()){
        ros::spinOnce();
        r.sleep();
    }
}

string convertInt(int number)
{
    stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    ros::Rate loop_rate(5);


    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
    target.cloud = target_cloud;
    target.features_cloud = target_cloud;
    target.normals = target_normals;
    source.cloud = source_cloud;
    source.features_cloud = source_cloud;
    source.normals = source_normals;
    //    ros::Subscriber sub_image = n.subscribe("/camera/points",1,imgReceived);
    //    boost::thread spinThread(spinFunction);


    //===========================================LOAD POINT CLOUDS===============================================//
    //generateCustomClouds();
    //takeSnapshots();
    loadPCDFiles("/home/jp/Devel/masterWS/src/fun/scans/tasse",1,6);
    //saveModels();

    //===========================================METHOD IMPLEMENTATION===============================================//
    target.cloud = cloud_vector.front();
    cloud_vector.erase(cloud_vector.begin());
    //target.cloud = downSample(target.cloud);

    while(!cloud_vector.empty()){
        //Load Files
        source.cloud = cloud_vector.front();
        cloud_vector.erase(cloud_vector.begin());

        //Preprocessing
        cout << "Before sampling: TARGET= " << target.cloud->points.size() << "  SOURCE= " << source.cloud->points.size() << endl;
        target.cloud = downSample(target.cloud);
        source.cloud = downSample(source.cloud);
        target.normals = calculateNormals(target.cloud);
        source.normals = calculateNormals(source.cloud);
        //        target.cloud = segment(target.cloud,target.normals);
        //        target.normals = tmpNormals;
        //        source.cloud = segment(source.cloud,source.normals);
        //        source.normals = tmpNormals;
        cout << "After sampling: TARGET= " << target.cloud->points.size() << "  SOURCE= " << source.cloud->points.size() << endl;



        //Features Calculation (FPFH)
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_src;
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target;
        features_target = calculateFPFH(target);
        features_src = calculateFPFH(source);

        //Coarse alignment
        Eigen::Matrix4f transform = mergePointClouds(features_src,features_target);
        merged_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*source.cloud,*merged_cloud,transform);

        // pcl::PointCloud<pcl::PointXYZ> tmp_result = (*merged_cloud).operator +(*target.cloud);
        // merged_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(tmp_result));



        //setRangeImage(target);
        //cout << "Size de la range image = " << target.range_image->points.size() << endl;

        //===========================================PCL VIEWER===============================================//

        boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        pclViewer->setBackgroundColor (0, 0, 0);
        // pclViewer->addCoordinateSystem (1.0);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green_color(source.cloud, 0, 255, 0);
        pclViewer->addPointCloud<pcl::PointXYZ>(source.cloud,green_color,"source cloud");
        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue_color(target.cloud, 0, 0, 255);
        pclViewer->addPointCloud<pcl::PointXYZ>(target.cloud,blue_color,"target cloud");
        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color(merged_cloud, 255, 0, 0);
        pclViewer->addPointCloud<pcl::PointXYZ>(merged_cloud,red_color,"merged cloud");
        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged cloud");

//        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> lol_color(target.range_image, 255, 255, 0);
//        pclViewer->addPointCloud<pcl::PointWithRange>(target.range_image,lol_color,"range image");
//        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
//        setViewerPose((void*)&pclViewer, target.range_image->getTransformationToWorldSystem ());

        //        pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
        //        range_image_widget.showRangeImage (*(target.range_image));
        //        while(!range_image_widget.wasStopped()){
        //            range_image_widget.spinOnce (100);
        //            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        //        }

        pclViewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer);
        while (!pclViewer->wasStopped())
        {
            pclViewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

        }

        target.cloud = source.cloud;

        //        pclViewer->close();
    }


    //spinThread.~thread();
    return 0;
}
