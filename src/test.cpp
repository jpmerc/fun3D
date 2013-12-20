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
#include <pcl/filters/statistical_outlier_removal.h>
using namespace std;

const float DOWN_SAMPLE_VOXELS_LEAF_SIZE = 0.005;
double NORMAL_SPHERE_RADIUS = 0.03;
float FPFH_PERSISTENCE_SCALES[] = {0.01,0.015,0.02};
float FPFH_PERSISTENCE_ALPHA = 1.2f;
double SAC_IA_MAXIMUM_DISTANCE = 1.0;
float SAC_IA_MINIMUM_SAMPLING_DISTANCE = 0.02;
int SAC_IA_MAXIMUM_ITERATIONS = 1000;
int SAC_IA_NUMBER_OF_SAMPLES = 15;
int SAC_IA_CORRESPONDANCE_RANDOMNESS = 20;

bool firstCapture = false;
bool capturePointCloud = false;

struct CloudAndNormals{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr features_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_planes;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr signature_fpfh;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    boost::shared_ptr<pcl::RangeImage> range_image;
};

CloudAndNormals target;
CloudAndNormals source;
pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::Normal>::Ptr tmpNormals(new pcl::PointCloud<pcl::Normal>);
vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_vector;
bool sData = true;
bool tData = true;
bool mData = true;
int l_count = 0;
Eigen::Matrix4f temp_transform;


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

void downSample(CloudAndNormals& data){
    vector<int> index;
    removeNaNFromPointCloud(*data.cloud,*data.cloud,index);

    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud (data.cloud);
    vox_grid.setLeafSize (DOWN_SAMPLE_VOXELS_LEAF_SIZE, DOWN_SAMPLE_VOXELS_LEAF_SIZE, DOWN_SAMPLE_VOXELS_LEAF_SIZE);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);
    vox_grid.filter (*tempCloud);
    data.cloud = tempCloud;
    //    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    //    outrem.setInputCloud(tempCloud);
    //    outrem.setRadiusSearch(0.05);
    //    outrem.setMinNeighborsInRadius(100);
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    //    outrem.filter (*tempCloud2);
}
void segment(CloudAndNormals& data){
    cout << "Starting Planes Segmentation!" << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight (0.1);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (20);
    seg.setDistanceThreshold (0.05);
    seg.setInputCloud (data.cloud);
    seg.setInputNormals (data.normals);

    seg.segment (*inliers_plane, *coefficients_plane);
    //cout << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (data.cloud);
    extract.setIndices (inliers_plane);
    extract.setNegative (true);
    extract.filter (*tempCloud);

    //Extracted Planes
    pcl::ExtractIndices<pcl::PointXYZ> extract1;
    extract1.setInputCloud (data.cloud);
    extract1.setIndices (inliers_plane);
    extract1.setNegative (false);
    extract1.filter (*data.segmented_planes);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    extract_normals.setNegative (true);
    extract_normals.setInputCloud (data.normals);
    extract_normals.setIndices (inliers_plane);
    extract_normals.filter (*cloud_normals2);

    data.normals = cloud_normals2;
    data.cloud = tempCloud;
}

void calculateNormals(CloudAndNormals& data){
    //Normal estimation
    cout << "Starting normals estimation!" << endl;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); //search method
    ne.setInputCloud (data.cloud);
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (NORMAL_SPHERE_RADIUS); //neighbours in 3cm sphere
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>); //output
    ne.compute (*cloud_normals);
    data.normals = cloud_normals;
    //bool sizeComp = cloud_normals->points.size() == cloud_to_merge->points.size();
    //cout << "Taille Normales = " << cloud_normals->points.size() << "   Taille PointCloud = " << cloud->points.size() << endl;
    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
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

    std::vector<float> scale_values(FPFH_PERSISTENCE_SCALES,FPFH_PERSISTENCE_SCALES + sizeof(FPFH_PERSISTENCE_SCALES)/sizeof(float)); //set the multi scales
//    scale_values.push_back(0.01);
//    scale_values.push_back(0.015);
//    scale_values.push_back(0.02);
    pcl::MultiscaleFeaturePersistence<pcl::PointXYZ, pcl::FPFHSignature33> feature_persistence;
    feature_persistence.setScalesVector (scale_values);
    feature_persistence.setAlpha (FPFH_PERSISTENCE_ALPHA);
    feature_persistence.setFeatureEstimator (fpfh_estimation);
    feature_persistence.setDistanceMetric (pcl::CS);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr output_features (new pcl::PointCloud<pcl::FPFHSignature33> ()); //Output
    boost::shared_ptr<std::vector<int> > output_indices (new std::vector<int> ());
    feature_persistence.determinePersistentFeatures (*output_features, output_indices);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (data.cloud);
    extract.setIndices (output_indices);
    extract.setNegative (false);
    extract.filter (*data.features_cloud);

    cout << "Taille FPFH = " << output_features->points.size() << endl;
    return output_features;
}

void filterFeatures(CloudAndNormals& data){
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tempFeatures (new pcl::PointCloud<pcl::FPFHSignature33> ());

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorfilter(true); // Initializing with true will allow us to extract the removed indices
    sorfilter.setInputCloud (data.features_cloud);
    sorfilter.setMeanK (30);
    sorfilter.setStddevMulThresh (1.0);
    sorfilter.filter (*tempCloud);

    pcl::IndicesConstPtr output_indices = sorfilter.getRemovedIndices();
    cout << "Extracted " << output_indices->size() << " points from FPFH" << endl;

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (data.features_cloud);
    extract.setIndices (output_indices);
    extract.setNegative (true);
    extract.filter (*tempCloud);
    data.features_cloud = tempCloud;

    pcl::ExtractIndices<pcl::FPFHSignature33> extract2;
    extract2.setInputCloud (data.signature_fpfh);
    extract2.setIndices (output_indices);
    extract2.setNegative (true);
    extract2.filter (*tempFeatures);
    data.signature_fpfh = tempFeatures;

}


Eigen::Matrix4f mergePointClouds(pcl::PointCloud<pcl::FPFHSignature33>::Ptr f_src,pcl::PointCloud<pcl::FPFHSignature33>::Ptr f_target){
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
    sac_ia_.setMinSampleDistance(SAC_IA_MINIMUM_SAMPLING_DISTANCE);
    sac_ia_.setMaxCorrespondenceDistance(SAC_IA_MAXIMUM_DISTANCE);
    sac_ia_.setMaximumIterations(SAC_IA_MAXIMUM_ITERATIONS);
    sac_ia_.setNumberOfSamples(SAC_IA_NUMBER_OF_SAMPLES);
    sac_ia_.setCorrespondenceRandomness(SAC_IA_CORRESPONDANCE_RANDOMNESS);

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

    float sac_score = sac_ia_.getFitnessScore(SAC_IA_MAXIMUM_DISTANCE);
    Eigen::Matrix4f sac_transformation = sac_ia_.getFinalTransformation();
    temp_transform = sac_transformation;

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
    target.segmented_planes = target_cloud;
    source.cloud = source_cloud;
    source.features_cloud = source_cloud;
    source.normals = source_normals;
    source.segmented_planes = source_cloud;
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
        downSample(target);
        downSample(source);
        calculateNormals(target);
        calculateNormals(source);
        segment(target);
        segment(source);
        cout << "After sampling: TARGET= " << target.cloud->points.size() << "  SOURCE= " << source.cloud->points.size() << endl;

        //Features Calculation (FPFH)
//        target.signature_fpfh = calculateFPFH(target);
//        source.signature_fpfh = calculateFPFH(source);
//        filterFeatures(target);
//        filterFeatures(source);

//        //Coarse alignment
//        Eigen::Matrix4f transform = mergePointClouds(source.signature_fpfh,target.signature_fpfh);
//        merged_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
//        pcl::transformPointCloud(*source.cloud,*merged_cloud,transform);
        //merged_cloud2.reset(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::transformPointCloud(*source.features_cloud,*merged_cloud2,temp_transform);


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

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color(source.segmented_planes, 255, 0, 0);
        pclViewer->addPointCloud<pcl::PointXYZ>(source.segmented_planes,red_color,"merged cloud");
        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged cloud");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yellow_color(target.segmented_planes, 255, 255, 0);
        pclViewer->addPointCloud<pcl::PointXYZ>(target.segmented_planes,yellow_color,"merged cloud2");
        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged cloud2");

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
