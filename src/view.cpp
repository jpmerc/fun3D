#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "pcl/pcl_base.h"
#include "pcl/point_cloud.h"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>


using namespace std;

boost::shared_ptr<pcl::visualization::CloudViewer> viewer (new pcl::visualization::CloudViewer("Viewer"));
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

void imgReceived(const sensor_msgs::PointCloud2Ptr& msg){

    pcl::fromROSMsg(*msg,*cloud);

    cout << "PC Size = " << cloud->size() << endl;

    viewer->showCloud(cloud);

}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "view");
    ros::NodeHandle n;
    ros::Rate loop_rate(5);

    ros::Subscriber sub_image = n.subscribe("/camera/points", 1, imgReceived);

    while (ros::ok() && !viewer->wasStopped ())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
