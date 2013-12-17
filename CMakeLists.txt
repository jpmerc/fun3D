cmake_minimum_required(VERSION 2.8.3)
project(fun)

find_package(catkin REQUIRED COMPONENTS pcl pcl_ros roscpp rospy sensor_msgs std_msgs std_msgs)


catkin_package(
   INCLUDE_DIRS include
#  LIBRARIES fun
#  CATKIN_DEPENDS pcl pcl_ros roscpp rospy sensor_msgs std_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(include
  ${catkin_INCLUDE_DIRS}
)

add_executable(test src/test.cpp)
target_link_libraries(test ${catkin_LIBRARIES})


