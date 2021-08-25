// This file is part of DVS-ROS - the RPG DVS ROS Package
#include <ros/ros.h>
#include "cm_ros/cm_ros.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "cm_ros");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  ROS_INFO("cm_ros node established");
  cm_ros::ContrastMaximizationRos cm(nh,nh_private);

  ros::spin();
  return 0;
}
