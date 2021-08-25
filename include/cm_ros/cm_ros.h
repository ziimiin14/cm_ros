#ifndef CM_ROS_H
#define CM_ROS_H

#include <ros/ros.h>
#include <torch/torch.h>
#include <dvs_msgs/EventStruct.h>
#include "cm_ros/CM_Model.h"
#include "cm_ros/EventPacket.h"
#include "cm_ros/CM_Opt.h"


namespace cm_ros{

class ContrastMaximizationRos{
public:
	ContrastMaximizationRos(ros::NodeHandle & nh, ros::NodeHandle nh_private);

private:
	ros::NodeHandle nh_;
	ros::Subscriber event_sub_;
	torch::TensorOptions option_;
	eventPacket eventPackets_;
	CM_Model model_;
	CM_Opt opt_;
	

	void eventStructCallback(const dvs_msgs::EventStruct::Ptr& msg);
};

}



#endif // CM_ROS_H
