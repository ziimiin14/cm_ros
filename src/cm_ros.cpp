#include "cm_ros/cm_ros.h"
#include <iostream>

using namespace torch::indexing;

namespace cm_ros{

ContrastMaximizationRos::ContrastMaximizationRos(ros::NodeHandle & nh, ros::NodeHandle nh_private) : nh_(nh), option_(torch::TensorOptions().device(torch::kCUDA)), model_(CM_Model(option_,2,240,180)), opt_(CM_Opt(model_,option_))
{

    eventPackets_.K = torch::eye(3,option_);
    eventPackets_.K.index({0,0}) = 3.2372678681198357e+02;
    eventPackets_.K.index({1,1}) = 3.2306065962539299e+02;
    eventPackets_.K.index({0,2}) = 1.2054931481123528e+02;
    eventPackets_.K.index({1,2}) = 8.7519021071803550e+01;
    std::cout << eventPackets_.K << std::endl;
    event_sub_ = nh_.subscribe("/dvs/eventStruct",1,&ContrastMaximizationRos::eventStructCallback,this);
}

void ContrastMaximizationRos::eventStructCallback(const dvs_msgs::EventStruct::Ptr& msg)
{
    int eventSize = msg->eventArr.data.size();
    int timeSize = msg->eventTime.data.size();
    if (timeSize>10000){
        torch::Tensor eventTemp = torch::tensor(msg->eventArr.data,option_.dtype(torch::kUInt8));
        torch::Tensor timeTemp = torch::tensor(msg->eventTime.data,option_.dtype(torch::kFloat32));
        // std::cout << "1st: " << timeTemp.index({0})-timeTemp.index({0}) << " , " << "2nd: " << timeTemp.index({timeSize-1})-timeTemp.index({0})  << std::endl;


        eventTemp = eventTemp.view({3,timeSize});
        // timeTemp = timeTemp.view({1,timeSize});

        eventPackets_.event = torch::vstack({eventTemp.index({Slice(0,2),Slice()}),torch::ones({1,timeSize},option_.dtype(torch::kUInt8))});
        eventPackets_.event = eventPackets_.event.to(eventPackets_.K.dtype());
        eventPackets_.event = torch::mm(eventPackets_.K.inverse(),eventPackets_.event);
        eventPackets_.eventTime = timeTemp.view({1,timeSize});;
        eventPackets_.polarityOn = eventTemp.index({2,Slice()});
        eventPackets_.polarityOff = eventTemp.index({2,Slice()}).logical_not().to(torch::kUInt8);

        opt_.runOptimiser(eventPackets_);

        
    }   


}

}
