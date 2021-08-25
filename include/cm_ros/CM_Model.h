#ifndef CM_MODEL_H
#define CM_MODEL_H

#include "cm_ros/EventPacket.h"
#include <torch/torch.h>


struct CM_Model : torch::nn::Module 
{
        public:
                CM_Model(torch::TensorOptions &options, int sigma, int width, int height);

                torch::Tensor forward(eventPacket &eventPackets);
                torch::Tensor a;
        private:
                
                torch::Tensor event_homogeneous;
                torch::TensorOptions options_;
                // Fixed unchanging values
                torch::Tensor amplitude, indexRange, gaussLinspaceX, gaussLinspaceY;
                torch::Tensor kernel;
                // Size of eventImage + buffer
                int N;
                const int sigma,w_, h_,k_; // width, height, kernel width
                
                torch::Tensor gaussEventImage(torch::Tensor &eventHnorm, torch::Tensor &eventRnd, torch::Tensor &mask);
                torch::Tensor propagateEvents(eventPacket &eventPackets,torch::Tensor a);


};


#endif // CM_MODEL_H
