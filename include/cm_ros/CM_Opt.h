#ifndef CM_OPT_H
#define CM_OPT_H

#include "cm_ros/CM_Model.h"
#include "cm_ros/EventPacket.h"
#include <torch/torch.h>


class CM_Opt 
{
    public:
        
        CM_Opt(CM_Model &model, torch::TensorOptions &options);

        void runOptimiser(eventPacket &eventPackets);

    private:
        torch::Tensor pred, loss;
        torch::Tensor lastLoss;
        int patience;
        torch::optim::AdagradOptions options_solve;
        torch::TensorOptions options_;
        CM_Model model_;
        void optimiser(eventPacket &eventPackets);

};

#endif // CM_OPT_H
