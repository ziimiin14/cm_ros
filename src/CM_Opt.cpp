#include "cm_ros/CM_Opt.h"


CM_Opt::CM_Opt(CM_Model &model, torch::TensorOptions &options) : options_(options), model_(model)
{
    lastLoss= torch::zeros(1,options_);
    patience = 0;
    options_solve = torch::optim::AdagradOptions().lr(0.01).lr_decay(0.02);
    model_.printInfo();
}

void CM_Opt::optimiser(eventPacket &eventPackets)
{   


    // std::cout << "Done 1"<< std::endl;

    // Instantiate optimizer
    torch::optim::Adagrad optimizerCM(model_.parameters(), options_solve);
    // std::cout << "Done 2"<< std::endl;

    for (size_t epoch = 1; epoch <= 500; ++epoch){
        // std::cout << "Done 3"<< std::endl;

        // Reset Gradients
        optimizerCM.zero_grad();
        // std::cout << "Done 4"<< std::endl;

        // Compute Loss
        loss = model_.forward(eventPackets);
        // std::cout << "Done 5"<< std::endl;

        // Compute gradients of the loss w.r.t. the parameters of our model.
        loss.backward();
        // std::cout << "Done 6"<< std::endl;


        // Update the parameters based on the calculated gradients.
        optimizerCM.step();
        // std::cout << "Done 7"<< std::endl;

        // Terminate if loss is 1e-3 for 7 epochs
        if ((lastLoss - loss).abs().lt(1e-3).item<bool>()){
            ++patience;
            if (patience==7){
                std::cout << "Termination criteria reached" << std::endl;
                std::cout << "Epoch: " << epoch
                    << " | Loss: " << loss.item<float>() 
                    << std::endl;
                break;
            }
        } else{
            patience = 0;
        }
        lastLoss = loss;
        
    }
    std::cout << model_.a;

}    

void CM_Opt::runOptimiser(eventPacket &eventPackets)
{
    std::cout << "Done"<< std::endl;
    optimiser(eventPackets);
}



