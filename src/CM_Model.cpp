#include "cm_ros/CM_Model.h"

CM_Model::CM_Model( torch::TensorOptions &options, int sigma, int width, int height) : options_(options)
, amplitude(1/(torch::acos(torch::zeros(1, options))*4).sqrt_())
, indexRange(torch::arange(3+2*(sigma-1),options.dtype(torch::kInt)).unsqueeze(1))
, gaussLinspaceX(torch::linspace(-sigma,sigma,3+2*(sigma-1),options).repeat_interleave(3+2*(sigma-1)).unsqueeze(1))
, gaussLinspaceY(torch::linspace(-sigma,sigma,3+2*(sigma-1),options).repeat({3+2*(sigma-1)}).unsqueeze(1))
, kernel(amplitude*((gaussLinspaceX.square()+gaussLinspaceY.square())/-2).exp())
, N((height+2*sigma)*(width+2*sigma))
, w_(width)
, h_(height)
, k_(3+2*(sigma-1))
, sigma(sigma)
{
    a = register_parameter("a", torch::zeros({1,1}, options));
}


torch::Tensor CM_Model::forward(eventPacket &eventPackets)
{
    using slice = torch::indexing::Slice;
    // Project events into image space
    torch::Tensor eventTemp = propagateEvents(eventPackets,a);
    // Calculate event influence
    torch::Tensor eventHnorm = eventTemp.div(eventTemp.index({2,slice()}));
    torch::Tensor eventRnd = eventHnorm.round().to(torch::kInt);
    torch::Tensor mask =    
    /* mask of 0<=x<=239*/  eventRnd.index({0,slice()}).le(w_-1) * 
                            eventRnd.index({0,slice()}).ge(0) * 
    /* mask of 0<=y<=179*/  eventRnd.index({1,slice()}).le(h_-1) * 
                            eventRnd.index({1,slice()}).ge(0);
    return -gaussEventImage(eventHnorm,eventRnd,mask).var();
}


inline torch::Tensor CM_Model::propagateEvents(eventPacket &eventPackets,torch::Tensor a)
{
    torch::Tensor b = torch::vstack({torch::zeros({2,1},a.options()),a});

    // torch::Tensor output = eventPackets.K.mm(eventPackets.event + torch::cross(b.mm(eventPackets.eventTime),eventPackets.event));
    torch::Tensor output = torch::mm(eventPackets.K,eventPackets.event + torch::cross(torch::mm(b,eventPackets.eventTime),eventPackets.event));
    
    return output;
}

inline torch::Tensor CM_Model::gaussEventImage( torch::Tensor &eventHnorm, torch::Tensor &eventRnd, torch::Tensor &mask)
{
    using slice = torch::indexing::Slice;
    torch::Tensor GaussTemp =   amplitude*
                                (-((eventRnd.index( {0,slice()}).masked_select(mask) + 
                                    gaussLinspaceX - 
                                    eventHnorm.index({0,slice()}).masked_select(mask)).square()+ 
                                (eventRnd.index({1,slice()}).masked_select(mask) + 
                                    gaussLinspaceY - 
                                    eventHnorm.index({1,slice()}).masked_select(mask)).square())/2).exp();

    torch::Tensor eventImage = torch::zeros({N},options_);
    torch::Tensor index = ((eventRnd.index({1,slice()}).masked_select(mask) + indexRange).repeat({k_,1})
                            *(w_+2*sigma)+
                            (eventRnd.index({0,slice()}).masked_select(mask) + indexRange).repeat_interleave(k_,0))
                            .flatten();

    eventImage.index_add_(0,index,GaussTemp.flatten());
    
    return eventImage.view({h_+2*sigma,w_+2*sigma}).index({{slice(sigma,h_+sigma),slice(sigma,w_+sigma)}}); 

}

void CM_Model::printInfo()
{
    std::cout  << "Height: " << h_ << "   " << "Width: " << w_ << "   " << "Sigma: " << sigma << "   "<< "k: " << k_ << std::endl;
}

