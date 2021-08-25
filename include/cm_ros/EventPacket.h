#ifndef EVENTPACKET_H
#define EVENTPACKET_H
#include <torch/torch.h>


struct eventPacket {
    torch::Tensor event;
    torch::Tensor eventTime;
    torch::Tensor K;
    torch::Tensor polarityOn;
    torch::Tensor polarityOff;
};

#endif // EVENTPACKET_H
