#ifndef _L2MODEL_
#define _L2MODEL_

#include "./Model.h"

class L2Model : public Model{
public:
    L2Model(int length) : Model(length){}  

    double ComputeLoss(Datapoint *datapoint, int loss_type) override{
        double loss = 0;
        if(loss_type == 1)
            loss = logistic_forward(datapoint->GetFeature(),vec2mat(this->weight,0 ,784, 1),datapoint->GetLabel());
        else if(loss_type == 2)
            loss = Softmax_forward(datapoint->GetFeature(),vec2mat(this->weight,0 , 10,784),datapoint->GetLabel());
        else if(loss_type == 3)
            loss =  FCN_forward(datapoint->GetFeature(),datapoint->GetLabel(),this->weight);

        return loss + ComputeL2Loss();
    }
};

#endif