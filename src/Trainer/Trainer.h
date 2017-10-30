#ifndef _TRAINER_
#define _TRAINER_

#include "../Model/Model.h"
#include "../Datapoint/Datapoint.h"
#include "../Updater/Updater.h"
#include<time.h>

class Trainer{
protected:
    Model *model;
    Datapoint *datapoint;
    Updater *updater;

public:
    Trainer(){}

    Trainer(Model *model, Datapoint *datapoint){
        this->model = model;
        this->datapoint = datapoint;
        this->updater = new Updater(model,datapoint);
    }

    void Train(int update_type, int loss_type){
        /**
         * update_type: 1:SGD, 2:SVRG, 3:ADA-SVRG
         * 
         * loss_type: 1:logistic, 2:softmax, 3: FCN 784*100*10
         * 
         * */
        int epoch = 0;
        int num = 0;
        double accu_IFO = 0.0;

        if(loss_type == 1)
            num = 12665;
        else
            num = 60000;

        if(update_type == 2)
            num *= 3;

        auto time_begin = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> accu_time = time_begin - time_begin;

        double loss = this->model->ComputeLoss(this->datapoint, loss_type);

        printf("time      loss          accu_IFO\n");
        printf("%.5f   %.9f   %f\n",accu_time.count(),loss,accu_IFO);

        while(epoch < epoch_num){
            epoch += 1;
            auto time_start = std::chrono::high_resolution_clock::now();

            if(update_type == 1){//SGD
                for(int iter = 0; iter < in_inter / mini_batch; iter++){
                            this->updater->ApplySGD(loss_type);
                    }
            }else if(update_type == 2){//SVRG,前2个epoch用SGD pre-train
                if(epoch < 3){
                    for(int iter = 0; iter < in_inter / mini_batch; iter++){
                        this->updater->ApplySGD(loss_type);
                    }
                }else{
                    this->updater->ApplySVRG(loss_type);
                }
            }else if(update_type == 3){//ADA-SVRG，自适应增加train samples
                if(epoch < 3){
                    for(int iter = 0; iter < in_inter / mini_batch; iter++){
                        this->updater->ApplySGD(loss_type);
                    }
                }else{
                    num = this->updater->ApplyADASVRG(epoch);
                }
            }
            auto time_end = std::chrono::high_resolution_clock::now();

            accu_time += (time_end - time_start);

            loss = this->model->ComputeLoss(this->datapoint, loss_type);
            accu_IFO += (double)num / datapoint->GetSize();
            printf("%.5f   %0.9f   %f\n",accu_time.count(),loss,accu_IFO);
        }
    }
};


#endif