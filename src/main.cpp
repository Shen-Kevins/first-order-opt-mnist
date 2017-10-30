#include<iostream>
#include<armadillo>
#include "./Datapoint/LogisticDatapoint.h"
#include "./Model/L2Model.h"
#include "./Updater/Updater.h"
#include "./Trainer/Trainer.h"
#include "./Utils/Utils.h"

#include<time.h>


using namespace std;
using namespace arma;

int main(int argc,char **argv){

    srand((unsigned)time(NULL));

    int update_type = 2;    // 1:SGD, 2:SVRG, 3:ADA-SVRG
    int loss_type = 1;      // 1:logistic, 2:Softmax, 3:FCN 784*100*10

    if(loss_type == 1) in_inter = 10000;

    Datapoint *datapoint = new LogisticDatapoint("../data/mnist/", loss_type);
    Model *model = new L2Model(784);
    // cout<<datapoint->GetSize()<<endl;
    Trainer *trainer = new Trainer(model, datapoint);
    trainer->Train(update_type, loss_type);
    return 0;
}