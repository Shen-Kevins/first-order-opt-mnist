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

    Datapoint *datapoint = new LogisticDatapoint("../data/mnist/");
    Model *model = new L2Model(784*100 + 100 + 100*10);

    double loss =  FCN_forward(datapoint->GetFeature(),datapoint->GetLabel(),model->GetWeight());
    cout<<loss<<endl;
    for(int j=0;j<10;j++){
        for(int i = 0;i<60000-10;i+=10){
            vector<double> grad = FCN_backward(datapoint->GetFeaturesRows(i,i+10),datapoint->GetLabelsRols(i,i+10),model->GetWeight());

            model->UpdateWeight(learning_rate,grad);
        }
        loss =  FCN_forward(datapoint->GetFeature(),datapoint->GetLabel(),model->GetWeight());

        cout<<loss<<endl;
    }
    
    return 0;
}