#ifndef _MODEL_
#define _MODEL_

#include<iostream>
#include<armadillo>
#include "../Datapoint/Datapoint.h"
#include "../define.h"
#include "../Utils/Utils.h"

using namespace std;
using namespace arma;

class Model{
protected:
    int size;
    vector<double> weight;

public:
    Model(int length){
        weight.resize(length,0.01);
        for(int i = 0; i < length; i++){
            weight[i] = (rand() % length - length / 2) / (length * 10 + 0.01);
        }
        size = length;
    }

    int GetSize(){
        return this->size;
    }

    vector<double>& GetWeight(){
        return this->weight;
    }

    void SetWeight(vector<double> &weights){
        this->weight = weights;
    }

    double ComputeL2Loss(){
        double L2loss = 0.0;
        for(int i = 0; i < this->size; i++){
            L2loss += weight[i] * weight[i];
        }

        return 0.5 * L2_lambda * L2loss;
    }

    void UpdateWeight(double learning_rate,vector<double> &gradient){
        for(int i = 0; i < weight.size(); i++)
            weight[i] -=learning_rate * gradient[i];
    }

    // void Save_weight(){
    //     this->weight.save("../../data/generated/least_train_weight_mat");
    // }

    virtual double ComputeLoss(Datapoint *datapoint, int loss_type) = 0;
};

#endif