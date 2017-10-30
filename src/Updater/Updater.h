#ifndef _UPDATER_
#define _UPDATER_

#include "../Model/Model.h"
#include "../Utils/Utils.h"

class Updater {
protected:
    Model *model;
    Datapoint *datapoint;

public:
    Updater(){}

    Updater(Model *model, Datapoint *datapoint){
        this->model = model;
        this->datapoint = datapoint;
    }

    void ApplyGradient(){
        // mat gradient = least_backward(datapoint->GetFeature(),vec2mat(model->GetWeight(),datapoint->GetFeature().n_cols,datapoint->GetLabel().n_cols),datapoint->GetLabel());
        // vector<double> gradients = mat2vec(gradient);
        // // gradients += L2_lambda * model->GetWeight();
        // for(int i = 0; i < model->GetSize(); i++){
        //     gradients[i] += L2_lambda * model->GetWeight()[i];
        // }

        // model->UpdateWeight(learning_rate,gradients);
    }
    
    void ApplySGD(int loss_type){
        int begin = rand()%(datapoint->GetSize()-mini_batch);
        vector<double> gradients;

        //calculate gradient
        if(loss_type == 1){//logistic
            mat gradient = logistic_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                vec2mat(model->GetWeight(),0,784,1),datapoint->GetLabelsRols(begin,begin+mini_batch));
            gradients = mat2vec(gradient);
        }else if(loss_type == 2){//softmax
            mat gradient = Softmax_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                vec2mat(model->GetWeight(),0,10,784),datapoint->GetLabelsRols(begin,begin+mini_batch));
            gradients = mat2vec(gradient);
        }else if(loss_type == 3){//FCN 784*100*10
            gradients = FCN_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                datapoint->GetLabelsRols(begin, begin+mini_batch), model->GetWeight());
        }

        for(int i = 0; i < gradients.size(); i++){
            gradients[i] += L2_lambda * model->GetWeight()[i];
        }
        model->UpdateWeight(learning_rate,gradients);
    }

    //0.267217842
    void ApplySVRG(int loss_type){
        
        std::vector<double> out_weight;
        out_weight.resize(this->model->GetSize());
        
        std::vector<double> weight = this->model->GetWeight();
        for(int i = 0; i < out_weight.size(); i++){
            out_weight[i] = weight[i];
        }
       
        mat out_grad_mat;
        vector<double> out_grad;
        
        switch(loss_type){
            case 1:
                out_grad_mat = logistic_backward(datapoint->GetFeature(),vec2mat(model->GetWeight(),0,784,1),datapoint->GetLabel());
                out_grad = mat2vec(out_grad_mat);
                break;
            case 2:
                out_grad_mat = Softmax_backward(datapoint->GetFeature(),vec2mat(model->GetWeight(),0,10,784),datapoint->GetLabel());
                out_grad = mat2vec(out_grad_mat);
                break;
            case 3:
                out_grad = FCN_backward(datapoint->GetFeature(),datapoint->GetLabel(),model->GetWeight());
                break;
            default:cout<<"error"<<endl;break;
        }

        for(int i = 0; i < datapoint->GetSize() / mini_batch; i++){
            int begin = rand()%(datapoint->GetSize()-mini_batch);
            vector<double> in_gradients, in_gradients1;
            mat gradient_mat;
            switch(loss_type){
                case 1:
                    gradient_mat = logistic_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                        vec2mat(model->GetWeight(),0,784,1),datapoint->GetLabelsRols(begin,begin+mini_batch));
                    in_gradients = mat2vec(gradient_mat);

                    gradient_mat = logistic_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                        vec2mat(out_weight,0,784,1),datapoint->GetLabelsRols(begin,begin+mini_batch));
                    in_gradients1 = mat2vec(gradient_mat);
                    break;
                case 2:
                    gradient_mat = Softmax_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                        vec2mat(model->GetWeight(),0,10,784),datapoint->GetLabelsRols(begin,begin+mini_batch));
                    in_gradients = mat2vec(gradient_mat);

                    gradient_mat = Softmax_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                        vec2mat(out_weight,0,10,784),datapoint->GetLabelsRols(begin,begin+mini_batch));
                    in_gradients1 = mat2vec(gradient_mat);
                    break;
                case 3:
                    in_gradients = FCN_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                        datapoint->GetLabelsRols(begin,begin+mini_batch),model->GetWeight());
                    in_gradients1 = FCN_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
                        datapoint->GetLabelsRols(begin,begin+mini_batch),out_weight);
                    break;
                default:cout<<"error"<<endl;break;
            }           
            for(int i = 0; i < in_gradients.size(); i++){
                in_gradients[i] = in_gradients[i] - in_gradients1[i] + out_grad[i] +  L2_lambda * model->GetWeight()[i];
            }
            
            model->UpdateWeight(learning_rate,in_gradients);
        }
    }

    int ApplyADASVRG(int epoch){
        std::vector<double> out_weight;
        out_weight.resize(this->model->GetSize());
        std::vector<double> weight = this->model->GetWeight();
        for(int i = 0; i < out_weight.size(); i++){
            out_weight[i] = weight[i];
        }

        int m = 400;

        int index = (epoch-1) / 1;

        m = m * std::pow(2,index);
        // m = m * 

        m = m > (datapoint->GetSize()-1)?(datapoint->GetSize()-1):m;

        // m = 60000-1;

        vector<double> out_grad = FCN_backward(datapoint->GetFeaturesRows(0,m),datapoint->GetLabelsRols(0,m),model->GetWeight());
        
        // cout<<"a"<<endl;

        for(int i = 0; i < m / mini_batch; i++){
            int begin = rand()%(m-mini_batch);
            vector<double> in_gradients = FCN_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
            datapoint->GetLabelsRols(begin,begin+mini_batch),model->GetWeight());
            
            vector<double> in_gradients1 = FCN_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),\
            datapoint->GetLabelsRols(begin,begin+mini_batch),out_weight);
            
            for(int i = 0; i < in_gradients.size(); i++){
                in_gradients[i] = in_gradients[i] - in_gradients1[i] + out_grad[i] +  1.0 / std::sqrt( (double)m) * model->GetWeight()[i];
            }
            
            model->UpdateWeight(learning_rate,in_gradients);
        }
        return m;
    }

    ~Updater(){}
};

#endif