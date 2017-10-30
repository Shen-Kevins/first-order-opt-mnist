#ifndef _UTILS_
#define _UTILS_

#include<armadillo>
#include "../define.h"

using namespace arma;



template<typename T>
T max_element(std::vector<T> vec){
    T max_val = vec[0];
    for(int i = 1; i < vec.size();i++){
        if(max_val < vec[i])
            max_val = vec[i];
    }
    return max_val;
}

mat vec2mat(const std::vector<double> &w,int begin,int row, int col){
    mat out(row,col);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            out(i,j) = w[i*col+j + begin];
        }
    }
    return out;
}

std::vector<double> mat2vec(const mat &matrix){
    int row = matrix.n_rows;
    int col = matrix.n_cols;
    std::vector<double> out(row*col,0);
    for(int i = 0;i < row; i++){
        for(int j = 0;j < col; j++){
            out[i*col+j] = matrix(i,j);
        }
    }
    return out;
}

mat relu(mat input){
    mat hidden(input.n_rows, input.n_cols,fill::zeros);
    for(int i = 0; i < input.size();i++){
        hidden(i) = input(i) > 0 ? input(i) : 0;
    }
    return hidden;
}


double least_forward(const mat &feature, const mat &weight, const mat &label){
    return accu(square(feature * weight - label)) / (2.0 * feature.n_rows);
}

mat least_backward(const mat &feature, const mat &weight, const mat label){
    return feature.t() *(feature *weight - label)/feature.n_rows;
}

double logistic_forward(const mat &feature, const mat &weight, const mat &label){
    double a = accu(label.t() * log(pow(1 + exp(- feature * weight), -1))) / feature.n_rows;
    double b = accu((1 - label.t()) * log(1- pow(1 + exp(- feature * weight), -1))) / feature.n_rows;
    return -(a + b);
}

mat logistic_backward(const mat &feature, const mat &weight, const mat &label){
    return -feature.t() * (label - pow( 1 + exp(- feature * weight), -1)) / (feature.n_rows);
}

double Softmax_forward(const mat &feature, const mat &weight, const mat &label){
    double loss = 0;
	for (int i = 0; i < feature.n_rows ;i++) {
		double tmp = accu(exp(weight * feature.row(i).t()));
		double prob = accu(exp(weight.row((int)label(i)) * feature.row(i).t())) / tmp;
		loss += log(prob);
	}
	return -loss / feature.n_rows;
}

mat Softmax_backward(const mat &feature, const mat &weight, const mat &label){
    mat grad(weight.n_rows, weight.n_cols);
	grad.zeros();
	double denomi = 0.0;
	for (int i = 0;i < feature.n_rows; i++) {
		denomi = accu(exp(weight * feature.row(i).t()));
		for (int j = 0; j < weight.n_rows; j++) {					
			double prob = accu(exp(weight.row(j)*feature.row(i).t())) / denomi;
			double ind = (label(i) == j ? 1 : 0);
			grad.row(j) += feature.row(i) *(ind - prob);
		}
	}
	return -grad / feature.n_rows;
}


double FCN_forward(const mat &feature, const mat &label, const std::vector<double> &weight){
    mat W1 = vec2mat(weight, 0, hidden_num, feature.n_cols);
    mat bia1 = vec2mat(weight, feature.n_cols * hidden_num, hidden_num, 1);
    mat W2 = vec2mat(weight, (feature.n_cols+1) * hidden_num, 10, hidden_num);

    double loss = 0.0;

    for(int i = 0; i < feature.n_rows; i++){
        mat hidden = relu(W1 * feature.row(i).t() + bia1);
        loss += Softmax_forward(hidden.t(), W2, label.row(i));
    }

    return loss / feature.n_rows;
}

std::vector<double> FCN_backward(const mat &feature, const mat &label, const std::vector<double> &weight){
    std::vector<double> grad(weight.size(),0);
    mat W1 = vec2mat(weight, 0, hidden_num, feature.n_cols);
    mat bia1 = vec2mat(weight, feature.n_cols * hidden_num, hidden_num, 1);
    mat W2 = vec2mat(weight, (feature.n_cols+1) * hidden_num, 10, hidden_num);

    mat W1_grad(W1.n_rows,W1.n_cols, fill::zeros);
    mat bia1_grad(bia1.n_rows,bia1.n_cols, fill::zeros);
    mat W2_grad(W2.n_rows, W2.n_cols, fill::zeros);

    mat hidden_grad(hidden_num, 1, fill::zeros);

    for(int i = 0; i < feature.n_rows; i++){
        mat hidden = relu(W1 * feature.row(i).t() + bia1);
        W2_grad += Softmax_backward(hidden.t(), W2, label.row(i));
        
        mat prob = exp(W2 * hidden) / accu(exp(W2 * hidden));
        
        hidden_grad = -W2.row(label(i)).t() + (prob.t() * W2).t();

        for(int j = 0; j < hidden_num; j++){
            if(hidden(j) == 0){
                W1_grad.row(j) += 0;
                bia1_grad(j) += 0;
            }else{
                W1_grad.row(j) += hidden_grad(j) * feature.row(i);
                bia1_grad(j) += hidden_grad(j);
            }
        }
    }

    std::vector<double> W1_grad1 = mat2vec(W1_grad);
    std::vector<double> bia1_grad1 = mat2vec(bia1_grad);
    std::vector<double> W2_grad1 = mat2vec(W2_grad);

    int num = feature.n_rows;

    for(int i = 0; i < W1_grad1.size(); i++)
        grad[i] = W1_grad1[i] / num;
    for(int i = 0; i < bia1_grad1.size();i++)
        grad[W1_grad1.size()+i] = bia1_grad1[i] / num;
    for(int i = 0; i < W2_grad1.size();i++)
        grad[W1_grad1.size()+bia1_grad1.size()+i] = W2_grad1[i] / num;
    
    return grad;
}
// mat sigmoid(){

// }





#endif