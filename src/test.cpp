#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include "svm.h"

using namespace cv;
using namespace std; 


struct svm_node *x;

struct svm_model* model;


int main() 
{ 
    double *prob_estimates=NULL;
    struct svm_node *x;
    double predict_label;

    if((model=svm_load_model("../Data/svm.model"))==0)
    {
        fprintf(stderr,"can't open model file %s\n","svm.model");
        exit(1);
    }

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);

    int *labels=(int *) malloc(nr_class*sizeof(int));
    svm_get_labels(model,labels);

    ////////////////////////////////////////////////////////////////////////////////////////////

    std::string model_path = "../Data/model.pt";
    std::string image_path = "../Data/test.jpg";

    torch::jit::script::Module module = torch::jit::load(model_path);

    module.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs;

    Mat image = imread(image_path);

    resize(image, image, Size(112,112));

    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    auto input_tensor = torch::from_blob(image.data, {1, 112, 112, 3});

    input_tensor = input_tensor.permute({0, 3, 1, 2});

    input_tensor[0][0] = input_tensor[0][0].sub_(0.5).div_(0.5);
    input_tensor[0][1] = input_tensor[0][1].sub_(0.5).div_(0.5);
    input_tensor[0][2] = input_tensor[0][2].sub_(0.5).div_(0.5);

    input_tensor = input_tensor.to(at::kCUDA);

    torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();

    // cout << out_tensor.sizes() << endl;
    
    // double prob = out_tensor[0][1].item<double>();
    // cout << prob << endl;
    // cout << out_tensor << endl;

    ////////////////////////////////////////////////////////////////////////////////////////////


    x = (struct svm_node *) malloc(513*sizeof(struct svm_node));

    for(int i = 0; i < 512; i++)
    {
        x[i].index = i;
        x[i].value = out_tensor[0][i].item<double>();
        // cout << x[i].index << ": " << x[i].value << endl;
    }

    x[512].index = -1;

    

    prob_estimates = (double *) malloc(nr_class*sizeof(double));

    predict_label = svm_predict_probability(model,x,prob_estimates);

    cout << "Predict class " << predict_label << " with " << 100 * prob_estimates[(int) predict_label] << "%" << " confidence!" <<endl;

    // for(int i =0; i < nr_class; i++)
    //     cout << labels[i] << ":  " << prob_estimates[i] << endl;

    return 0; 
} 
