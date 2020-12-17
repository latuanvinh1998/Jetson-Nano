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
  
// Driver code 

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=1;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}

int main() 
{ 
    double *prob_estimates=NULL;
    struct svm_node *x;
    double predict_label;

    x = (struct svm_node *) malloc(513*sizeof(struct svm_node));

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);

    int *labels=(int *) malloc(nr_class*sizeof(int));
    svm_get_labels(model,labels);

    if((model=svm_load_model("../test.model"))==0)
    {
        fprintf(stderr,"can't open model file %s\n","test.model");
        exit(1);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////

    std::string model_path = "../model.pt";
    std::string image_path = "../test.jpg";

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

    cout << out_tensor.sizes() << endl;
    
    float prob = out_tensor[0][0].item<float>();

    ////////////////////////////////////////////////////////////////////////////////////////////

    

    prob_estimates = (double *) malloc(nr_class*sizeof(double));

    predict_label = svm_predict_probability(model,x,prob_estimates);

    cout << predict_label << endl;

    for(int i =0; i < nr_class; i++)
        cout << labels[i] << ":  " << prob_estimates[i] << endl;

    return 0; 
} 


for(int i = 0; i < 127; i++)
    {
        x[i].index = i;
        x[i].value = ((double) (i%2))/2;
        cout << x[i].index << ": " << x[i].value << endl;
    }

    x[780].index = -1;