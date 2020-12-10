#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

int main() {

	std::string model_path = "../model.pt";
    std::string image_path = "../test.jpg";

    torch::jit::script::Module module = torch::jit::load(model_path);
	std::cout << "Switch to GPU mode" << std::endl;

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
    
    // auto out_tensor_cvt = out_tensor;
    // auto predict = std::get<0>(out_tensor);
    float prob = out_tensor[0][0].item<float>();
    cout << prob << endl;
}