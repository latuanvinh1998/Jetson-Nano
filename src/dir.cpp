#include <iostream>
#include <errno.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <memory>

#include <experimental/filesystem>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>
#include "svm.h"

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

int i = 0;
static int len_of_folders_in_directory;

int find_length(string* paths){
	int temp = 0;
	while(!paths[temp].empty()){
		++temp;
	}
	return temp;
}

void read_files_inside_one_directory(string*& paths, string read_path){
	i = 0;
	for(const auto & entry : fs::directory_iterator(read_path)){
		paths[i] = entry.path();
		i += 1;
	}

	// Xuat files trong 1 folder
	// for(int temp = 0; temp < i; temp++){
	// 	cout << paths[temp] << endl;
	// }
}

string** read_directory(string read_path){
	string* directory = new string[100];
	read_files_inside_one_directory(directory, read_path);

	len_of_folders_in_directory = find_length(directory);

	string** folders = new string*[len_of_folders_in_directory];
	cout << "len_of_folders_in_directory: " << len_of_folders_in_directory << endl;
	for (int temp = 0; temp < len_of_folders_in_directory; temp++){
		folders[temp] = new string[1000];
	}

	for(int temp = 0; temp < len_of_folders_in_directory; temp++){
		cout << "Reading this folder: " << directory[temp] << endl;
		read_files_inside_one_directory(folders[temp], directory[temp]);
	}

	// Xuat files trong folders trong directory_iterator
	// for(int temp = 0; temp < len_of_folders_in_directory; temp++){
	// 	int len_of_each_folder = find_length(folders[temp]);
	// 	cout << "len_of_each_folder: " << len_of_each_folder << endl;
	// 	for(int temp1 = 0; temp1 < len_of_each_folder; temp1++){
	// 		cout << folders[temp][temp1] << endl;
	// 	}
	// }

	return folders;
}

int findNthOccur(string str, char ch, int N) 
{ 
    int occur = 0; 
  
    // Loop to find the Nth 
    // occurence of the character 
    for (int i = 0; i < str.length(); i++) { 
        if (str[i] == ch) { 
            occur += 1; 
        } 
        if (occur == N) 
            return i; 
    } 
    return -1; 
}

void write_file_txt(string** directory){
	ofstream myfile, labels;

	myfile.open ("../Data/paths.txt");
	labels.open("../Data/labels.txt");

	for(int temp = 0; temp < len_of_folders_in_directory; temp++){
		int len_folder = find_length(directory[temp]);
		for(int temp1 = 0; temp1 < len_folder; temp1++){
			myfile << directory[temp][temp1] << endl;
			labels << temp << endl;
		}
	}

	myfile.close();
	labels.close();
}

void write_file_name_only(string* directory, string input){
	ofstream myfile;
	myfile.open ("../Data/names.txt");

	string* name_only = new string[len_of_folders_in_directory];
	for(int temp = 0; temp < len_of_folders_in_directory; temp++){
		int pos1 = findNthOccur(directory[temp], '/', 2);
		name_only[temp] = (directory[temp]).substr(pos1 + 1);
		myfile << name_only[temp] << endl;
	}

	myfile.close();
}

void create_txt()
{
	string** paths = new string*[100];
	for(int temp = 0; temp < 3; temp++){
		paths[temp] = new string[1000];
	}

	// string path = "input";
	paths = read_directory("../Processed");
	write_file_txt(paths);

	string* directory = new string[100];
	read_files_inside_one_directory(directory, "../Processed");
	write_file_name_only(directory, "../Processed");

	cout << "Finished" << endl;
}


int main() {

	ifstream path,label;
	ofstream train;
	string line_path, line_label;

	vector <string> paths;
	vector <int> labels;

	path.open("../Data/paths.txt");
	label.open("../Data/labels.txt");

	while(getline(path, line_path))
		paths.push_back(line_path);


	while(getline(label, line_label))
		labels.push_back(stoi(line_label));


	if(labels.size() != paths.size())
	{
		cout << "Wrong input!" << endl;
		return -1;
	}

	/////////////////////////////////Create Data for SVM///////////////////////////////////

	string model_path = "../Data/model.pt";

	torch::jit::script::Module module = torch::jit::load(model_path);
	module.to(at::kCUDA);

	std::vector<torch::jit::IValue> inputs;

	Mat image;

	train.open("../Data/train");

	for(int i = 0; i < paths.size(); i++)
	{
		image = imread(paths[i]);

	    resize(image, image, Size(112,112));

	    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

	    auto input_tensor = torch::from_blob(image.data, {1, 112, 112, 3});

	    input_tensor = input_tensor.permute({0, 3, 1, 2});

	    input_tensor[0][0] = input_tensor[0][0].sub_(0.5).div_(0.5);
	    input_tensor[0][1] = input_tensor[0][1].sub_(0.5).div_(0.5);
	    input_tensor[0][2] = input_tensor[0][2].sub_(0.5).div_(0.5);

	    input_tensor = input_tensor.to(at::kCUDA);

	    torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();

	    train << labels[i] << " ";
	    for(int j =  0; j < 512; j++)
	    	train << j << ":" << out_tensor[0][j].item<float>() << " ";
	    train << endl;
	}

	return 0;
}