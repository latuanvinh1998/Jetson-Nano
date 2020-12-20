#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <string>

namespace fs = std::experimental::filesystem;

using namespace std;

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
	ofstream myfile;
	myfile.open ("paths.txt");

	for(int temp = 0; temp < len_of_folders_in_directory; temp++){
		int len_folder = find_length(directory[temp]);
		for(int temp1 = 0; temp1 < len_folder; temp1++){
			myfile << directory[temp][temp1] << endl;
		}
	}

	myfile.close();
}

void write_file_name_only(string* directory, string input){
	ofstream myfile;
	myfile.open ("names.txt");

	string* name_only = new string[len_of_folders_in_directory];
	for(int temp = 0; temp < len_of_folders_in_directory; temp++){
		int pos1 = findNthOccur(directory[temp], '/', 1);
		name_only[temp] = (directory[temp]).substr(pos1 + 1);
		myfile << temp << " : " << name_only[temp] << endl;
	}

	myfile.close();

	string** read_input = read_directory(input);
	string* name_cut = new string[10000];
	int k = 0;
	for(int temp = 0; temp < len_of_folders_in_directory; temp++){
		int len_of_each_folder = find_length(read_input[temp]);
		for(int temp1 = 0; temp1 < len_of_each_folder; temp1++){
			int pos1 = findNthOccur(read_input[temp][temp1], '/', 1);
			int pos2 = findNthOccur(read_input[temp][temp1], '/', 2);
			int len_for_sub = pos2 - pos1;
			string name_temp = (read_input[temp][temp1]).substr(pos1 + 1, len_for_sub - 1);
			name_cut[k] = name_temp;
			k += 1;
		}
	}

	ofstream myfile1;
	myfile1.open ("encode_name.txt");

	for(int temp = 0; temp < k; temp++){
		for(int temp1 = 0; temp1 < len_of_folders_in_directory; temp1++){
			if(name_cut[temp].compare(name_only[temp1]) == 0){
				myfile1 << temp1 << endl;
			}
		}
	}

	myfile1.close();
}

int main() {
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

	return 0;
}
