#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <vector>

// using namespace std;

int main(int argc, const char* argv[]) {
	// if (argc != 2) {
	// std::cerr << "usage: example-app <path-to-exported-script-module>\n";
	// return -1;
	// }


	// torch::jit::script::Module module;
	// try {
	// // Deserialize the ScriptModule from a file using torch::jit::load().
	// module = torch::jit::load(argv[1]);
	// }
	// catch (const c10::Error& e) {
	// std::cerr << "error loading the model\n";
	// return -1;
	// }

	// std::cout << "ok\n";

	// std::vector<torch::jit::IValue> inputs;

	// inputs.push_back(torch::ones({1, 3, 112, 112}).to(at::kCUDA));

	// std::cout << inputs << std::endl;

	// // Execute the model and turn its output into a tensor.
	// at::Tensor output = module.forward(inputs).toTensor();
	// std::cout << output << '\n';

	double array[] = {1, 2, 3, 4, 5};
	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor tharray = torch::from_blob(array, {5}, options).to(at::kCUDA);

	std::cout << tharray << std::endl;
}