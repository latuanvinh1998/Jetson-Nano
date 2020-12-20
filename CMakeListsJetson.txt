cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

include_directories(/home/tanphatnguyen/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include)
include_directories(/home/tanphatnguyen/.local/lib/python3.6/site-packages/torch/include)

link_directories(/home/tanphatnguyen/.local/lib)
link_directories(/home/tanphatnguyen/.local/lib/python3.6/site-packages/torch/lib)

set(CMAKE_PREFIX_PATH /home/tanphatnguyen/.local/lib/python3.6/site-packages/torch)
set(Torch_DIR /home/tanphatnguyen/.local/lib/python3.6/dist-packages/torch)

#OpenBlas
include_directories(/home/tanphatnguyen/OpenBLAS/include)
link_directories(/home/tanphatnguyen/OpenBLAS/lib)

#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(example-app test.cpp svm.h svm.cpp)
#add_executable(example-app test_predict.cpp svm.h svm.cpp)
#add_executable(example-app test_train.cpp svm.h svm.cpp)

target_link_libraries(example-app ${OpenCV_LIBS})
target_link_libraries(example-app ${TORCH_LIBRARIES})

target_link_libraries(example-app openblas)
target_link_libraries(example-app cudart)

set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

