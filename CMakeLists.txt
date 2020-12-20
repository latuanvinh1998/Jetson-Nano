cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_PREFIX_PATH /home/vinh/libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(example-app src/test.cpp src/svm.h src/svm.cpp)
#add_executable(example-app src/dir.cpp src/svm.h src/svm.cpp)
#add_executable(example-app src/test_predict.cpp src/svm.h src/svm.cpp)
#add_executable(example-app src/test_train.cpp src/svm.h src/svm.cpp)

target_link_libraries(example-app ${OpenCV_LIBS})
target_link_libraries(example-app ${TORCH_LIBRARIES})
target_link_libraries(example-app stdc++fs)