
# Face ID with Jetson Nano
### Using MTCNN and MobileFacenet architecture with ArcFace with C++

## Required:
- Libtorch
- OpenCV 3.x.x
- OpenBlas
- Cuda/Cudnn
- TensorRT
 
## Installation:
For Pytorch. Installation follow link:
```bash
https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
```
Next change the path {user} with your user name in CMakeList.txt. 
In this repository, I change the name of CMakeList for Jetson Nano to CMakeListJetson.txt

For MTCNN, follow link:
```bash
https://github.com/AlphaQi/MTCNN-light
```
## Face Classifier:
Using LibSVM for classification. 
LibSVM can train on multi-class and can produce probability for each class - which can use for detect Unknown Identification. For more detail, see:
```bash
https://github.com/cjlin1/libsvm
```
I have edited code in LibSVM so when processing, our program doesn't need to write data to txt file. Make it easier and faster to process.

## Compile and run:
```bash
cmake .
make
./main
```
### Hope this can help you guys

