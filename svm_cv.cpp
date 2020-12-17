#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <dlib/svm.h>
#include <vector> 

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	// int a = 10;
	// float arr[a][a];

	// for(int i = 0; i < a; i++)
	// {
	// 	for(int j = 0; j < a; j++)
	// 		arr[i][j] = i + j;
	// }

	// Mat trainingDataMat(a, a, CV_32F, arr);

	// cout << trainingDataMat << endl;
	int labels[4] = {1, -1, -1, -1};
    float trainingData[4][2] = { {5, 1}, {25, 10}, {50, 25}, {10, 51} };

    Mat trainingDataMat(4, 2, CV_32F, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);

    cv::Mat test_mat(1, 2, CV_32F);
    Mat sampleMat = (Mat_<float>(1,2) << 25,10);


    Ptr<SVM> svm = SVM::create();

    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

    Mat output;

    svm->predict(test_mat,output, ml::StatModel::RAW_OUTPUT);

    int cls = svm->predict(test_mat);

    cout << output << endl;
    cout << cls << endl;

    // svm->save("trained-svm.xml");

    // Ptr<SVM> svmNew = Algorithm::load<SVM>("trained-svm.xml");

    // svmNew->predict(data_test, labels_SVM, ml::StatModel::RAW_OUTPUT);

    return 0;
}
