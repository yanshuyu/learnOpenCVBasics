#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>

int main(int argc, char* argv[]) {

	cv::Mat inputMat = cv::imread("flowers.jpg", cv::IMREAD_COLOR);
	cv::imshow("Flowers.jpg", inputMat);

	cartoonFilter(inputMat, inputMat);
	cv::imshow("cartoon effect", inputMat);
	
	cv::waitKey();

	return 0;
}