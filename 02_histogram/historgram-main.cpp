#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>
	



int main(int argc, char* argv[]) {

	cv::Mat flowers = cv::imread("flower.jpg", cv::IMREAD_COLOR);

	if (flowers.empty()) {
		std::cout << "image not founded!" << std::endl;
		return -1;
	}

	cv::imshow("flowers.jpg", flowers);

	cv::Mat histMat(300, 512, CV_8UC3, cv::Scalar(120, 120, 120));
	histogramMat(flowers, histMat, 2);
	cv::imshow("histogram", histMat);

	cv::Mat eqMat = equalizeMat(flowers);
	cv::imshow("color equalized", eqMat);

	cv::Mat eqHistMat(300, 512, CV_8UC3, cv::Scalar(120, 120, 120));
	histogramMat(eqMat, eqHistMat);
	cv::imshow("equalized histogram", eqHistMat);


	cv::waitKey();

	return 0;
}


