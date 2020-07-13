#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>

int main(int argc, char* argv[]) {

	 cv::Mat inputMat = cv::imread("flowers.jpg", cv::IMREAD_COLOR);
	cv::imshow("Flowers.jpg", inputMat);
	/*
	cartoonFilter(inputMat, inputMat);
	cv::imshow("cartoon effect", inputMat);

	cv::Mat image = cv::imread("flower.jpg");
	cv::imshow("flower", image);

	cv::Mat meanBlur = smoothFilter(image, SmoothType::mean, 7);
	cv::imshow("blur", meanBlur);

	cv::Mat boxBlur = smoothFilter(image, SmoothType::box, 7);
	cv::imshow("boxblur", boxBlur);
	

	cv::Mat medianBlur = smoothFilter(image, SmoothType::median, 7);
	cv::imshow("medianblur", medianBlur);

	cv::Mat guassianBlur = smoothFilter(image, SmoothType::guassian, 7);
	cv::imshow("guassianblur", guassianBlur);

	cv::Mat bilateralBlur = smoothFilter(image, SmoothType::bilateral, 7);
	cv::imshow("bilateralblur", bilateralBlur);

	*/

	cv::Mat sobel;
	cv::Mat gray;
	cv::cvtColor(inputMat, gray, cv::COLOR_BGR2GRAY);
	cv::Sobel(gray, sobel, CV_16S, 1, 1, 5);
	cv::imshow("sobel", sobel);

	cv::waitKey();

	cv::destroyAllWindows();

	return 0;
}