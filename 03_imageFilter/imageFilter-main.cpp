#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>


int main(int argc, char* argv[]) {

	 cv::Mat inputMat = cv::imread("noise.jpg", cv::IMREAD_COLOR);
	cv::imshow("orignal.jpg", inputMat);

	//
	// sobel operator find image gradient(liner convolution)
	//
	cv::Mat sobel_x;
	cv::Mat sobel_y;
	cv::Mat sobel;
	cv::Mat gray;
	
	cv::cvtColor(inputMat, gray, cv::COLOR_BGR2GRAY);
	cv::Sobel(gray, sobel_x, CV_16S, 1, 0);
	cv::Sobel(gray, sobel_y, CV_16S, 0, 1);
	cv::convertScaleAbs(sobel_x, sobel_x);
	cv::convertScaleAbs(sobel_y, sobel_y);
	cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, sobel);
	cv::imshow("sobel", sobel);

	cv::Mat lap;
	cv::Laplacian(gray, lap, CV_16S, 3);
	cv::convertScaleAbs(lap, lap);
	cv::imshow("laplacian", lap);

	//
	// geometry transform (affine & persperctive)
	//
	cv::Mat affineMat;
	cv::Mat affineImg;
	std::vector<cv::Point2f> srcTri = { cv::Point2f(0,0),
		cv::Point2f(inputMat.cols - 1, 0),
		cv::Point2f(inputMat.cols - 1, inputMat.rows - 1) };
	std::vector<cv::Point2f> dstTri(srcTri.size(), cv::Point2f(0, 0));

	std::vector<float> pos_x;
	std::vector<float> pos_y;
	pos_x.reserve(srcTri.size());
	pos_y.reserve(srcTri.size());
	std::for_each(srcTri.begin(), srcTri.end(), [&](const cv::Point2f& pt) {
		pos_x.push_back(pt.x);
		pos_y.push_back(pt.y);
	});

	std::vector<float> coord_r(pos_x.size(), 0);
	std::vector<float> coord_a(pos_x.size(), 0);
	cv::cartToPolar(pos_x, pos_y, coord_r, coord_a);
	std::transform(coord_a.begin(), coord_a.end(), coord_a.begin(), [](float x) {
		return x + degreeToRadius(15);
	});
	cv::polarToCart(coord_r, coord_a, pos_x, pos_y);
	
	for (size_t i = 0; i < dstTri.size(); ++i) {
		dstTri[i] = cv::Point2f(pos_x[i], pos_y[i]);
	}
	
	affineMat = cv::getAffineTransform(srcTri, dstTri);
	cv::warpAffine(inputMat, affineImg, affineMat, inputMat.size());
	cv::imshow("affine transform", affineImg);

	cv::Mat perspMat;
	cv::Mat perspImg;
	std::vector<cv::Point2f> srcRect = {cv::Point2f(0,0),
		cv::Point2f(inputMat.cols-1, 0),
		cv::Point2f(inputMat.cols-1, inputMat.rows-1),
		cv::Point2f(0, inputMat.rows-1)};
	std::vector<cv::Point2f> dstRect = {cv::Point2f(inputMat.cols * .15f, inputMat.rows * .25f),
		cv::Point2f(inputMat.cols * .85f, 0),
		cv::Point2f(inputMat.cols * .85f, inputMat.rows-1),
		cv::Point2f(inputMat.cols * .15f, inputMat.rows * .75f)};

	perspMat = cv::getPerspectiveTransform(srcRect, dstRect);
	cv::warpPerspective(inputMat, perspImg, perspMat, inputMat.size());
	cv::imshow("perspective transform", perspImg);

	//
	// denoise
	//
	//cv::Mat denoiseImg;
	//cv::fastNlMeansDenoisingColored(inputMat, denoiseImg);
	//cv::imshow("denoise", denoiseImg);

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



	cv::waitKey();

	cv::destroyAllWindows();

	return 0;
}