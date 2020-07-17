#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>
#include<sstream>
[]#include<vector>


int main(int argc, char* argv[]) {

	cv::Mat imageMat = cv::imread("segmentObjects.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("orignal.jpg", imageMat);

	cv::threshold(imageMat, imageMat, 250, 255, cv::THRESH_BINARY_INV);
	
	// find contours in imahe
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(imageMat, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size() > 0) {
		cv::Mat contoursMat(imageMat.rows, imageMat.cols, CV_8UC3);
		contoursMat.setTo(cv::Scalar(0, 0, 0));
		cv::drawContours(contoursMat, contours, -1, randomColor(), 1, 8, hierarchy);
		cv::imshow("contours", contoursMat);

		std::vector<std::vector<cv::Point>> approxyCurves;
		for (auto& contour : contours) {
			std::vector<cv::Point> curve;
			cv::approxPolyDP(contour, curve, 10, true);
			if (curve.size() > 0) {
				approxyCurves.push_back(std::move(curve));
			}
		}

		if (approxyCurves.size() > 0) {
			cv::Mat curveMat(cv::Size(imageMat.rows, imageMat.cols), CV_8UC3, cv::Scalar(0, 0, 0));
			cv::drawContours(curveMat, approxyCurves, -1, randomColor());
			cv::imshow("approxy contours", curveMat);
		}
	}

	// find connect parts in image
	cv::Mat segments;
	cv::Mat status;
	cv::Mat centroids;
	int objCount = cv::connectedComponentsWithStats(imageMat, segments, status, centroids);

	if (objCount > 1) {
		cv::Mat visualizeSegments = cv::Mat::zeros(segments.rows, segments.cols, CV_8UC3);
		for (size_t i = 1; i < objCount; i++) {
			int area = status.at<int>(i, cv::CC_STAT_AREA);
			if (area < 100)
				continue;

			cv::Mat mask = segments == i;
			auto c = randomColorNormalized() * 255;
			visualizeSegments.setTo(randomColor(), mask);

			std::stringstream ss;
			ss << "area " << i << ": " << area;
			cv::putText(visualizeSegments, ss.str(), centroids.at<cv::Point2d>(i, 0), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255));
		}
		char windowName[32] = { 0 };
		sprintf_s(windowName, "segements count: %d", objCount);
		cv::imshow(windowName, visualizeSegments);
	}


	cv::waitKey();

	return 0;
}