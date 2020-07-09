#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>
#include<sstream>


int main(int argc, char* argv[]) {

	cv::Mat imageMat = cv::imread("segmentObjects.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("orignal.jpg", imageMat);

	cv::Mat segments;
	cv::Mat status;
	cv::Mat centroids;
	int objCount = connectedComponentsWithStatus(imageMat, segments, status, centroids);

	if (objCount > 1) {
		cv::Mat visualizeSegments = cv::Mat::zeros(segments.rows, segments.cols, CV_8UC3);
		for (size_t i = 1; i < objCount; i++) {
			cv::Mat mask = segments == i;
			visualizeSegments.setTo(randomColor(), mask);

			std::stringstream ss;
			ss << "area " << i << ": " << status.at<int>(i, cv::CC_STAT_AREA);
			cv::putText(visualizeSegments, ss.str(), centroids.at<cv::Point2d>(i,0), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255));
		}
		char windowName[32] = { 0 };
		sprintf_s(windowName, "segements count: %d", objCount);
		cv::imshow(windowName, visualizeSegments);
	}


	cv::waitKey();

	return 0;
}