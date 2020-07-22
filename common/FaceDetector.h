#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>
#include<common/utility.h>
#include<vector>


class FaceDetector {

private:
	cv::Ptr<cv::face::Facemark> m_facemarkAlgo;
	cv::CascadeClassifier m_faceCascade;

	std::vector<cv::Rect> m_faceROIs;
	std::vector<std::vector<cv::Point2f>> m_faceMarks;

	cv::Mat grayImg;

public:
	FaceDetector(cv::Ptr<cv::face::Facemark> facemarkAlgo = nullptr, const char* cascadePath = nullptr);

	size_t detect(cv::Mat& Img);

	const std::vector<cv::Rect>& getFaceROI() const;
	const std::vector<std::vector<cv::Point2f>>& getFaceMarks() const;
	std::vector<std::vector<cv::Point>> getFaceMarkZone(FacemarkZone zone, int index = -1) const;

};