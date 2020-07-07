#pragma once
#include<opencv2/opencv.hpp>

bool histogramMat(const cv::Mat& iMat, cv::Mat& oMat, int iLineWidth = 1);

cv::Mat equalizeMat(const cv::Mat& iMat);


// image filter

void cartoonFilter(cv::Mat& iMat, cv::Mat& oMat, double edgeThreshold1 = 50, double degeThreshold2 = 150, int edgeWidth = 5);