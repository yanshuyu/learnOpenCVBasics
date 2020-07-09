#pragma once
#include<opencv2/opencv.hpp>

// helper
cv::Scalar randomColor();


// histogram

bool histogramMat(const cv::Mat& iMat, cv::Mat& oMat, int iLineWidth = 1);

cv::Mat equalizeMat(const cv::Mat& iMat);


// image filter

void cartoonFilter(cv::Mat& iMat, cv::Mat& oMat, double edgeThreshold1 = 50, double degeThreshold2 = 150, int edgeWidth = 5);


// object segmentation  (connected components algorithme & find contours algorithme) 

int connectedComponents(cv::Mat& iMat, cv::Mat& oLables);

int connectedComponentsWithStatus(cv::Mat& iMat, cv::Mat& oLables, cv::Mat& oStatus, cv::Mat& oCentroids);