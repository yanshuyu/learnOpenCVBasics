#pragma once
#include<opencv2/opencv.hpp>

// helper
cv::Scalar randomColorNormalized();
cv::Scalar randomColor();

float radiusToDegree(float r);
float degreeToRadius(float d);

enum class FacemarkZone
{
	jaw,
	rightBrow,
	leftBrow,
	noseUpper,
	noseLower,
	rightEye,
	leftEye,
	mouseOuter,
	mouseInner,
	end,
};

bool getFacemarkZoneIndexRange(FacemarkZone zone, size_t& begin, size_t& end);


// 
// image processing
//
enum MorphologyTransformType {
	expand,
	shrink,
	opening,
	closing,
	stroke,
	topHat,
	blackHat,
};

// morphologycal operation
cv::Mat edgeMorphologyTransform(cv::Mat& iMat, MorphologyTransformType type, int morphElement = cv::MORPH_RECT, int morphSize = 2);


// smooth filter
enum SmoothType {
	mean,
	box,
	guassian,
	median,
	bilateral,
};

cv::Mat smoothFilter(cv::Mat& iMat, SmoothType type, int kernelSize, int borderType = 4, double sigmal1 = 1.0, double sigmal2 = 1.0);


// histogram
bool histogramMat(const cv::Mat& iMat, cv::Mat& oMat, int iLineWidth = 1);

cv::Mat equalizeMat(const cv::Mat& iMat);

// filter
void cartoonFilter(cv::Mat& iMat, cv::Mat& oMat, double edgeThreshold1 = 50, double degeThreshold2 = 150, int edgeWidth = 5);


//
// video processing
//

// object motion
cv::Mat frameDifference(const cv::Mat& prevFrame, const cv::Mat& curFrame, const cv::Mat& nextFrame, cv::Mat* df = nullptr);