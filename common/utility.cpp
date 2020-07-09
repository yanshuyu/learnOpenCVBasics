#include"utility.h"

static bool initRandSeed = false;

// helper
cv::Scalar randomColor() {
	if (!initRandSeed) {
		srand(time(0));
		initRandSeed = true;
	}
	return cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}


// histogram
bool histogramMat(const cv::Mat& iMat, cv::Mat& oMat, int iLineWidth) {
	// sperate source image b g r color channel
	std::vector<cv::Mat> bgrMats;
	cv::split(iMat, bgrMats);

	if (bgrMats.size() < 3) {
		return false;
	}

	cv::Mat bHistMat;
	cv::Mat gHistMat;
	cv::Mat rHistMat;
	const int histBinSize = 256;
	const float  histRange[] = { 0, 255 };
	const float* histRangePtr = histRange;
	const int matChannel = 0;

	cv::calcHist(&bgrMats[0],  // input  image array
		1,  // number of images in input image array
		0, //  channel number used to calc histogram 
		cv::Mat(), // mask mat
		bHistMat, // the cacluated histogram mat
		1, // dimension of input image mat
		&histBinSize, // histogram bin size
		&histRangePtr // min max value in input image mat
	);
	cv::calcHist(&bgrMats[1],  // input  image array
		1,  // number of images in input image array
		0, //  channel number used to calc histogram 
		cv::Mat(), // mask mat
		gHistMat, // the cacluated histogram mat
		1, // dimension of input image mat
		&histBinSize, // histogram bin size
		&histRangePtr // min max value in input image mat
	);
	cv::calcHist(&bgrMats[2],  // input  image array
		1,  // number of images in input image array
		0, //  channel number used to calc histogram 
		cv::Mat(), // mask mat
		rHistMat, // the cacluated histogram mat
		1, // dimension of input image mat
		&histBinSize, // histogram bin size
		&histRangePtr // min max value in input image mat
	);

	int width = oMat.size().width;
	int height = oMat.size().height;

	// normalize b g r histogram for visualiztion
	cv::normalize(bHistMat, bHistMat, 0, height, cv::NORM_MINMAX);
	cv::normalize(gHistMat, gHistMat, 0, height, cv::NORM_MINMAX);
	cv::normalize(rHistMat, rHistMat, 0, height, cv::NORM_MINMAX);

	// draw visualize histogram
	size_t binStep = cvRound(float(width) / histBinSize);
	for (size_t i = 1; i < histBinSize; i++) {
		cv::line(oMat,
			cv::Point((i - 1) * binStep, height - cvRound(bHistMat.at<float>(cv::Point(0, i - 1)))),
			cv::Point(i * binStep, height - cvRound(bHistMat.at<float>(cv::Point(0, i)))),
			cv::Scalar(0, 0, 255),
			iLineWidth);

		cv::line(oMat,
			cv::Point((i - 1) * binStep, height - cvRound(gHistMat.at<float>(cv::Point(0, i - 1)))),
			cv::Point(i * binStep, height - cvRound(gHistMat.at<float>(cv::Point(0, i)))),
			cv::Scalar(0, 255, 0),
			iLineWidth);

		cv::line(oMat,
			cv::Point((i - 1) * binStep, height - cvRound(rHistMat.at<float>(cv::Point(0, i - 1)))),
			cv::Point(i * binStep, height - cvRound(rHistMat.at<float>(cv::Point(0, i)))),
			cv::Scalar(255, 0, 0),
			iLineWidth);
	}

	return true;
}

cv::Mat equalizeMat(const cv::Mat& iMat) {
	// convert bgr mat to ycrcb mat
	cv::Mat yCrCbMat;
	cv::cvtColor(iMat, yCrCbMat, cv::COLOR_BGR2YCrCb);

	//split color channels to access luminancechannel
	std::vector<cv::Mat> colorChannels;
	cv::split(yCrCbMat, colorChannels);

	// equalized luminance
	cv::equalizeHist(colorChannels[0], colorChannels[0]);

	cv::merge(colorChannels, yCrCbMat);

	cv::cvtColor(yCrCbMat, yCrCbMat, cv::COLOR_YCrCb2BGR);

	return yCrCbMat;
}



// filter
void cartoonFilter(cv::Mat& iMat, cv::Mat& oMat, double edgeThreshold1, double degeThreshold2, int edgeWidth) {
	// apply median filter to remove possible noise
	cv::Mat cannyMatReady;
	cv::medianBlur(iMat, cannyMatReady, 5);
	//cv::imshow("noise reduction", resultMat);

	// apply canny filter to look up edges
	cv::Mat cannyMat;
	cv::Canny(cannyMatReady, cannyMat, edgeThreshold1, degeThreshold2 );

	// apply dilate filter to expand and link edge
	cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(edgeWidth, edgeWidth));
	cv::dilate(cannyMat, cannyMat, k);

	// scale edge to [0,1], invert edge color
	cannyMat = 255 - cannyMat;
	cannyMat = cannyMat / 255;

	// convert to float component for blending
	cv::Mat cannyMatFlt;
	cannyMat.convertTo(cannyMatFlt, CV_32F);

	cv::Mat cannyMatFlt3C;
	cv::Mat cannyMatChannels[] = { cannyMatFlt, cannyMatFlt, cannyMatFlt };
	cv::merge(cannyMatChannels, 3, cannyMatFlt3C);

	// smooth edge
	cv::blur(cannyMatFlt3C, cannyMatFlt3C, cv::Size(5, 5));

	//cv::imshow("edges", cannyMatFlt3C);

	// apply  bilateral filter reduce noise and keep edge
	// truncate color to create stonger cartoon efflect
	cv::Mat blendReadyMat;
	cv::bilateralFilter(iMat, blendReadyMat, 5, 150, 150);
	cv::Mat resultMat = blendReadyMat / 25;
	resultMat = resultMat * 25;

	cv::Mat resultMatFlt;
	resultMat.convertTo(resultMatFlt, CV_32FC3);

	// blend edge and source image
	cv::multiply(resultMatFlt, cannyMatFlt3C, resultMatFlt);

	resultMatFlt.convertTo(oMat, CV_8UC3);
}



// object segmentation (connected components algorithme & find contours algorithme)
int connectedComponents(cv::Mat& iMat, cv::Mat& oLables) {
	//
	// 1. preproessing
	//
	// denoise 
	cv::Mat denoiseMat;
	cv::medianBlur(iMat, denoiseMat, 5);

	// remove light / background
	int size = MIN(denoiseMat.cols, denoiseMat.rows);
	cv::Mat estimateLightMat;
	cv::blur(denoiseMat, estimateLightMat, cv::Size(size / 3, size / 3));

	cv::Mat noLightMat = estimateLightMat - denoiseMat;

	// binarized 
	cv::Mat noLightBinarized;
	cv::threshold(noLightMat, noLightBinarized, 30, 255, cv::THRESH_BINARY);

	//
	// 2.segmenting
	//
	int segmentCount = cv::connectedComponents(noLightBinarized, oLables);
	
	return segmentCount;
}

int connectedComponentsWithStatus(cv::Mat& iMat, cv::Mat& oLables, cv::Mat& oStatus, cv::Mat& oCentroids) {
	//
	// 1. preproessing
	//
	// denoise 
	cv::Mat denoiseMat;
	cv::medianBlur(iMat, denoiseMat, 5);

	// remove light / background
	int size = MIN(denoiseMat.cols, denoiseMat.rows);
	cv::Mat estimateLightMat;
	cv::blur(denoiseMat, estimateLightMat, cv::Size(size / 3, size / 3));

	cv::Mat noLightMat = estimateLightMat - denoiseMat;

	// binarized 
	cv::Mat noLightBinarized;
	cv::threshold(noLightMat, noLightBinarized, 30, 255, cv::THRESH_BINARY);

	//
	// 2.segmenting
	//
	int segmentCount = cv::connectedComponentsWithStats(noLightBinarized, oLables, oStatus, oCentroids);

	return segmentCount;
}