#include"utility.h"


// helper
cv::Scalar randomColorNormalized() {
	cv::RNG& rng = cv::theRNG();
	return cv::Scalar(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
}

cv::Scalar  randomColor() {
	cv::RNG& rng = cv::theRNG();
	return cv::Scalar(rng.uniform(0.0, 256.0), rng.uniform(0.0, 256.0), rng.uniform(0.0, 256.0));
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

cv::Mat edgeMorphologyTransform(cv::Mat& iMat, MorphologyTransformType type, int morphElement, int morphSize) {
	cv::Mat output;
	
	// get kernel
	cv::Mat k = cv::getStructuringElement(morphElement, cv::Size(morphSize * 2 + 1, morphSize * 2 + 1), cv::Point(morphSize, morphSize));
	cv::Mat ex;
	cv::Mat sh;

	switch (type) 
	{
	case expand:
		cv::dilate(iMat, output, k);
		break;

	case shrink:
		cv::erode(iMat, output, k);
		break;

	case stroke:
		// expend - shrink
		cv::dilate(iMat, ex, k);
		cv::erode(iMat, sh, k);
		output = ex - sh;

	case opening:
		cv::erode(iMat, sh, k);
		cv::dilate(sh, output, k);
		break;

	case closing:
		cv::dilate(iMat, ex, k);
		cv::erode(ex, output, k);
		break;

	case topHat:
		// input - opening 
		cv::erode(iMat, sh, k);
		cv::dilate(sh, ex, k);
		output = iMat - ex;
		break;

	case blackHat:
		// closing - input
		cv::dilate(iMat, ex, k);
		cv::erode(ex, sh, k);
		output = sh - iMat;
		break;

	default:
		break;
	}

	return output;
}

cv::Mat smoothFilter(cv::Mat& iMat, SmoothType type, int kernelSize, int borderType, double sigmal1, double sigmal2) {
	cv::Mat output;
	switch (type)
	{
	case mean:
		cv::blur(iMat, output, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), borderType);
		break;

	case box:
		cv::boxFilter(iMat, output, CV_8U, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), true, borderType);
		break;

	case guassian:
		cv::GaussianBlur(iMat, output, cv::Size(kernelSize, kernelSize), sigmal1, sigmal2, borderType);
		break;

	case median:
		cv::medianBlur(iMat, output, kernelSize);
		break;

	case bilateral:
		cv::bilateralFilter(iMat, output, kernelSize, sigmal1, sigmal2, borderType);
		break;
	default:
		break;
	}
	return output;
}

void cartoonFilter(cv::Mat& iMat, cv::Mat& oMat, double edgeThreshold1, double degeThreshold2, int edgeWidth) {
	// apply median filter to remove possible noise
	cv::Mat cannyMatReady = smoothFilter(iMat, SmoothType::bilateral, 5);
	//cv::imshow("noise reduction", resultMat);

	// apply canny filter to look up edges
	cv::Mat cannyMat;
	cv::Canny(cannyMatReady, cannyMat, edgeThreshold1, degeThreshold2 );

	// apply dilate filter to expand and link edge
	cannyMat = edgeMorphologyTransform(cannyMat, MorphologyTransformType::closing);
	cannyMat = edgeMorphologyTransform(cannyMat, MorphologyTransformType::expand, cv::MORPH_RECT, 3);

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
	cannyMatFlt3C = smoothFilter(cannyMatFlt3C, SmoothType::mean, 7);

	//cv::imshow("edges", cannyMatFlt3C);

	// apply  bilateral filter reduce noise and keep edge
	// truncate color to create stonger cartoon efflect
	cv::Mat blendReadyMat = smoothFilter(iMat, SmoothType::bilateral, 5, 4, 100, 100);
	cv::Mat resultMat = blendReadyMat / 25;
	resultMat = resultMat * 25;

	cv::Mat resultMatFlt;
	resultMat.convertTo(resultMatFlt, CV_32FC3);

	// blend edge and source image
	cv::multiply(resultMatFlt, cannyMatFlt3C, resultMatFlt);

	resultMatFlt.convertTo(oMat, CV_8UC3);
}


cv::Mat  frameDifference(const cv::Mat& prevFrame, const cv::Mat& curFrame, const cv::Mat& nextFrame, cv::Mat* df) {
	static cv::Mat df_1;
	static cv::Mat df_2;
	cv::Mat output;
	df = df ? df : &output;

	cv::absdiff(curFrame, prevFrame, df_1);
	cv::absdiff(nextFrame, curFrame, df_2);
	cv::bitwise_and(df_1, df_2, *df);

	return *df;
}