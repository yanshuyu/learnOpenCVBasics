#include"utility.h"


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


