#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>

using std::cout;
using std::endl;


cv::Mat getFrame(cv::VideoCapture& cap, cv::Mat* iframeStore = nullptr, cv::Mat* oframeStore = nullptr);

int main(int agrc, char* agrv[]) {
	cv::VideoCapture cap;

	if (!cap.open(0)) {
		cap.release();
		return -1;
	}
	
	cv::namedWindow("frameGray");
	cv::namedWindow("frameDiffrence");
	cv::namedWindow("fgMOG2Mask");

	cv::Mat frame;
	cv::Mat curGray;
	cv::Mat prevGray;
	cv::Mat nextGray;
	cv::Mat frameDF;

	cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2();
	cv::Mat fgMOG2Mask;

	
	prevGray = getFrame(cap);
	curGray = getFrame(cap, &frame);
	nextGray = getFrame(cap);


	cv::Mat cornerMask;
	cv::Mat cornerMaskNorm;
	cv::namedWindow("harris corners");
	std::vector<cv::Point2f> cornerPoints;

	while (true)
	{
		//
		// 1 background subtract with frame diffrences
		cv::imshow("frameDiffrence", frameDifference(prevGray, curGray, nextGray, &frameDF));


		//
		// 2 background subtract with MOG / MOG2
		pMOG2->apply(frame, fgMOG2Mask);
		cv::imshow("fgMOG2Mask", fgMOG2Mask);

		// 3 detect corners
		/* 
		cv::cornerHarris(curGray, cornerMask, 4, 0.08, cv::BORDER_DEFAULT);
		cv::normalize(cornerMask, cornerMask, 0, 255, cv::NORM_MINMAX, CV_32FC1);
		cv::convertScaleAbs(cornerMask, cornerMaskNorm);
		cv::threshold(cornerMaskNorm, cornerMaskNorm, 200, 255, cv::THRESH_BINARY_INV);
		cv::imshow("harris corners", cornerMaskNorm);
		*/

		// 4 feature base track
		cornerPoints.clear();
		cv::goodFeaturesToTrack(curGray, cornerPoints, 100, 0.04, 15, cv::noArray(), 5, false);
		if (cornerPoints.size() > 0) {
			for (auto& corner : cornerPoints) {
				cv::circle(frame, corner, 8, randomColor(), 3, cv::LineTypes::FILLED);
			}
		}

		cv::imshow("frameGray", frame);

		//cout << cv::max(cornerMask, 0) << endl;

		curGray.copyTo(prevGray);
		nextGray.copyTo(curGray);
		getFrame(cap, &frame, &nextGray);


		if (cv::waitKey(10) >= 0)
			break;
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}


cv::Mat getFrame(cv::VideoCapture& cap, cv::Mat* iframeStore, cv::Mat* oframeStore) {
	cv::Mat frame;
	cv::Mat gray;
	iframeStore = iframeStore ? iframeStore : &frame;
	oframeStore = oframeStore ? oframeStore : &gray;

	cap >> *iframeStore;
	cv::cvtColor(*iframeStore, *oframeStore, cv::COLOR_BGR2GRAY);
	return *oframeStore;
}