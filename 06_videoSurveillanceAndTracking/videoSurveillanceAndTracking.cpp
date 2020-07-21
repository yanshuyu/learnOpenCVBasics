#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>

using std::cout;
using std::endl;


cv::Mat getFrame(cv::VideoCapture& cap, cv::Mat* iframeStore = nullptr, cv::Mat* oframeStore = nullptr);
void glRenderCallback(void*);

int main(int agrc, char* agrv[]) {
	cv::VideoCapture cap;

	if (!cap.open(0)) {
		cap.release();
		return -1;
	}
	
	cv::namedWindow("capturing");
	cv::namedWindow("frameDiffrence");
	cv::namedWindow("fgMOG2Mask");
	cv::namedWindow("keyPoints");

	cv::Mat frame;
	cv::Mat curGray;
	cv::Mat prevGray;
	cv::Mat nextGray;
	cv::Mat frameDF;

	cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2();
	cv::Mat fgMOG2Mask;


	cv::Ptr<cv::Feature2D> orbFeature2dImp = cv::ORB::create();
	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat kpDescriptors;
	
	prevGray = getFrame(cap);
	curGray = getFrame(cap, &frame);
	nextGray = getFrame(cap);


	cv::Mat cornerMask;
	cv::Mat cornerMaskNorm;
	//cv::namedWindow("harris corners");
	//std::vector<cv::Point2f> keyPoints;


	while (true)
	{
		cv::imshow("capturing", frame);

		//
		// 1 detection motion with frame diffrences
		cv::imshow("frameDiffrence", frameDifference(prevGray, curGray, nextGray, &frameDF));


		//
		// 2 background subtract with MOG / MOG2
		pMOG2->apply(frame, fgMOG2Mask);
		cv::imshow("fgMOG2Mask", fgMOG2Mask);

		// 3 find feature(key) point using harris algorithme
		/* 
		cv::cornerHarris(curGray, cornerMask, 4, 0.08, cv::BORDER_DEFAULT);
		cv::normalize(cornerMask, cornerMask, 0, 255, cv::NORM_MINMAX, CV_32FC1);
		cv::convertScaleAbs(cornerMask, cornerMaskNorm);
		cv::threshold(cornerMaskNorm, cornerMaskNorm, 200, 255, cv::THRESH_BINARY_INV);
		cv::imshow("harris corners", cornerMaskNorm);
		*/

		// 4 finding feature(key) points
		/*
		keyPoints.clear();
		cv::goodFeaturesToTrack(curGray, keyPoints, 100, 0.05, 15, cv::noArray(), 5, false);
	
		if (keyPoints.size() > 0) {
			for (auto& kp : keyPoints) {
				cv::circle(frame, kp, 3, randomColor());
			}
			cv::imshow("keyPoints", frame);
		}
		*/

		// finding keypoints , computing keypoint description, visulize keypoints
		keyPoints.clear();
		orbFeature2dImp->detectAndCompute(frame, cv::noArray(), keyPoints, kpDescriptors);
		cv::drawKeypoints(frame, keyPoints, frame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow("keyPoints", frame);
	

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


void glRenderCallback(void*) {

}