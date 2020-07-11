#include<opencv2/opencv.hpp>
#include<common/utility.h>



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


	while (true)
	{
		//
		// 1 background subtract with frame diffrences
		cv::imshow("frameGray", frame);
		cv::imshow("frameDiffrence", frameDifference(prevGray, curGray, nextGray, &frameDF));


		//
		// 2 background subtract with MOG / MOG2
		pMOG2->apply(frame, fgMOG2Mask);
		cv::imshow("fgMOG2Mask", fgMOG2Mask);


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