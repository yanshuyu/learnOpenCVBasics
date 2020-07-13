#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>
#include<iostream>
#include<vector>

using std::cout;
using std::endl;



void detectFaceROI(cv::CascadeClassifier& faceCascade, cv::Mat& frame, std::vector<cv::Rect>& detectedFaces); 
bool detectFaceLandmark(cv::Ptr<cv::face::Facemark> pFacemark, cv::Mat& frame, std::vector<cv::Rect>& faceROI,std::vector<std::vector<cv::Point2f>>& detectedFaceLandmarks);
void drawFaceROI(cv::Mat& frame, std::vector<cv::Rect>& faces);
void drawFaceLandmark(cv::Mat& frame, cv::Ptr<cv::face::Facemark> pFacemark, std::vector<std::vector<cv::Point2f>>& faceLandmarks);



int main(int argc, char* argv[]) {
	cv::VideoCapture cap(0);
	cv::Mat frame;

	if (!cap.isOpened()) {
		cout << " failed to open webCam!" << endl;
		return -1;
	}


	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load("haarcascade_frontalface_alt.xml")) {
		cout << "failed to load face cascade modle!" << endl;
		return -1;
	}

	cv::Ptr<cv::face::Facemark> pFaceMarkLBF = cv::face::FacemarkLBF::create();
	pFaceMarkLBF->loadModel("lbfmodel.yaml");

	const char* wndTitle = "facial landmark detection";
	cv::namedWindow(wndTitle);

	std::vector<cv::Rect> faceROIVec;
	std::vector<std::vector<cv::Point2f>> faceLandMarkVec;

	while (true) {
		cap >> frame;
		
		if (frame.empty()) 
			continue;

		detectFaceROI(faceCascade, frame, faceROIVec);

		if (faceROIVec.size() > 0) {
			if (detectFaceLandmark(pFaceMarkLBF, frame, faceROIVec, faceLandMarkVec)) {
				drawFaceLandmark(frame, pFaceMarkLBF, faceLandMarkVec);
			}
		}

		drawFaceROI(frame, faceROIVec);

		cv::imshow(wndTitle, frame);

		if (cv::waitKey(1) > 0)
			break;
	}

	cv::destroyAllWindows();

	return 0;
}




void detectFaceROI(cv::CascadeClassifier& faceCascade, cv::Mat& frame, std::vector<cv::Rect>& detectedFaces) {
	cv::Mat gray;
	if (frame.channels() > 1) {
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	} else {
		frame.copyTo(gray);
	}

	cv::equalizeHist(gray, gray);

	detectedFaces.clear();

	faceCascade.detectMultiScale(gray, detectedFaces, 1.2);
}


void drawFaceROI(cv::Mat& frame, std::vector<cv::Rect>& faces) {
	for (const auto& rect : faces) {
		cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
	}
}


bool detectFaceLandmark(cv::Ptr<cv::face::Facemark> pFacemark, cv::Mat& frame, std::vector<cv::Rect>& faceROI, std::vector<std::vector<cv::Point2f>>& detectedFaceLandmarks) {
	if (!pFacemark)
		return false;
	if (faceROI.size() <= 0)
		return false;

	detectedFaceLandmarks.clear();
	 return pFacemark->fit(frame, faceROI, detectedFaceLandmarks);
}


void drawFaceLandmark(cv::Mat& frame, cv::Ptr<cv::face::Facemark> pFacemark, std::vector<std::vector<cv::Point2f>>& faceLandmarks) {
	if (!pFacemark)
		return;
	for (auto& landmark : faceLandmarks) {
		cv::face::drawFacemarks(frame, landmark, cv::Scalar(255, 0, 0));
	}
}