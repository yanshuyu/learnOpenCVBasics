#include<opencv2/opencv.hpp>
#include<common/utility.h>
#include<iostream>


int main(int argc, char* argv[]) {
	cv::VideoCapture cam;
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;
	std::vector<cv::Rect> faceRects;
	std::vector<cv::Rect> eyeRects;

	if (!cam.open(0) || !faceCascade.load("haarcascade_frontalface_alt.xml") || !eyeCascade.load("haarcascade_eye.xml") ) {
		cam.release();
		return -1;
	}

	cv::namedWindow("faceMask");
	cv::namedWindow("glassMask");

	cv::Mat frame;
	cv::Mat frameGray;

	cv::Mat faceMask = cv::imread("mask.jpg");
	cv::Mat faceMaskScaled;
	cv::Mat faceMaskScaledGray;
	cv::Mat faceMaskThresh;
	cv::Mat faceMaskThreshInv;
	cv::Mat faceMaskOverlay;

	cv::Mat glassFrame;
	cv::Mat glassMask = cv::imread("glasses.jpg");
	cv::Mat glassMaskScaled;
	cv::Mat glassMaskScaledGray;
	cv::Mat glassMaskThresh;
	cv::Mat glassMaskThreshInv;
	cv::Mat glassMaskOverlay;

	while (true) {
		cam >> frame;
		frame.copyTo(glassFrame);

		if (!frame.empty()) {
			// preprocessing
			cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(frameGray, frameGray);

			// detect faces
			faceRects.clear();
			faceCascade.detectMultiScale(frameGray, faceRects);

			// overlay face mask
			if (faceRects.size() > 0) {
					cv::Rect face = faceRects[0];
					int x = face.x - int(0.1 * face.width);
					int y = face.y - int(0.0 * face.height);
					int w = int(1.1 * face.width);
					int h = int(1.3 * face.height);

				if (x > 0 && y > 0 && x + w <= frame.cols && y + h <= frame.rows) {
					cv::resize(faceMask, faceMaskScaled, cv::Size(w, h));
					cv::cvtColor(faceMaskScaled, faceMaskScaledGray, cv::COLOR_BGR2GRAY);

					cv::threshold(faceMaskScaledGray, faceMaskThresh, 230, 255, cv::THRESH_BINARY_INV);
					cv::bitwise_not(faceMaskThresh, faceMaskThreshInv);

					cv::Mat  frameROI = frame(cv::Rect(x, y, w, h));
					cv::add(faceMaskScaled, 0, faceMaskOverlay, faceMaskThresh);
					cv::add(frameROI, 0, faceMaskOverlay, faceMaskThreshInv);
					cv::add(faceMaskOverlay, 0, frameROI);
				}
				cv::imshow("faceMask", frame);
			}

			// detect eyes
			if (faceRects.size() > 0) {
				cv::Rect face = faceRects[0];
				cv::Mat frameROI = frameGray(face);
				eyeRects.clear();
				eyeCascade.detectMultiScale(frameROI, eyeRects);
				
				if (eyeRects.size() >= 2) {
					// overlay eye mask
					cv::Point eye_center_1(face.x + eyeRects[0].x + eyeRects[0].width / 2, face.y + eyeRects[0].y + eyeRects[0].height / 2);
					cv::Point eye_center_2(face.x + eyeRects[1].x + eyeRects[1].width / 2, face.y + eyeRects[0].y + eyeRects[1].height / 2);
					cv::Point lEyeCenter = eye_center_1.x < eye_center_2.x ? eye_center_1 : eye_center_2;
					cv::Point rEyrCenter = eye_center_1.x > eye_center_2.x ? eye_center_1 : eye_center_2;

					int w = 2.3 * (rEyrCenter.x - lEyeCenter.x);
					int h = int(0.4 * w);
					int x = lEyeCenter.x - 0.25 * w;
					int y = rEyrCenter.y - 0.5 * h;

					if (x >= 0 && y >= 0 && x + w <= frame.cols && y + h <= frame.rows) {
						cv::Rect rectROI(x, y, w, h);
						cv::resize(glassMask, glassMaskScaled, cv::Size(w, h));
						cv::cvtColor(glassMaskScaled, glassMaskScaledGray, cv::COLOR_BGR2GRAY);
						cv::threshold(glassMaskScaledGray, glassMaskThresh, 230, 255, cv::THRESH_BINARY_INV);
						cv::bitwise_not(glassMaskThresh, glassMaskThreshInv);

						cv::Mat frameROI = glassFrame(rectROI);
						cv::add(glassMaskScaled, 0, glassMaskOverlay, glassMaskThresh);
						cv::add(frameROI, 0, glassMaskOverlay, glassMaskThreshInv);
						cv::add(glassMaskOverlay, 0, frameROI);
					}
					cv::imshow("glassMask", glassFrame);
				}
			}
		}

		if (cv::waitKey(10) >= 0 )
			break;
	}

	cam.release();
	cv::destroyAllWindows();

	return 0;
}