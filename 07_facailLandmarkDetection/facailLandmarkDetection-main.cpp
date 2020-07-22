#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>
#include<common/utility.h>
#include<common/FaceDetector.h>
#include<iostream>
#include<vector>

using std::cout;
using std::endl;



void drawFaceROI(cv::Mat& frame, std::vector<cv::Rect>& faces);
void drawFacemarkZone(cv::Mat& frame,const std::vector<cv::Point>& zone, cv::Scalar color);

int main(int argc, char* argv[]) {
	cv::VideoCapture cap(0);
	cv::Mat frame;
	cv::Mat faceMask;
	FaceDetector faceDetector;

	if (!cap.isOpened()) {
		cout << " failed to open webCam!" << endl;
		return -1;
	}

	const char* wndTitle = "facial landmark detection";
	cv::namedWindow(wndTitle);

	std::vector<cv::Scalar> faceZoneColors;
	faceZoneColors.reserve((size_t)FacemarkZone::end);
	while (faceZoneColors.size() < (size_t)FacemarkZone::end) {
		faceZoneColors.push_back(randomColor());
	}

	while (true) {
		cap >> frame;

		if (frame.empty()) 
			continue;
	
		if (faceDetector.detect(frame) > 0) {
			for (size_t zone = 0; zone < (size_t)FacemarkZone::end; zone++) {
				auto faceZones = faceDetector.getFaceMarkZone((FacemarkZone)zone);
				for (auto& zonePoints : faceZones) {
					cv::Point* pointsData = zonePoints.data();
					int numPoints = zonePoints.size();
					cv::polylines(frame, &pointsData, &numPoints, 1, false, faceZoneColors[(size_t)zone], 2);
				}
			}
		}

		cv::imshow(wndTitle, frame);

		if (cv::waitKey(5) > 0)
			break;
	}

	cv::destroyAllWindows();

	return 0;
}



void drawFaceROI(cv::Mat& frame, std::vector<cv::Rect>& faces) {
	for (const auto& rect : faces) {
		cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
	}
}


void drawFacemarkZoneMask(cv::Mat& outImage, FacemarkZone zone, const std::vector<cv::Point2f>& landMarks) {
	if (landMarks.size() != 68)
		return;

	size_t begIdx = 0;
	size_t endIdx = 0;
	if (getFacemarkZoneIndexRange(FacemarkZone(zone), begIdx, endIdx) && endIdx > begIdx) {
		std::vector<cv::Point> zonePoints;
		std::vector<cv::Point> hullPoints;

		zonePoints.reserve(endIdx - begIdx);
		for (size_t i = begIdx; i < endIdx; i++) {
			zonePoints.push_back(landMarks[i]);
		}

		cv::convexHull(zonePoints, hullPoints);
		cv::fillConvexPoly(outImage, hullPoints, 255);
	}
}