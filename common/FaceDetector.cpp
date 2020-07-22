#include"FaceDetector.h"


FaceDetector::FaceDetector(cv::Ptr<cv::face::Facemark> facemarkAlgo, const char* casecadePath) {
	const char* casecadeDataSet = casecadePath == nullptr ? "haarcascade_frontalface_alt.xml" : casecadePath;
	bool success = this->m_faceCascade.load(casecadeDataSet);
#ifdef _DEBUG
	assert(success);
#endif // 
	this->m_facemarkAlgo = facemarkAlgo;

	if (this->m_facemarkAlgo == nullptr) {
		this->m_facemarkAlgo = cv::face::createFacemarkLBF();
		this->m_facemarkAlgo->loadModel("lbfmodel.yaml");
	}
}



size_t FaceDetector::detect(cv::Mat& img) {
	if (img.empty())
		return 0;

	if (img.channels() == 1) {
		this->grayImg = img;
	} else {
		if (img.channels() == 3) {
			cv::cvtColor(img, this->grayImg, cv::COLOR_BGR2GRAY);
		} else if (img.channels() == 4) {
			cv::cvtColor(img, this->grayImg, cv::COLOR_BGRA2GRAY);
		} else
			return 0;
	}

	// detecte faces roi
	this->m_faceROIs.clear();
	this->m_faceMarks.clear();

	this->m_faceCascade.detectMultiScale(this->grayImg, this->m_faceROIs);
	if (this->m_faceROIs.size() <= 0)
		return 0;

	// detecte face marks
	bool success = this->m_facemarkAlgo->fit(this->grayImg, this->m_faceROIs, this->m_faceMarks);
	if (not success)
		return 0;

	return this->m_faceROIs.size();
}



const std::vector<cv::Rect>& FaceDetector::getFaceROI() const {
	return this->m_faceROIs;
}


const std::vector<std::vector<cv::Point2f>>& FaceDetector::getFaceMarks() const {
	return this->m_faceMarks;
}


std::vector<std::vector<cv::Point>> FaceDetector::getFaceMarkZone(FacemarkZone zone, int index) const {
	std::vector<std::vector<cv::Point>> zonePoints;
	std::vector<size_t> queryIndices;
	size_t begIdx = 0;
	size_t endIdx = 0;

	if (this->m_faceMarks.size() > 0 && getFacemarkZoneIndexRange(zone, begIdx, endIdx) && endIdx > begIdx) {
		if (index >= 0) {
			queryIndices.push_back(index);
		} else {
			queryIndices.reserve(this->m_faceMarks.size());
			for (size_t i = 0; i < this->m_faceMarks.size(); i++) {
				queryIndices.push_back(i);
			}
		}

		zonePoints.resize(queryIndices.size());
		for (auto& points : zonePoints) {
			points.reserve(endIdx - begIdx);
		}

		size_t i = 0;
		for (size_t idx : queryIndices) {
			size_t curIdx = begIdx;
			while (curIdx < endIdx) {
				zonePoints[i].push_back(this->m_faceMarks[idx][curIdx]);
				curIdx++;
			}
			i++;
		}
	}

	return std::move(zonePoints);
}