#include<opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char* argv[]) {
	Mat orignal = imread("flower.jpg", IMREAD_COLOR);
	imshow("flowers.jpg", orignal);

	// split color channels
	Mat bgrChannels[3];
	split(orignal, bgrChannels);

	// access mat element
	for (size_t row = 0; row < orignal.cols; row++)
	{
		for (size_t col = 0; col < orignal.rows; col++)
		{
			bgrChannels[1].at<uint8_t>(Point(row, col)) = 0;
		}
	}

	// merge color channels
	Mat  filteredGreen;
	merge(bgrChannels, 3, filteredGreen);
	namedWindow("flowers.jpg(b0r)", WINDOW_KEEPRATIO);
	imshow("flowers.jpg(b0r)", filteredGreen);
	
	waitKey();
}