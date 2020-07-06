#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;

int main(int argc, char* argv[]) {
	Mat orignal = imread("flowers.jpg", IMREAD_COLOR);
	imshow("flowers.jpg", orignal);
	std::cout << "orignal rows: " << orignal.rows << ",\t" << "cols: " << orignal.cols << std::endl;

	// split color channels
	Mat bgrChannels[3];
	split(orignal, bgrChannels);

	// access mat element
	for (size_t row = 0; row < orignal.rows; row++)
	{
		for (size_t col = 0; col < orignal.cols; col++)
		{
			bgrChannels[1].at<uint8_t>(Point(col, row)) = 0;
		}
	}

	// merge color channels
	Mat  filteredGreen;
	merge(bgrChannels, 3, filteredGreen);
	namedWindow("flowers.jpg(b0r)", WINDOW_FREERATIO);
	imshow("flowers.jpg(b0r)", filteredGreen);

	imwrite("flowersb0r.jpg", filteredGreen);

	waitKey();

	destroyWindow("flowers.jpg(b0r)");

	return 0;
}