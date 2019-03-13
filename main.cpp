#include <iostream>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "barrel_distortion.hpp"

int main(int argc, char **argv)
{
	using namespace std;
	using namespace cv;

	Mat input, output;
	float K;
	float centerX, centerY;
	int width, height;

	if(argc == 3)
	{
		input = imread(argv[1], IMREAD_COLOR);
		output = Mat(input.rows, input.cols, input.type());
		K = atof(argv[2]);
		width = input.cols;
		height = input.rows;
		centerX = width / 2;
		centerY = height / 2;
	}
	else if(argc == 5)
	{
		input = imread(argv[1], IMREAD_COLOR);
		output = Mat(input.rows, input.cols, input.type());
		K = atof(argv[2]);
		width = input.cols;
		height = input.rows;
		centerX = atof(argv[3]);
		centerY = atof(argv[4]);	
	}
	else
	{
		cout << "usage: ./barrel_distort_cpp \
			<input image> \
			<K: coefficient of distortion> \
			[x corrdinate of center of distortion (in pixel)] \
			[y corrdinate of center of distortion (in pixel)]"
			<< endl;
		return 1;
	}

	// barrel distort here
	barrelDistortion barrel = barrelDistortion(input, output, 
		K, centerX, centerY, width, height);
	barrel.barrel_distort();

	// show output
	imshow("distorted", output);
	waitKey();

	return 0;
}
