#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "barrel_distortion.cuh"

using std::cout;
using std::endl;
using cv::Mat;
using cv::Scalar;

barrelDistortion::barrelDistortion(Mat& src, Mat& dst,
	float K,
	float centerX, float centerY,
	int width, int height)
{
	this->src = src;
	this->dst = dst;
	this->K = K;
	this->centerX = centerX;
	this->centerY = centerY;
	this->width = width;
	this->height = height;
}

void barrelDistortion::barrel_distort()
{
	cout << "channels: " << src.channels() 
		<< " type: " << src.type() << endl;

	xshift = calc_shift(0, centerX-1, centerX, K);
	float newcenterX = width - centerX;
	float xshift_2 = calc_shift(0, newcenterX-1, newcenterX, K);

	yshift = calc_shift(0, centerY-1, centerY, K);
	float newcenterY = height - centerY;
	float yshift_2 = calc_shift(0, newcenterY-1, newcenterY, K);

	xscale = (width - xshift - xshift_2) / width;
	yscale = (height - yshift - yshift_2) / height;

	cout << xshift << " " << yshift << " " << xscale << " " << yscale << endl;
	cout << height << endl;
	cout << width << endl;

	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			//Scalar temp;
			cv::Vec3b temp;
			float x = getRadialX((float)i, (float)j, centerX, centerY, K);
			float y = getRadialY((float)i, (float)j, centerX, centerY, K);
			//sampleImage(src, y, x, temp);
			sampleImageTest(src, y, x, temp);
			//dst.at<Scalar>(j, i) = temp;
			dst.at<cv::Vec3b>(j, i) = temp;
		}
	}
}

float barrelDistortion::calc_shift(float x1, float x2, float cx, float k)
{
	float x3 = x1 + (x2 - x1) * 0.5;
	float result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
	float result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

	if(result1 > -thresh and result1 < thresh)
		return x1;
	if(result3 < 0)
	{
		return calc_shift(x3, x2, cx, k);
	}
	else
	{
		return calc_shift(x1, x3, cx, k);
	}
}

float barrelDistortion::getRadialX(float x, float y, float cx, float cy, float k)
{
	x = (x * xscale + xshift);
	y = (y * yscale + yshift);
	float result = x + ((x - cx) * k * ((x - cx) * (x - cx) + (y - cy) * (y - cy)));
	return result;
}

float barrelDistortion::getRadialY(float x, float y, float cx, float cy, float k)
{
	x = (x * xscale + xshift);
    	y = (y * yscale + yshift);
    	float result = y + ((y - cy) * k * ((x - cx) * (x - cx) + (y - cy) * (y - cy)));
	return result;
}

void barrelDistortion::sampleImage(Mat& src, float idx0, float idx1, Scalar& result)
{
	// if one of index is out-of-bound
	if((idx0 < 0) ||
		(idx1 < 0) ||
		(idx0 > height - 1) ||
		(idx1 > width - 1))
	{
		//temp = Scalar(0, 0, 0, 0);
		result.val[0] = 0;
		result.val[1] = 0;
		result.val[2] = 0;
		result.val[3] = 0;
		return;
	}

	float idx0_floor = floor(idx0);
    	float idx0_ceil = ceil(idx0);
	float idx1_floor = floor(idx1);
    	float idx1_ceil = ceil(idx1);

	Scalar s1 = src.at<Scalar>((int)idx0_floor, (int)idx1_floor);
	Scalar s2 = src.at<Scalar>((int)idx0_floor, (int)idx1_ceil);
	Scalar s3 = src.at<Scalar>((int)idx0_ceil, (int)idx1_ceil);
	Scalar s4 = src.at<Scalar>((int)idx0_ceil, (int)idx1_floor);

	float x = idx0 - idx0_floor;
	float y = idx1 - idx1_floor;

	result.val[0] = s1.val[0] * (1 - x) * (1 - y) + s2.val[0] * (1 - x) * y + s3.val[0] * x * y + s4.val[0] * x * (1 - y);
	result.val[1] = s1.val[1] * (1 - x) * (1 - y) + s2.val[1] * (1 - x) * y + s3.val[1] * x * y + s4.val[1] * x * (1 - y);
	result.val[2] = s1.val[2] * (1 - x) * (1 - y) + s2.val[2] * (1 - x) * y + s3.val[2] * x * y + s4.val[2] * x * (1 - y);
	result.val[3] = s1.val[3] * (1 - x) * (1 - y) + s2.val[3] * (1 - x) * y + s3.val[3] * x * y + s4.val[3] * x * (1 - y);
}

void barrelDistortion::sampleImageTest(Mat& src, float idx0, float idx1, cv::Vec3b& result)
{
	// if one of index is out-of-bound
	if((idx0 < 0) ||
		(idx1 < 0) ||
		(idx0 > height - 1) ||
		(idx1 > width - 1))
	{
		//temp = Scalar(0, 0, 0, 0);
		result.val[0] = 0;
		result.val[1] = 0;
		result.val[2] = 0;
		//result.val[3] = 0;
		return;
	}

	float idx0_floor = floor(idx0);
    	float idx0_ceil = ceil(idx0);
	float idx1_floor = floor(idx1);
    	float idx1_ceil = ceil(idx1);

	cv::Vec3b s1 = src.at<cv::Vec3b>((int)idx0_floor, (int)idx1_floor);
	cv::Vec3b s2 = src.at<cv::Vec3b>((int)idx0_floor, (int)idx1_ceil);
	cv::Vec3b s3 = src.at<cv::Vec3b>((int)idx0_ceil, (int)idx1_ceil);
	cv::Vec3b s4 = src.at<cv::Vec3b>((int)idx0_ceil, (int)idx1_floor);

	float x = idx0 - idx0_floor;
	float y = idx1 - idx1_floor;

	result.val[0] = s1.val[0] * (1 - x) * (1 - y) + s2.val[0] * (1 - x) * y + s3.val[0] * x * y + s4.val[0] * x * (1 - y);
	result.val[1] = s1.val[1] * (1 - x) * (1 - y) + s2.val[1] * (1 - x) * y + s3.val[1] * x * y + s4.val[1] * x * (1 - y);
	result.val[2] = s1.val[2] * (1 - x) * (1 - y) + s2.val[2] * (1 - x) * y + s3.val[2] * x * y + s4.val[2] * x * (1 - y);
	//result.val[3] = s1.val[3] * (1 - x) * (1 - y) + s2.val[3] * (1 - x) * y + s3.val[3] * x * y + s4.val[3] * x * (1 - y);
}
