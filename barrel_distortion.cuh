#ifndef BARREL_DISTORTION_HPP
#define BARREL_DISTORTION_HPP

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

using cv::Mat;
using cv::Scalar;

class barrelDistortion {
public:
	barrelDistortion(Mat& src, Mat& dst,
		float K,
		float centerX, float centerY,
		int width, int height);

	void barrel_distort();
private:
	Mat src;
	Mat dst;
	float K;
	float centerX;
	float centerY;
	int width;
	int height;

	float thresh = 1;
	float xscale;
	float yscale;
	float xshift;
	float yshift;

	float calc_shift(float x1, float x2, float cx, float k);
	float getRadialX(float x, float y, float cx, float cy, float k);
	float getRadialY(float x, float y, float cx, float cy, float k);
	__global__ void barrel_distort_kernel();
	__device__ void sampleImageTest(Mat& src, float idx0, float idx1, cv::Vec3b& result);
};

#endif
