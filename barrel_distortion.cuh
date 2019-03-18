#ifndef BARREL_DISTORTION_CUH
#define BARREL_DISTORTION_CUH

#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

using cv::Mat;
using cv::Scalar;

typedef struct Properties{
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
};

void barrelDistortion(Mat& _src, Mat& _dst,
	float _K,
	float _centerX, float _centerY,
	int _width, int _height);

float calc_shift(float x1, float x2, float cx, float k, float thresh);
__device__ float getRadialX(float x, float y, struct Properties* prop);
__device__ float getRadialY(float x, float y, struct Properties* prop);
__device__ void sampleImageTest(uchar3* src, float idx0, float idx1, uchar3& result, struct Properties* prop);

__global__ void barrel_distort_kernel(uchar3* src, uchar3* dst, struct Properties* prop);

#endif
