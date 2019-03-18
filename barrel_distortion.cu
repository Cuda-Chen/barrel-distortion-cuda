#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "barrel_distortion.cuh"

using std::cout;
using std::endl;
using cv::Mat;

Mat src;
Mat dst;
struct Properties prop;

void barrelDistortion(Mat& _src, Mat& _dst,
	float _K,
	float _centerX, float _centerY,
	int _width, int _height)
{
	src = _src;
	dst = _dst;

	struct Properties* prop = new Properties[1];
	prop->K = _K;
	prop->centerX = _centerX;
	prop->centerY = _centerY;
	prop->width = _width;
	prop->height = _height;

	cout << "channels: " << src.channels() 
		<< " type: " << src.type() << endl;

	prop->xshift = calc_shift(0, prop->centerX - 1, prop->centerX, prop->K, prop->thresh);
	float newcenterX = prop->width - prop->centerX;
	float xshift_2 = calc_shift(0, newcenterX - 1, newcenterX, prop->K, prop->thresh);

	prop->yshift = calc_shift(0, prop->centerY - 1, prop->centerY, prop->K, prop->thresh);
	float newcenterY = prop->height - prop->centerY;
	float yshift_2 = calc_shift(0, newcenterY - 1, newcenterY, prop->K, prop->thresh);

	prop->xscale = (prop->width - prop->xshift - xshift_2) / prop->width;
	prop->yscale = (prop->height - prop->yshift - yshift_2) / prop->height;

	cout << prop->xshift << " " << prop->yshift << " " << prop->xscale << " " << prop->yscale << endl;
	cout << prop->height << endl;
	cout << prop->width << endl;

	uchar3* d_src;
	uchar3* d_dst;
	struct Properties* d_prop;
	int imageSize = prop->height * prop->width;

	cudaMalloc(&d_src, imageSize * sizeof(uchar3));
	cudaMalloc(&d_dst, imageSize * sizeof(uchar3));
	cudaMalloc(&d_prop, sizeof(Properties));

	cudaMemcpy(d_src, src.data, imageSize * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst.data, imageSize * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prop, prop, sizeof(Properties), cudaMemcpyHostToDevice);

	// grid and block here

	// call kernel function
	barrel_distort_kernel<<<1, 1>>>(d_src, d_dst, d_prop);

	cudaMemcpy(dst.data, d_dst, imageSize * sizeof(uchar3), cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(d_prop);

	delete [] prop;
}

float calc_shift(float x1, float x2, float cx, float k, float thresh)
{
	float x3 = x1 + (x2 - x1) * 0.5;
	float result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
	float result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

	if(result1 > -thresh and result1 < thresh)
		return x1;
	if(result3 < 0)
	{
		return calc_shift(x3, x2, cx, k, thresh);
	}
	else
	{
		return calc_shift(x1, x3, cx, k, thresh);
	}
}

__device__ float getRadialX(float x, float y, struct Properties* prop)
{
	x = (x * prop->xscale + prop->xshift);
	y = (y * prop->yscale + prop->yshift);
	float result = x + ((x - prop->centerX) * prop->K * ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
	return result;
}

__device__ float getRadialY(float x, float y, struct Properties* prop)
{
	x = (x * prop->xscale + prop->xshift);
    	y = (y * prop->yscale + prop->yshift);
    	float result = y + ((y - prop->centerY) * prop->K * ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
	return result;
}

__global__ void barrel_distort_kernel(uchar3* src, uchar3* dst, struct Properties* prop)
{
	for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < prop->height; j += blockDim.x * gridDim.x)
	{
		for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < prop->width; i += blockDim.y * gridDim.y)
		{
			uchar3 temp;
			float x = getRadialX((float)i, (float)j, prop);
			float y = getRadialY((float)i, (float)j, prop);
			sampleImageTest(src, y, x, temp, prop);
			dst[(j * prop->width) + i] = temp;
		}
	}
}

__device__ void sampleImageTest(uchar3* src, float idx0, float idx1, uchar3& result, struct Properties* prop)
{
	// if one of index is out-of-bound
	if((idx0 < 0) ||
		(idx1 < 0) ||
		(idx0 > prop->height - 1) ||
		(idx1 > prop->width - 1))
	{
		//temp = Scalar(0, 0, 0, 0);
		result.x = 0;
		result.y = 0;
		result.z = 0;
		//result.val[3] = 0;
		return;
	}

	int idx0_floor = (int)floor(idx0);
    	int idx0_ceil = (int)ceil(idx0);
	int idx1_floor = (int)floor(idx1);
    	int idx1_ceil = (int)ceil(idx1);

	uchar3 s1 = src[(idx0_floor * prop->width) + idx1_floor];
	uchar3 s2 = src[(idx0_floor * prop->width) + idx1_ceil];
	uchar3 s3 = src[(idx0_ceil * prop->width) + idx1_ceil];
	uchar3 s4 = src[(idx0_ceil * prop->width) + idx1_floor];

	float x = idx0 - idx0_floor;
	float y = idx1 - idx1_floor;

	result.x = s1.x * (1 - x) * (1 - y) + s2.x * (1 - x) * y + s3.x * x * y + s4.x * x * (1 - y);
	result.y = s1.y * (1 - x) * (1 - y) + s2.y * (1 - x) * y + s3.y * x * y + s4.y * x * (1 - y);
	result.z = s1.z * (1 - x) * (1 - y) + s2.z * (1 - x) * y + s3.z * x * y + s4.z * x * (1 - y);
}
