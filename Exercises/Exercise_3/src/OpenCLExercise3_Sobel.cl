#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//TODO

int getIndexGlobal(unsigned long countX, int i, int j) {
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image
float getValueGlobal(const float* a, unsigned long countX, unsigned long countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY) {
        return 0;
    }
	else {
		return a[getIndexGlobal(countX, i, j)];
    }
}

__kernel void sobelKernel1(__global const float* d_input, __global float* d_output) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);

    float Gx = getValueGlobal(d_input, countX, countY, i-1, j-1)+2*getValueGlobal(d_input, countX, countY, i-1, j)+getValueGlobal(d_input, countX, countY, i-1, j+1)
            -getValueGlobal(d_input, countX, countY, i+1, j-1)-2*getValueGlobal(d_input, countX, countY, i+1, j)-getValueGlobal(d_input, countX, countY, i+1, j+1);
    float Gy = getValueGlobal(d_input, countX, countY, i-1, j-1)+2*getValueGlobal(d_input, countX, countY, i, j-1)+getValueGlobal(d_input, countX, countY, i+1, j-1)
            -getValueGlobal(d_input, countX, countY, i-1, j+1)-2*getValueGlobal(d_input, countX, countY, i, j+1)-getValueGlobal(d_input, countX, countY, i+1, j+1);
    d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

__kernel void sobelKernel2(__global const float* d_input, __global float* d_output) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);

    float upper_left = getValueGlobal(d_input, countX, countY, i-1, j-1);
    float upper_right = getValueGlobal(d_input, countX, countY, i-1, j+1);
    float lower_left = getValueGlobal(d_input, countX, countY, i+1, j-1);
    float lower_right = getValueGlobal(d_input, countX, countY, i+1, j+1);

    float Gx = upper_left+2*getValueGlobal(d_input, countX, countY, i-1, j)+upper_right
            -lower_left-2*getValueGlobal(d_input, countX, countY, i+1, j)-lower_right;
    float Gy = upper_left+2*getValueGlobal(d_input, countX, countY, i, j-1)+lower_left
            -upper_right-2*getValueGlobal(d_input, countX, countY, i, j+1)-lower_right;
    d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void sobelKernel3(__read_only image2d_t d_input, __global float* d_output) {

    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);

    //float upper_left = read_imagef(d_input, sampler, (int2){i-1, j-1}).x;
    float upper_left = read_imagef(d_input, sampler, (int2){0,0}).x;
    // float upper_right = read_imagef(d_input, sampler, (int2){i-1, j+1}).x;
    // float lower_left = read_imagef(d_input, sampler, (int2){i+1, j-1}).x;
    // float lower_right = read_imagef(d_input, sampler, (int2){i+1, j+1}).x;

    // float Gx = upper_left+2*read_imagef(d_input, sampler, (int2){i-1, j}).x+upper_right
    //            -lower_left-2*read_imagef(d_input, sampler, (int2){i+1, j}).x-lower_right;
    // float Gy = upper_left+2*read_imagef(d_input, sampler, (int2){i, j-1}).x+lower_left
    //            -upper_right-2*read_imagef(d_input, sampler, (int2){i, j+1}).x-lower_right;

    // d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
    //d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}
