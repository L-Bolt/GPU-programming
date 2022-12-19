#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void kernel1 (__global const float* d_input, __global float* d_output, unsigned long size) {
	int id = get_global_id(0);

	if (id < size) {
		d_output[id] = cos(d_input[id]);
	}
}
