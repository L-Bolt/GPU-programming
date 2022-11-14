#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__attribute__((reqd_work_group_size(WG_SIZE, WG_SIZE, 1)))
__kernel void matrixMulKernel1(__global const float* d_inputA, __global const float* d_inputB, __global float* d_output, unsigned int countAY, unsigned int countBX, unsigned int countAX_BY) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

	float sum = 0;
	int k = get_local_id(0);
	int g = get_local_id(1);
	__local float l_A[WG_SIZE][WG_SIZE];
	__local float l_B[WG_SIZE][WG_SIZE];

	for (uint bs = 0; bs < countAX_BY; bs += WG_SIZE) {
		l_A[g][k] = d_inputA[(k + bs) + j * countAX_BY];
		l_B[g][k] = d_inputB[i + (g + bs) * countBX];

		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint m = 0; m < WG_SIZE; m++) {
			sum += l_A[g][m] * l_B[m][k];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	d_output[i + j * countBX] = sum;
}


// The preprocessor constant WG_SIZE will contain the size of a work group in X/Y-direction

// __attribute__((reqd_work_group_size(WG_SIZE, WG_SIZE, 1)))
// __kernel void matrixMulKernel2(/*...*/) {
// 	//TODO
// }
