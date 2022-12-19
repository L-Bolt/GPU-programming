#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//TODO
__kernel void mandelbrotKernel(__global unsigned int* d_output, unsigned int niter, float xmin, float xmax, float ymin, float ymax) {
    size_t i = get_global_id(0); // x itterable
    size_t j = get_global_id(1); // y itterable
    size_t countX = get_global_size(0); // constant
    size_t countY = get_global_size(1); // constant

    float xc = xmin + (xmax - xmin) / (countX - 1) * i; //xc=real(c)
    float yc = ymin + (ymax - ymin) / (countY - 1) * j; //yc=imag(c)
    float x = 0.0; //x=real(z_k)
    float y = 0.0; //y=imag(z_k)
    for (size_t k = 0; k < niter; k = k + 1) { //iteration loop
        float tempx = x * x - y * y + xc; //z_{n+1}=(z_n)^2+c;
        y = 2 * x * y + yc;
        x = tempx;
        float r2 = x * x + y * y; //r2=|z_k|^2
        if ((r2 > 4) || k == niter - 1) { //divergence condition
            d_output[i + j * countX] = k;
            break;
        }
    }
}