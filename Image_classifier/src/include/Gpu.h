#ifndef GPU
#define GPU

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <OpenCL/OpenCL.hpp>
#include <include/Image.h>

class Gpu {
    public:
        Gpu(std::vector<std::string> source_paths);
        ~Gpu() = default;

        bool build_program();
        bool gpu_enabled() const {return enabled;};
        std::vector<double> normalize(std::vector<std::vector<unsigned char>> images);
        std::vector<Matrix2D<double>> convolute(std::vector<double> input, Matrix3D<double> conv_kernel, double bias);

    private:
        std::vector<std::string> source_paths;
        bool enabled = true;

        cl::Device device;
        cl::Platform platform;
        cl::Context context;
        cl::Program program;

        cl::Device get_default_device();
        cl::Platform get_platform();
        cl::Program make_program();
        cl::Context make_context();

};

#endif
