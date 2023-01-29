#ifndef GPU
#define GPU

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <OpenCL/OpenCL.hpp>

#include "Image.h"
#include "Cnn.h"

class Gpu {
    public:
        Gpu() = default;
        Gpu(std::vector<std::string> source_paths);
        ~Gpu() = default;

        bool build_program();
        bool gpu_enabled() const {return enabled;};
        std::vector<double> normalize(std::vector<std::vector<unsigned char>>* images, int rows, int cols, int channels, int size);
        std::vector<double> convolute(std::vector<double> input, Matrix3D<double> conv_kernel, double bias, int rows, int cols, int channels, int size);
        std::vector<Matrix2D<double>> max_pooling(std::vector<double> input, int size, int rows, int cols, Shape &pooling_window);
        std::vector<Matrix2D<double>> preprocess(std::vector<std::vector<unsigned char>>* images, Matrix3D<double> conv_kernel, int rows, int cols, int channels, Shape &pooling_window, double bias=1.0);
        std::vector<std::vector<double>> forward_prop(std::vector<Image> *input, std::vector<std::vector<double>> *a, std::vector<std::vector<double>> *z);

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
