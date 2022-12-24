#ifndef GPU
#define GPU

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <OpenCL/OpenCL.hpp>

class Gpu {

    public:
        Gpu(std::string source_path);
        ~Gpu() = default;

        bool build_program();
        bool gpu_enabled() const {return enabled;};
        void test();

    private:
        std::string source_path;
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