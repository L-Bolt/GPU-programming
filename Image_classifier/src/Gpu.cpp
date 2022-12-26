#include "include/Gpu.h"


Gpu::Gpu(std::vector<std::string> source_paths) {
    this->source_paths = source_paths;

    try {
        this->platform = get_platform();
        this->device = get_default_device();
        this->context = make_context();
        this->program = make_program();

        std::cout << "Using platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "' from '" << platform.getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
        std::cout << "Using GPU '" << device.getInfo<CL_DEVICE_NAME>() << "'\n" << std::endl;
    }
    catch(...) {
        this->enabled = false;
        std::cout << "Could not find a GPU to use. continuing only on CPU" << std::endl;
    }
}

void Gpu::test() {
    build_program();
    char buf[16];
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(program, "helloWorld", nullptr);
    kernel.setArg(0, memBuf);
    cl::CommandQueue queue(context, device);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);
    std::cout << "Message from GPU: " << std::endl;
    std::cout << buf << std::endl;
}

cl::Context Gpu::make_context() {
    cl::Context context(this->device);
    return context;
}

bool Gpu::build_program() {
    cl_int err = program.build();
    if(err != CL_BUILD_SUCCESS){
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
        << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return false;
    }
    return true;
}

cl::Program Gpu::make_program() {
    // TODO als de gpu_test() functie niet hello world print moet deze code weer
    // aan.
    // std::vector<std::string> kernels;
    // for (std::string source_path : this->source_paths) {
    //     std::ifstream kernel_file(source_path);
    //     std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    //     kernels.push_back(src);
    // }
    // cl::Program program(this->context, kernels);

    cl::Program program(this->context, this->source_paths);
    return program;
}

cl::Platform Gpu::get_platform() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()){
        std::cerr << "No platforms found!" << std::endl;
        throw("No platforms found!\n");
    }

    return platforms.front();
}

cl::Device Gpu::get_default_device(){
    // Search for all the devices on the first platform and check if
    // there are any available.
    cl::Platform platform = this->platform;
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()){
        std::cerr << "No devices found!" << std::endl;
        throw("No devices found!\n");
    }

    // Return the first device found.
    return devices.front();
}