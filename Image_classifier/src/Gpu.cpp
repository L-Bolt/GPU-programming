#include "include/Gpu.h"
#include "include/Cnn.h"


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

void Gpu::test(Image input) {
    Matrix3D<double> output = Matrix3D<double>(input.matrix.get_rows(), input.matrix.get_columns(), input.matrix.get_channels());
    build_program();
    cl::Buffer memBuf(context, CL_MEM_READ_WRITE, sizeof(input));
    cl::Buffer memBuf2(context, CL_MEM_READ_WRITE, sizeof(output));
    cl::Kernel kernel(program, "proc", nullptr);
    cl::CommandQueue queue(context, device);
    queue.enqueueWriteBuffer(memBuf, CL_TRUE, 0, sizeof(memBuf), &input, NULL, NULL);
    kernel.setArg(0, memBuf);
    kernel.setArg(1, memBuf2);
    kernel.setArg(2, input.matrix.get_rows());
    kernel.setArg(3, input.matrix.get_columns());
    kernel.setArg(4, input.matrix.get_channels());
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(memBuf2, CL_TRUE, 0, sizeof(memBuf2), &output);
    std::cout << "Message from GPU: " << std::endl;
    std::cout << output.get(2, 8, 2) << std::endl;
}

// void Gpu::convolve(int input_channels, int input_size, int pad, int stride, int start_channel, int output_size, std::vector<Image> &input_im, std::vector<double> output_im) {
//     build_program();

// }

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
    std::vector<std::string> kernels;
    for (std::string source_path : this->source_paths) {
        std::ifstream kernel_file(source_path);
        std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
        kernels.push_back(src);
    }
    cl::Program program(this->context, kernels);

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