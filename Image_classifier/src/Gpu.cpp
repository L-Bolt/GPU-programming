#include "include/Gpu.h"


Gpu::Gpu(std::vector<std::string> source_paths) {
    this->source_paths = source_paths;

    // TODO: kan weg wanneer gpu bug op maccie gefixt is.
    #if defined(__APPLE__)
        this->enabled = false;
    #endif

    try {
        this->platform = get_platform();
        this->device = get_default_device();
        this->context = make_context();
        this->program = make_program();
        build_program();

        std::cout << "Using platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "' from '" << platform.getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
        std::cout << "Using GPU '" << device.getInfo<CL_DEVICE_NAME>() << "'\n" << std::endl;
    }
    catch(...) {
        this->enabled = false;
        std::cout << "Could not find a GPU to use. continuing only on CPU" << std::endl;
    }
}

void Gpu::forward_prop(std::vector<Image> *input, std::vector<std::vector<double>> *a, std::vector<std::vector<double>> *z, Matrix2D<double> *weights0, Matrix2D<double> *weights1, Matrix2D<double> *bias0, Matrix2D<double> *bias1) {
    std::vector<std::vector<double>> flattened_pool(input->size(), std::vector<double>(input->at(1).preprocessed_data.size()));
    for (size_t i = 0; i < input->size(); i++) {
        flattened_pool.at(i) = input->at(i).preprocessed_data;
    }
    std::vector<double> biases;
    std::vector<double> biases1;
    biases = std::vector<double>(bias0->array.begin(), bias0->array.end());
    biases1 = std::vector<double>(bias1->array.begin(), bias1->array.end());
    std::vector<std::vector<double>> raw_Z1(input->size(), std::vector<double>(weights0->transpose().dot(flattened_pool[0]).size()));
    for (size_t j = 0; j < input->size(); j++) {
        raw_Z1.at(j) = weights0->transpose().dot(flattened_pool.at(j));
    }
    std::vector<double> Z1;
    for (auto const &v: raw_Z1) {
        Z1.insert(Z1.end(), v.begin(), v.end());
    }
    std::vector<double> A1(Z1.size());
    cl::Buffer memBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * biases.size(), biases.data());
    cl::Buffer memBuf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * Z1.size(), Z1.data());
    cl::Buffer memBuf3(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * A1.size(), A1.data());
    cl::Kernel kernel(program, "forward_pass", nullptr);
    cl::CommandQueue queue(context, device);
    kernel.setArg(0, memBuf);
    kernel.setArg(1, memBuf2);
    kernel.setArg(2, memBuf3);
    kernel.setArg(3, bias0->get_columns());
    kernel.setArg(4, (int) raw_Z1.at(0).size());
    kernel.setArg(5, 0);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange((int) input->size(), (int) raw_Z1.at(0).size()));
    queue.enqueueReadBuffer(memBuf2, CL_TRUE, 0, sizeof(double) * Z1.size(), Z1.data());
    queue.enqueueReadBuffer(memBuf3, CL_TRUE, 0, sizeof(double) * A1.size(), A1.data());
    std::vector<std::vector<double>> A1_2D(raw_Z1.size(), std::vector<double>(raw_Z1.at(0).size()));
    for (size_t k = 0; k < raw_Z1.size(); k++) {
        for (size_t l = 0; l < raw_Z1.at(0).size(); l++) {
            A1_2D.at(k).at(l) = A1.at(l + (k * raw_Z1.at(0).size()));
        }
    }
    std::vector<std::vector<double>> raw_Z2(input->size(), std::vector<double>(weights1->transpose().dot(A1_2D[0]).size()));
    for (size_t m = 0; m < input->size(); m++) {
        raw_Z2.at(m) = weights1->transpose().dot(A1_2D.at(m));
    }
    std::vector<double> Z2;
    for (auto const &v: raw_Z2) {
        Z2.insert(Z2.end(), v.begin(), v.end());
    }
    std::vector<double> A2(Z2.size());
    cl::Buffer memBuf4(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * biases1.size(), biases1.data());
    cl::Buffer memBuf5(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * Z2.size(), Z2.data());
    cl::Buffer memBuf6(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * A2.size(), A2.data());
    cl::Kernel kernel2(program, "forward_pass", nullptr);
    cl::CommandQueue queue2(context, device);
    kernel2.setArg(0, memBuf4);
    kernel2.setArg(1, memBuf5);
    kernel2.setArg(2, memBuf6);
    kernel2.setArg(3, bias1->get_columns());
    kernel2.setArg(4, (int) raw_Z2.at(0).size());
    kernel2.setArg(5, 1);
    queue2.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange((int) input->size(),(int) raw_Z2.at(0).size()));
    queue2.enqueueReadBuffer(memBuf5, CL_TRUE, 0, sizeof(double) * Z2.size(), Z2.data());
    queue2.enqueueReadBuffer(memBuf6, CL_TRUE, 0, sizeof(double) * A2.size(), A2.data());
    std::vector<std::vector<double>> A2_2D(raw_Z2.size(), std::vector<double>(raw_Z2.at(0).size()));
    for (size_t n = 0; n < raw_Z2.size(); n++) {
        for (size_t o = 0; o < raw_Z2.at(0).size(); o++) {
            A2_2D.at(n).at(o) = A2.at(o + (n * raw_Z2.at(0).size()));
        }
        A2_2D.at(n) = np::normalize(A2_2D.at(n));
        a->push_back(A1_2D.at(n));
        a->push_back(A2_2D.at(n));
    }
    std::vector<std::vector<double>> Z1_2D(raw_Z1.size(), std::vector<double>(raw_Z1.at(0).size()));
    std::vector<std::vector<double>> Z2_2D(raw_Z2.size(), std::vector<double>(raw_Z2.at(0).size()));
    for (size_t p = 0; p < raw_Z1.size(); p++) {
        for (size_t q = 0; q < raw_Z1.at(0).size(); q++) {
            Z1_2D.at(p).at(q) = Z1.at(q + (p * raw_Z1.at(0).size()));
        }
        for (size_t r = 0; r < raw_Z2.at(0).size(); r++) {
            Z2_2D.at(p).at(r) = Z2.at(r + (p * raw_Z2.at(0).size()));
        }
        z->push_back(Z1_2D.at(p));
        z->push_back(Z2_2D.at(p));
    }
}

std::vector<Matrix2D<double>> Gpu::preprocess(std::vector<std::vector<unsigned char>>* images, Matrix3D<double> conv_kernel, int rows, int cols, int channels, Shape &pooling_window, double bias) {
    int size = images->size();
    return Gpu::max_pooling(Gpu::convolute(Gpu::normalize(images, rows, cols, channels, size), conv_kernel, bias, rows, cols, channels, size), size, (rows - conv_kernel.get_rows() + 1), (cols - conv_kernel.get_columns() + 1), pooling_window);
}

std::vector<double> Gpu::normalize(std::vector<std::vector<unsigned char>>* images, int rows, int cols, int channels, int size) {
    std::vector<unsigned char> test = images->front();
    std::vector<double> output(images->size() * images->front().size());
    std::vector<unsigned char> flattened;
    for (auto const &v: *images) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }

    cl::Buffer memBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * flattened.size(), flattened.data());
    cl::Buffer memBuf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * output.size(), output.data());
    cl::Kernel kernel(program, "normalization", nullptr);
    cl::CommandQueue queue(context, device);
    kernel.setArg(0, memBuf);
    kernel.setArg(1, memBuf2);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);
    kernel.setArg(4, channels);
    kernel.setArg(5, size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows, cols, channels));
    queue.enqueueReadBuffer(memBuf2, CL_TRUE, 0, sizeof(double) * output.size(), output.data());
    return output;
}

std::vector<double> Gpu::convolute(std::vector<double> input, Matrix3D<double> conv_kernel, double bias, int rows, int cols, int channels, int size) {
    int out_rows = (rows - conv_kernel.get_rows() + 1);
    int out_cols = (cols - conv_kernel.get_columns() + 1);
    std::vector<double> output = std::vector<double>(out_rows * out_cols * size);

    cl::Buffer memBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * input.size(), input.data());
    cl::Buffer memBuf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * conv_kernel.array.size(), conv_kernel.array.data());
    cl::Buffer memBuf3(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * output.size(), output.data());
    cl::Kernel kernel(program, "convolve", nullptr);
    cl::CommandQueue queue(context, device);
    kernel.setArg(0, memBuf);
    kernel.setArg(1, memBuf2);
    kernel.setArg(2, memBuf3);
    kernel.setArg(3, conv_kernel.get_rows());
    kernel.setArg(4, conv_kernel.get_columns());
    kernel.setArg(5, rows);
    kernel.setArg(6, cols);
    kernel.setArg(7, channels);
    kernel.setArg(8, out_cols);
    kernel.setArg(9, out_rows);
    kernel.setArg(10, bias);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(out_rows, out_cols, size));
    queue.enqueueReadBuffer(memBuf3, CL_TRUE, 0, sizeof(double) * output.size(), output.data());
    return output;
}

std::vector<Matrix2D<double>> Gpu::max_pooling(std::vector<double> input, int size, int rows, int cols, Shape &pooling_window) {
    int out_rows = (rows / pooling_window.rows);
    int out_cols = (cols / pooling_window.columns);
    std::vector<double> output(size * (out_rows * out_cols));

    cl::Buffer memBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * input.size(), input.data());
    cl::Buffer memBuf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * output.size(), output.data());
    cl::Kernel kernel(program, "max_pool", nullptr);
    cl::CommandQueue queue(context, device);
    kernel.setArg(0, memBuf);
    kernel.setArg(1, memBuf2);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);
    kernel.setArg(4, out_rows);
    kernel.setArg(5, out_cols);
    kernel.setArg(6, pooling_window.rows);
    kernel.setArg(7, pooling_window.columns);
    kernel.setArg(8, DBL_MAX);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows/pooling_window.rows, cols/pooling_window.columns, size));
    queue.enqueueReadBuffer(memBuf2, CL_TRUE, 0, sizeof(double) * output.size(), output.data());
    std::vector<Matrix2D<double>> result(size, Matrix2D<double>(out_rows, out_cols));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < out_rows; j++) {
            for (int k = 0; k < out_cols; k++) {
                result[i].set(j, k, output[(j * out_cols + k) + i * (out_cols * out_rows)]);
            }
        }
    }
    return result;
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
