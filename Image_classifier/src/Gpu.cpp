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

        std::cout << "Using platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "' from '" << platform.getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
        std::cout << "Using GPU '" << device.getInfo<CL_DEVICE_NAME>() << "'\n" << std::endl;
        build_program();
    }
    catch(...) {
        this->enabled = false;
        std::cout << "Could not find a GPU to use. continuing only on CPU" << std::endl;
    }
}

void Gpu::forward_prop(std::vector<Image> *input, std::vector<std::vector<double>> *a1, std::vector<std::vector<double>> *a2, std::vector<std::vector<double>> *z1, Matrix2D<double> *weights0, Matrix2D<double> *weights1, Matrix2D<double> *bias0, Matrix2D<double> *bias1) {
    std::vector<double> d_input1(input->size() * input->at(0).preprocessed_data.size());
    std::vector<double> d_outputA1(input->size() * 196);
    std::vector<double> d_outputZ1(input->size() * 196);
    std::vector<double> d_input2(d_outputA1.size());
    std::vector<double> d_outputA2(input->size() * 10);
    std::vector<double> d_outputZ2(input->size() * 10);

    for (int i = 0; i < input->size(); i++) {
        for (int j = 0; j < input->at(i).preprocessed_data.size(); j++) {
            d_input1.at(i * input->at(i).preprocessed_data.size() + j) = input->at(i).preprocessed_data.at(j);
        }
    }

    Matrix2D<double> weights_transposed = weights0->transpose();

    cl::Buffer input_buffer1(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * d_input1.size(), d_input1.data());
    cl::Buffer weights0_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * weights0->get_columns() * weights0->get_rows(), weights_transposed.data()->data());
    cl::Buffer biases0_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * bias0->get_columns() * bias0->get_rows(), bias0->data()->data());
    cl::Buffer output_bufferA1(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * d_outputA1.size(), d_outputA1.data());
    cl::Buffer output_bufferZ1(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * d_outputZ1.size(), d_outputZ1.data());
    cl::Buffer output_bufferA2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * d_outputA2.size(), d_outputA2.data());
    cl::Buffer output_bufferZ2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * d_outputZ2.size(), d_outputZ2.data());

    cl::Kernel kernel(program, "forward_pass", nullptr);
    cl::CommandQueue queue(context, device);

    kernel.setArg(0, input_buffer1);
    kernel.setArg(1, output_bufferA1);
    kernel.setArg(2, output_bufferZ1);
    kernel.setArg(3, weights0_buffer);
    kernel.setArg(4, biases0_buffer);
    kernel.setArg(5, (int) input->size());
    kernel.setArg(6, 196);
    kernel.setArg(7, 0);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(196));
    queue.finish();
    queue.enqueueReadBuffer(output_bufferA1, CL_TRUE, 0, sizeof(double) * d_outputA1.size(), d_outputA1.data());
    queue.enqueueReadBuffer(output_bufferZ1, CL_TRUE, 0, sizeof(double) * d_outputZ1.size(), d_outputZ1.data());

    Matrix2D<double> weights_transposed2 = weights1->transpose();

    cl::Buffer input_buffer2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * d_outputA1.size(), d_outputA1.data());
    cl::Buffer weights1_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * weights1->get_columns() * weights1->get_rows(), weights_transposed2.data()->data());
    cl::Buffer biases1_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * bias1->get_columns() * bias1->get_rows(), bias1->data()->data());

    cl::Kernel kernel2(program, "forward_pass", nullptr);
    cl::CommandQueue queue2(context, device);

    kernel2.setArg(0, input_buffer2);
    kernel2.setArg(1, output_bufferA2);
    kernel2.setArg(2, output_bufferZ2);
    kernel2.setArg(3, weights1_buffer);
    kernel2.setArg(4, biases1_buffer);
    kernel2.setArg(5, (int) input->size());
    kernel2.setArg(6, 10);
    kernel2.setArg(7, 1);

    queue2.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(10));
    queue2.finish();

    // na de eerste epoch wordt dit allemaal inf.
    queue2.enqueueReadBuffer(output_bufferA2, CL_TRUE, 0, sizeof(double) * d_outputA2.size(), d_outputA2.data());
    queue2.enqueueReadBuffer(output_bufferZ2, CL_TRUE, 0, sizeof(double) * d_outputZ2.size(), d_outputZ2.data());

    *a1 = util::split_vector(d_outputA1, input->size());

    *a2 = util::split_vector(d_outputA2, input->size());
    for (int i = 0; i < a2->size(); i++) {
        a2->at(i) = np::normalize(a2->at(i));
    }

    *z1 = util::split_vector(d_outputZ1, input->size());
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
    // TODO kan kernel arguments setten misschien met een functie?
    // die een vector van void pointers binnen krijgt waar alles in zit
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
    // TODO kan kernel arguments setten misschien met een functie?
    // die een vector van void pointers binnen krijgt waar alles in zit
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
    kernel.setArg(11, size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(out_rows, out_cols));
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
    kernel.setArg(9, size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows/pooling_window.rows, cols/pooling_window.columns));
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
