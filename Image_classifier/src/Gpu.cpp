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

double Gpu::forward_prop(std::vector<Image> *input, std::vector<std::vector<double>> *ytrain_raw, std::vector<std::vector<double>> *a, std::vector<std::vector<double>> *z, Matrix2D<double> *weights0, Matrix2D<double> *weights1, Matrix2D<double> *bias0, Matrix2D<double> *bias1, int out_dim, int hid_nodes, int learning_rate) {
    std::vector<std::vector<double>> flattened_pool(input->size(), std::vector<double>(input->at(1).preprocessed_data.size()));
    for (size_t i = 0; i < input->size(); i++) {
        flattened_pool.at(i) = input->at(i).preprocessed_data;
    }
    std::vector<double> flattened_flattened_pool(flattened_pool.size() * flattened_pool.at(0).size());
    for (auto const &v: flattened_pool) {
        flattened_flattened_pool.insert(flattened_flattened_pool.end(), v.begin(), v.end());
    }
    std::vector<double> ytrain;
    for (auto const &v: *ytrain_raw) {
        ytrain.insert(ytrain.end(), v.begin(), v.end());
    }
    std::vector<double> Z1(weights0->transpose().dot(flattened_pool[0]).size());
    std::vector<double> A1(Z1.size());
    std::vector<std::vector<double>> A1_2D(input->size(), std::vector<double>(Z1.size()));
    std::vector<double> Z2(weights1->transpose().dot(A1_2D[0]).size());
    std::vector<double> A2(Z2.size());
    std::vector<double> dZ2(Z2.size());
    std::vector<double> temp(dZ2.size());
    std::vector<double> temp2(dZ2.size());
    std::vector<double> temp3(A1_2D.at(0).size());
    std::vector<double> dW2(out_dim * hid_nodes);
    std::vector<double> dZ1(weights1->get_rows());
    std::vector<double> dz_deriv(Z1.size());
    std::vector<double> temp4(flattened_pool.at(0).size());
    std::vector<double> dW1(weights1->get_rows() * flattened_pool.at(0).size());
    std::vector<double> dW1trans(dW1.size());
    std::vector<double> dW2trans(dW2.size());
    std::vector<double> weights0_trans(weights0->array.size());
    std::vector<double> temp5(flattened_pool.at(0).size());
    std::vector<double> weights1_trans(weights1->array.size());
    double error;
    std::vector<double> z_temp(Z2.size());
    cl::Buffer memBuf1(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * bias0->array.size(), bias0->array.data());
    cl::Buffer memBuf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * Z1.size(), Z1.data());
    cl::Buffer memBuf3(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * A1.size(), A1.data());
    cl::Buffer memBuf4(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dZ2.size(), dZ2.data());
    cl::Buffer memBuf5(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * ytrain.size(), ytrain.data());
    cl::Buffer memBuf6(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * temp2.size(), temp2.data());
    cl::Buffer memBuf7(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * A2.size(), A2.data());
    cl::Buffer memBuf8(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dW2.size(), dW2.data());
    cl::Buffer memBuf9(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * weights1->array.size(), weights1->array.data());
    cl::Buffer memBuf10(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dZ1.size(), dZ1.data());
    cl::Buffer memBuf11(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dz_deriv.size(), dz_deriv.data());
    cl::Buffer memBuf12(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * Z2.size(), Z2.data());
    cl::Buffer memBuf13(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * flattened_flattened_pool.size(), flattened_flattened_pool.data());
    cl::Buffer memBuf14(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dW1.size(), dW1.data());
    cl::Buffer memBuf15(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dW1trans.size(), dW1trans.data());
    cl::Buffer memBuf16(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * weights0->array.size(), weights0->array.data());
    cl::Buffer memBuf17(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * bias1->array.size(), bias1->array.data());
    cl::Buffer memBuf18(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * dW2trans.size(), dW2trans.data());
    cl::Buffer memBuf19(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * weights0_trans.size(), weights0_trans.data());
    cl::Buffer memBuf20(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * temp5.size(), temp5.data());
    cl::Buffer memBuf21(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * weights1_trans.size(), weights1_trans.data());    
    cl::Buffer memBuf22(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double), &error);
    cl::Buffer memBuf23(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * z_temp.size(), z_temp.data());
    cl::Kernel kernel2(program, "forward_pass", nullptr);
    cl::CommandQueue queue2(context, device);
    kernel2.setArg(0, memBuf1);
    kernel2.setArg(1, memBuf2);
    kernel2.setArg(2, memBuf3);
    kernel2.setArg(3, bias0->get_columns());
    kernel2.setArg(4, (int) Z1.size());
    kernel2.setArg(5, memBuf4);
    kernel2.setArg(6, memBuf5);
    kernel2.setArg(7, memBuf6);
    kernel2.setArg(8, (int) Z2.size());
    kernel2.setArg(9, memBuf7);
    kernel2.setArg(10, memBuf8);
    kernel2.setArg(11, out_dim);
    kernel2.setArg(12, hid_nodes);
    kernel2.setArg(13, learning_rate);
    kernel2.setArg(14, memBuf9);
    kernel2.setArg(15, weights1->get_rows());
    kernel2.setArg(16, weights1->get_columns());
    kernel2.setArg(17, memBuf10);
    kernel2.setArg(18, memBuf11);
    kernel2.setArg(19, memBuf12);
    kernel2.setArg(20, memBuf13);
    kernel2.setArg(21, memBuf14);
    kernel2.setArg(22, (int) flattened_pool.at(0).size());
    kernel2.setArg(23, memBuf15);
    kernel2.setArg(24, memBuf16);
    kernel2.setArg(25, weights0->get_columns());
    kernel2.setArg(26, memBuf17);
    kernel2.setArg(27, bias1->get_rows());
    kernel2.setArg(28, bias1->get_columns());
    kernel2.setArg(29, memBuf18);
    kernel2.setArg(30, (int) input->size());
    kernel2.setArg(31, weights0->get_rows());
    kernel2.setArg(32, memBuf19);
    kernel2.setArg(33, memBuf20);
    kernel2.setArg(34, memBuf21);
    kernel2.setArg(35, (int) ytrain_raw->at(0).size());
    kernel2.setArg(36, bias0->get_rows());
    kernel2.setArg(37, memBuf22);
    kernel2.setArg(38, memBuf23);
    queue2.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(1));
    queue2.enqueueReadBuffer(memBuf16, CL_TRUE, 0, sizeof(double) * weights0->array.size(), weights0->array.data());
    queue2.enqueueReadBuffer(memBuf9, CL_TRUE, 0, sizeof(double) * weights1->array.size(), weights1->array.data());
    queue2.enqueueReadBuffer(memBuf1, CL_TRUE, 0, sizeof(double) * bias0->array.size(), bias0->array.data());
    queue2.enqueueReadBuffer(memBuf17, CL_TRUE, 0, sizeof(double) * bias1->array.size(), bias1->array.data());
    queue2.enqueueReadBuffer(memBuf22, CL_TRUE, 0, sizeof(double), &error);
    return error;
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
