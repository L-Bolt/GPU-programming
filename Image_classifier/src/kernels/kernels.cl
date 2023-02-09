__kernel void normalization(__global unsigned char* restrict images,
                            __global double* restrict output,
                            const int rows,
                            const int cols,
                            const int channels,
                            const int size) {
    int id = (get_global_id(2) * get_global_size(0) * get_global_size(1)) + (get_global_id(0) * get_global_size(0)) + get_global_id(1);
    for (int i = 0; i < size; i++) {
        double value = ((double) images[(i*rows*cols*channels)+(id+1)] / (double) 255);
        output[(i*rows*cols*channels) + id] = value;
    }
}

__kernel void convolve(__global double* restrict image,
                       __global double* restrict _kernel,
                       __global double* restrict output,
                       const int ker_rows,
                       const int ker_cols,
                       const int im_rows,
                       const int im_cols,
                       const int im_channels,
                       const int out_cols,
                       const int out_rows,
                       const double bias) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int q = get_global_id(2);
    double value = 0.0;
    for (int h = i; h < i + ker_rows; h++) {
        for (int w = j; w < j + ker_cols; w++) {
            for (int channel = 0; channel < im_channels; channel++) {
                value += _kernel[(channel * ker_rows * ker_cols) + ((h - i) * ker_rows) + (w - j)] * image[((im_rows * im_cols * im_channels + 1) * q) + (channel * im_rows * im_cols) + (h * im_rows) + w];
            }
        }
    }
    output[(out_cols * out_rows * q) + (i * out_cols + j)] = value + bias;
}

__kernel void max_pool(__global double* restrict image,
                       __global double* restrict output,
                       const int rows,
                       const int cols,
                       const int out_rows,
                       const int out_cols,
                       const int pl_rows,
                       const int pl_cols,
                       const double dbl_max) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int q = get_global_id(2);
    double max = -dbl_max;
    for (int x = i * pl_rows; x < (i * pl_rows) + pl_rows; x++) {
        for (int y = j * pl_cols; y < (j * pl_cols) + pl_cols; y++) {
            double val_at_index = image[(rows * cols * q) + (x * cols + y)];
            max = val_at_index > max ? val_at_index : max;
        }
    }
    output[(out_cols * out_rows * q) + (i * out_cols + j)] = max;
}

double relu(double x) {
    if (x > 0.0) {
        return x;
    }
    return 0.0;
}

double softmax(double x){
    if (isnan(x)) {
        return 0.0;
    }
    return (double) exp(x);
}

__kernel void forward_pass(__global double* restrict biases,
                           __global double* restrict Z1,
                           __global double* restrict A1,
                           const int bias_cols,
                           const int Z1_size,
                           const int func_selector) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    Z1[j + (i * Z1_size)] = Z1[j + (i * Z1_size)] + biases[j * bias_cols];
    if (func_selector == 0) {
        A1[j + (i * Z1_size)] = relu(Z1[j + (i * Z1_size)]);
    } else {
        A1[j + (i * Z1_size)] = softmax(Z1[j + (i * Z1_size)]);
        A1[j + (i * Z1_size)] = isnan(A1[j + (i * Z1_size)]) ? 1.0 : A1[j + (i * Z1_size)];
    }
}