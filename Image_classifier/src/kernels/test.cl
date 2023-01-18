__kernel void normalization(__global unsigned char* restrict images,
                            __global double* restrict output,
                            const int rows,
                            const int cols) {  
    int id = (get_global_id(2) * get_global_size(0) * get_global_size(1)) + (get_global_id(0) * get_global_size(0)) + get_global_id(1);
    for (int i = 0; i < 50000; i++) {
        double value = ((double) images[(i*3072)+(id+1)] / (double) 255);
        output[(i*3072) + id] = value;
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
    for (int q = 0; q < 50000; q++) {
        double value = 0.0;
        for (int h = i; h < i + ker_rows; h++) {
            for (int w = j; w < j + ker_cols; w++) {
                for (int channel = 0; channel < im_channels; channel++) {
                    value += _kernel[(channel * ker_rows * ker_cols) + ((h - i) * ker_rows) + (w - j)] * image[(3072 * q) + ((channel * im_rows * im_cols) + (h * im_rows) + w)];
                }
            }
        }
        output[(out_cols * out_rows * q) + (i * out_cols + j)] = value + bias;
    }
}