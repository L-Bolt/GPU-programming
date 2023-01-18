__kernel void helloWorld(__global char* data) {
    data[0] = 'H';
    data[1] = 'e';
    data[2] = 'l';
    data[3] = 'l';
    data[4] = 'o';
    data[5] = ' ';
    data[6] = 'W';
    data[7] = 'o';
    data[8] = 'r';
    data[9] = 'l';
    data[10] = 'd';
    data[11] = '!';
    data[12] = '\n';
}

__kernel void normalization(__global unsigned char* restrict image,
                            __global double* restrict output,
                            const int rows,
                            const int cols) {

    int id = (get_global_id(2) * get_global_size(0) * get_global_size(1)) + (get_global_id(0) * get_global_size(0)) + get_global_id(1);
    double value = ((double) image[id+1] / (double) 255);
    output[id] = value;
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
                       const double bias) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    double value = 0.0;
    for (int h = i; h < i + ker_rows; h++) {
        for (int w = j; w < j + ker_cols; w++) {
            for (int channel = 0; channel < im_channels; channel++) {
                value += _kernel[(channel * ker_rows * ker_cols) + ((h - i) * ker_rows) + (w - j)] * image[(channel * im_rows * im_cols) + (h * im_rows) + w];
            }
        }
    }
    output[i * out_cols + j] = value + bias;
}

// __kernel void convolve(const int input_channels, const int input_size,
//                          const int pad, const int stride,
//                          const int start_channel,
//                          const int output_size,
//                          __global double *restrict input_im,
//                          __global const float *restrict filter_weight,
//                          __global const float *restrict filter_bias,
//                          __global double *restrict output_im ) {

//     int filter_index = get_global_id(0);

//     filter_weight += filter_index * input_channels * 9;
//     float bias = filter_bias[filter_index];
//     output_im += (start_channel + filter_index) * output_size * output_size;

//     for (int i = 0; i < output_size; i++) {
//         for (int j = 0; j < output_size; j++) {
//             float tmp = bias;
//             for (int k = 0; k < input_channels; k++) {
//                 for (int l = 0; l < 3; l++) {
//                     int h = 1 * stride + 1 - pad;
//                     for (int m = 0; m < 3; m++) {
//                         int w = j * stride + m - pad;
//                         if ((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size)) {
//                             tmp += input_im[k + input_size * input_size + (i * stride + 1 - pad) * input_size + j * stride + m - pad] * filter_weight[9 * k + 3 * l + m];
//                         }
//                     }
//                 }
//             }
//             output_im[i * output_size + j] = (tmp > 0.0) ? tmp : 0.0;
//         }
//     }
// }