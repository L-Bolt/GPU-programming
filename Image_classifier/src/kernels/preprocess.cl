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
                       const double bias,
                       const int size) {
    for (int q = 0; q < size; q++) {
        int i = get_global_id(0);
        int j = get_global_id(1);
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
}

// Door ongelijke incremention van de index in de originele functie (wat uiteraard niet door opencl
// supported is) moest de index gedeeld worden door de incremention value, en later hier in de kernel
// weer vermenigvuldigd worden met de incremention value :). Vandaar dat i en j vermenigvuldigd worden
// met de pooling windows dimensies.

__kernel void max_pool(__global double* restrict image,
                       __global double* restrict output,
                       const int rows,
                       const int cols,
                       const int out_rows,
                       const int out_cols,
                       const int pl_rows,
                       const int pl_cols,
                       const double dbl_max,
                       const int size) {
    for (int q = 0; q < size; q++) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        double max = -dbl_max;
        for (int x = i * pl_rows; x < (i * pl_rows) + pl_rows; x++) {
            for (int y = j * pl_cols; y < (j * pl_cols) + pl_cols; y++) {
                double val_at_index = image[(rows * cols * q) + (x * cols + y)];
                max = val_at_index > max ? val_at_index : max;
            }
        }
        output[(out_cols * out_rows * q) + (i * out_cols + j)] = max;
    }
}