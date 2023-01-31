double relu(double x) {
    if (x > 0) {
        return x;
    }
    return 0.0;
}

double softmax(double x){
    if (isnan(x)) {
        return 0;
    }
    return exp(x);
}

__kernel void forward_pass(__global double* d_input, __global double* d_outputA, __global double* d_outputZ, __global double* weights, __global double* biases, int images, int z_rows, int layer) {
    for (int img = 0; img < images; img++) {
        int i = get_global_id(0);
        double value = 0.0;
        for (unsigned int k = 0; k < z_rows; ++k) {
            value += weights[i * z_rows + k] * d_input[img * z_rows + k];
        }

        value += biases[i];
        d_outputZ[img * z_rows + i] = value;

        if (layer == 0) {
            d_outputA[img * z_rows + i] = relu(value);
        }
        else if (layer == 1) {
            d_outputA[img * z_rows + i] = softmax(value);
        }
    }
}
