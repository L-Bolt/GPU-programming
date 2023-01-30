double relu(double x) {
    if (x > 0) {
        return x;
    }
    return 0.0;
}

__kernel void forward_pass(__global double* restrict d_input, __global double* restrict d_outputA, __global double* restrict d_outputZ, __global double* weights, __global double* biases, int images, int z_rows, int z_columns, int layer) {
    for (int img = 0; img < images; img++) {
        int i = get_global_id(0);
        double value = 0.0;
        for (unsigned int k = 0; k < z_rows; ++k) {
            value += weights[i * z_rows + k] * d_input[img * z_rows + k];
        }

        value += biases[i];
        d_outputZ[0] = 0.1;

        printf("%d, %d\n", img, i);

        // if (layer == 0) {
        //     d_outputA[image * z_rows + i] = relu(value);
        // }
    }
}
