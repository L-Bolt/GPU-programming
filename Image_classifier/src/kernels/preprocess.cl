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

void dot_mult(__global double* restrict mat1,
              __global double* restrict mat2,
              __global double* restrict res,
              const int mat1_rows,
              const int mat2_cols,
              const int mat1_cols) {
    for (int i = 0; i < mat1_rows; ++i) {
        for (int j = 0; j < mat2_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < mat1_cols; ++k) {
                sum += mat1[i * mat1_cols + k] * mat2[k * mat2_cols + j];
            }
            res[i * mat2_cols + j] = sum;
        }
    }
    return;
}

void vec_dot(__global double* restrict mat1,
             __global double* restrict mat2,
             __global double* restrict res,
             const int mat1_rows,
             const int mat1_cols) {
    for (int i = 0; i < mat1_rows; i++) {
        double w = 0.0;
        for (int j = 0; j < mat1_cols; j++) {
            w += (mat1[i * mat1_cols + j] * mat2[j]);
        }
        res[i] = w;
    }
    return;
}

void sub_func(__global double* restrict v1,
              __global double* restrict v2,
              __global double* restrict res,
              const int v1_size) {
    for (int i = 0; i < v1_size; i++) {
        res[i] = v1[i] - v2[i];
    }
    return;
}

double rel_grad(double x) {
    if (x > 0) {
        return (double) 1;
    }
    return (double) 0.2;
}

// WARNING! As implied by the first variable name the first vector will be used for the result,
// meaning it's contents get overwritten with the results of the multiplication.
void mult_func(__global double* restrict res_v1,
               __global double* restrict v2,
               const int v1_size) {
    for (int i = 0; i < v1_size; i++) {
        res_v1[i] = res_v1[i] * v2[i];
    }
    return;
}

void transpose(__global double* restrict mat,
               __global double* restrict res,
               const int rows,
               const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res[j * rows + i] = mat[i * cols + j];
        }
    }
    return;
}

double cross_entropy(__global double* restrict ypred, 
                     __global double* restrict ytrue, 
                     __global double* restrict z,
                     const int ypred_size) {
    for (int i = 0; i < ypred_size; i++) {
        z[i] = log(ypred[i]);
    }
    mult_func(z, ytrue, ypred_size);
    double error = 0.0;
    for (int j = 0; j < ypred_size; j++) {
        error += z[j];
    }

    return (-error);
}

__kernel void forward_pass(__global double* restrict biases0,
                           __global double* restrict Z1,
                           __global double* restrict A1,
                           const int bias0_cols,
                           const int Z1_size,
                           __global double* restrict dZ2,
                           __global double* restrict ytrain,
                           __global double* restrict temp2,
                           const int Z2_size,
                           __global double* restrict A2,
                           __global double* restrict dW2,
                           const int output_dim,
                           const int hid_nodes,
                           const int learning_rate,
                           __global double* restrict weights1,
                           const int weights1_rows,
                           const int weights1_cols,
                           __global double* restrict dZ1,
                           __global double* restrict dz_deriv,
                           __global double* restrict Z2,
                           __global double* restrict X_mat,
                           __global double* restrict dW1,
                           const int X_size,
                           __global double* restrict dW1_trans,
                           __global double* restrict weights0,
                           const int weights0_cols,
                           __global double* restrict biases1,
                           const int bias1_rows,
                           const int bias1_cols,
                           __global double* restrict dW2_trans,
                           const int batch_size,
                           const int weights0_rows,
                           __global double* restrict weights0_trans,
                           __global double* restrict temp5,
                           __global double* restrict weights1_trans,
                           const int ytrain_size,
                           const int bias0_rows,
                           double error,
                           __global double* restrict z) {
    for (int i = 0; i < batch_size; i++) {
        printf("%d\n", i);
        transpose(weights0, weights0_trans, weights0_rows, weights0_cols);
        for (int a = 0; a < X_size; a++) {
            temp5[a] = X_mat[a + (i * X_size)];
        }
        vec_dot(weights0_trans, temp5, Z1, weights0_cols, weights0_rows);
        for (int j = 0; j < Z1_size; j++) {
            Z1[j] = Z1[j] + biases0[j * bias0_cols];
            A1[j] = relu(Z1[j]);
        }
        transpose(weights1, weights1_trans, weights1_rows, weights1_cols);
        vec_dot(weights1_trans, A1, Z2, weights1_cols, weights1_rows);
        for (int j2 = 0; j2 < Z2_size; j2++) {
            Z2[j2] = Z2[j2] + biases1[j2 * bias1_cols];
            A2[j2] = softmax(Z2[j2]);
            A2[j2] = isnan(A2[j2]) ? 1.0 : A2[j2];
        }
        for (int l = 0; l < ytrain_size; l++) {
            temp2[l] = ytrain[l + (i * ytrain_size)];
        }
        error += cross_entropy(A2, temp2, z, Z2_size);
        sub_func(A2, temp2, dZ2, Z2_size);
        dot_mult(dZ2, A1, dW2, output_dim, hid_nodes, 1);
        double db2 = 0.0;
        for (int n = 0; n < Z2_size; n++) {
            db2 += dZ2[n];
        }
        db2 *= (1.0 / 10.0);
        db2 *= learning_rate;
        vec_dot(weights1, dZ2, dZ1, weights1_rows, weights1_cols);
        for (int o = 0; o < Z1_size; o++) {
            dz_deriv[o] = rel_grad(Z1[o]);
        }
        mult_func(dZ1, dz_deriv, weights1_rows);
        dot_mult(dZ1, temp5, dW1, weights1_rows, X_size, 1);
        for (int p = 0; p < weights1_rows; p++) {
            for (int q = 0; q < X_size; q++) {
                dW1[p * X_size + q] = dW1[p * X_size + q] * (1.0 / 10.0);
            }
        }
        double db1 = 0.0;
        for (int r = 0; r < weights1_rows; r++) {
            db1 += dZ1[r];
        }
        db1 *= (1.0 / 10.0);
        db1 *= learning_rate;
        transpose(dW1, dW1_trans, weights1_rows, X_size);
        for (int s = 0; s < X_size; s++) {
            for (int t = 0; t < weights1_rows; t++) {
                dW1_trans[s * weights1_rows + t] = dW1_trans[s * weights1_rows + t] * learning_rate;
                weights0[s * weights0_cols + t] = weights0[s * weights0_cols + t] - dW1_trans[s * weights1_rows + t];
            }
        }
        for (int u = 0; u < bias0_rows; u++) {
            for (int v = 0; v < bias0_cols; v++) {
                biases0[u * bias0_cols + v] = biases0[u * bias0_cols + v] * db1;
            }
        }
        transpose(dW2, dW2_trans, output_dim, hid_nodes);
        for (int w = 0; w < hid_nodes; w++) {
            for (int x = 0; x < output_dim; x++) {
                dW2_trans[w * output_dim + x] = dW2_trans[w * output_dim + x] * learning_rate;
                weights1[w * weights1_cols + x] = weights1[w * weights1_cols + x] - dW2_trans[w * output_dim + x];
            }
        }
        for (int y = 0; y < bias1_rows; y++) {
            for (int z = 0; z < bias1_cols; z++) {
                biases1[y * bias1_cols + z] = biases1[y * bias1_cols + z] * db2;
            }
        }
    }
}
