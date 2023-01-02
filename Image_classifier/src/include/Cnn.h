#ifndef CNN_H
#define CNN_H

#include "Matrix.hpp"
#include "Image.h"


class CNN {
    public:
        CNN(Shape3D input_dim, Shape kernel_size, Shape pool_size, int hidden_layer_nodes, int output_dim);
        ~CNN() = default;

        void train(std::vector<Image> &Xtrain,
                   std::vector<std::vector<double>> &Ytrain,
                   double learning_rate, int epochs);
        double validate(std::vector<Image> &Xval, std::vector<std::vector<double>> &Yval);

    private:
        void forward_propagate(Image &input, std::vector<Matrix2D<double>> &conv_activations, std::vector<std::vector<double>> &activations);
        void back_propagate(std::vector<double> &delta_L,
                            std::vector<Matrix2D<double>> &conv_activations,
                            std::vector<std::vector<double>> &activations,
                            Image &input, double (*active_fn_der)(double), double learning_rate);

        double cross_entropy(std::vector<double> &ypred, std::vector<double> &ytrue);

        std::vector<Matrix2D<double>> weights;
        Matrix3D<double> kernel;
        Shape pool_window;
};

#endif
