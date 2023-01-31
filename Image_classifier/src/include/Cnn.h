#ifndef CNN_H
#define CNN_H

#include "Matrix.hpp"
#include "Image.h"
#include "Gpu.h"

#include <atomic>


class CNN {
    public:
        CNN(Shape3D input_dim, Shape kernel_size, Shape pool_size, int hidden_layer_nodes, int output_dim, Matrix3D<double> &conv_kernel, Gpu &gpu);
        ~CNN() = default;

        void train(std::vector<Image> &Xtrain,
                   std::vector<std::vector<double>> &Ytrain,
                   double learning_rate, int epochs);
        double validate(std::vector<Image> &Xval, std::vector<std::vector<double>> &Yval);
        bool is_trained() const {return trained;};
        bool is_validated() const {return validated;};
        int images_correct() {return correctly_classified;};
        void quit() {stop = true;};
        float get_training_percentage();
        int classify (Image &input);

    private:
        void forward_propagate(Image &input, std::vector<std::vector<double>> &a, std::vector<std::vector<double>> &z);
        void back_propagate(std::vector<double> &dZ2,
                                std::vector<double> &a,
                                std::vector<double> &z,
                                Image &input, double (*active_fn_der)(double), double learning_rate);

        double cross_entropy(std::vector<double> &ypred, std::vector<double> &ytrue);

        std::vector<Matrix2D<double>> weights;
        std::vector<Matrix2D<double>> biases;
        Matrix3D<double> kernel;
        Shape pool_window;
        std::atomic<float> iteration = 0.0;
        std::atomic<float> epoch = 0.0;
        std::atomic<int> correctly_classified = 0;
        int epochs = 0;
        int training_size = 0;
        int output_dim;
        int hidden_layer_nodes;
        bool trained = false;
        bool validated = false;
        bool stop = false;
        Gpu gpu;
};

#endif
