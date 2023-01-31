#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"
#include "include/Matrix.hpp"
#include "include/Gui.h"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};
    Matrix3D<double> conv_kernel = Matrix3D<double>(kernel_dim.rows, kernel_dim.columns, input_dim.channels, true);

    Gpu gpu(std::vector<std::string>{"../src/kernels/preprocess.cl", "../src/kernels/forward.cl"});
    CNN cnn(input_dim, kernel_dim, pool_size, 196, 10, conv_kernel, gpu);
    Dataset dataset("../dataset/cifar-10-batches-bin", gpu, conv_kernel, pool_size);
    Gui gui("Image Classifier", &cnn, &dataset);

    if (gui.is_enabled()) {
        gui.run();
    }
    else {
        std::cout << "Training model with 12 epochs..." << std::endl;
        cnn.train(*dataset.get_training_set(), *dataset.get_training_labels(), 0.001, 12);
        std::cout << "validating model..." << std::endl;
        cnn.validate(*dataset.get_test_set(), *dataset.get_test_labels());
    }

    // std::vector<std::vector<double>> a0;
    // std::vector<std::vector<double>> a1;
    // std::vector<std::vector<double>> z1;
    // gpu.forward_prop(dataset.get_training_set(), &a0, &a1, &z1, &cnn.weights.at(0), &cnn.weights.at(1), &cnn.biases.at(0), &cnn.biases.at(1));
}
