#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"
#include "include/Matrix.hpp"
#include "include/Gui.h"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};
    Matrix3D<double> conv_kernel = Matrix3D<double>(kernel_dim.rows, kernel_dim.columns, 3, true);

    Gpu gpu(std::vector<std::string>{"../src/kernels/preprocess.cl", "../src/kernels/forward.cl"});
    Dataset dataset("../dataset/cifar-10-batches-bin", gpu, conv_kernel, pool_size);
    CNN cnn(input_dim, kernel_dim, pool_size, 196, 10, conv_kernel);
    // Gui gui("Image Classifier", &cnn, &dataset);

    // if (gui.is_enabled()) {
    //     gui.run();
    // }
    // else {
    //     std::cout << "Training model with 12 epochs..." << std::endl;
    //     cnn.train(*dataset.get_training_set(), *dataset.get_training_labels(), 0.001, 12);
    //     std::cout << "validating model..." << std::endl;
    //     cnn.validate(*dataset.get_test_set(), *dataset.get_test_labels());
    // }

    std::vector<double> a0(50000 * 196);
    std::vector<double> a1(50000 * 10);
    std::vector<double> z0(50000 * 196);
    std::vector<double> z1(50000 * 10);
    std::vector<std::vector<double>> a;
    std::vector<std::vector<double>> z;
    a.push_back(a0);
    a.push_back(a1);
    z.push_back(z0);
    z.push_back(z1);
    gpu.forward_prop(dataset.get_training_set(), &a, &z, &cnn.weights.at(0), &cnn.weights.at(1), &cnn.biases.at(0), &cnn.biases.at(1));
}
