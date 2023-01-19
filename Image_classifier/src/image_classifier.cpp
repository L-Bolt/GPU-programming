#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"
#include "include/Matrix.hpp"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};
    Matrix3D<double> conv_kernel = Matrix3D<double>(kernel_dim.rows, kernel_dim.columns, 3, true);

    Gpu gpu(std::vector<std::string>{"../src/kernels/test.cl"});
    Dataset dataset("../dataset/cifar-10-batches-bin", gpu, conv_kernel, pool_size);
    CNN cnn(input_dim, kernel_dim, pool_size, 196, 10, conv_kernel);

    cnn.train(dataset.training_set, dataset.labels, 0.001, 12);
    std::cout << "validating: " << std::endl;
    cnn.validate(dataset.test_set, dataset.test_labels);

    dataset.display_all_images();
}
