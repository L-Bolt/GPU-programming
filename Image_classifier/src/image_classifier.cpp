#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"
#include "include/Matrix.hpp"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};
    // Matrix3D<double> conv_kernel = Matrix3D<double>(kernel_dim.rows, kernel_dim.columns, 3, true);

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../dataset/cifar-10-batches-bin");
    std::unique_ptr<Gpu> gpu = std::make_unique<Gpu>(std::vector<std::string>{"../src/kernels/test.cl"});
    CNN cnn(input_dim, kernel_dim, pool_size, 196, 10);
    std::cout << "\n\n\n\n" << std::endl;
    cnn.train(dataset->training_set, dataset->labels, 0.001, 12);
    std::cout << "validating: " << std::endl;
    cnn.validate(dataset->test_set, dataset->test_labels);

    // std::vector<Matrix2D<double>> images = gpu->preprocess(dataset->get_buffer(), conv_kernel, 1, input_dim.rows, input_dim.columns, input_dim.channels);
    // images[5].print(true);

    dataset->display_all_images();
}
