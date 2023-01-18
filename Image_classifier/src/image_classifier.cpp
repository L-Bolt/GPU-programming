#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"
#include "include/Matrix.hpp"
#include "include/Gui.h"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};
    Matrix3D<double> kernel = Matrix3D<double>(kernel_dim.rows, kernel_dim.columns, 3, true);

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../dataset/cifar-10-batches-bin");
    std::unique_ptr<Gpu> gpu = std::make_unique<Gpu>(std::vector<std::string>{"../src/kernels/test.cl"});
    CNN cnn(input_dim, kernel_dim, pool_size, 196, 10);
    Gui gui("Image Classifier");

    if (gui.is_enabled()) {
        gui.run();
    }


    // cnn.train(dataset->training_set, dataset->labels, 0.001, 12);
    // std::cout << "validating: " << std::endl;
    // cnn.validate(dataset->test_set, dataset->test_labels);

    // dataset->display_all_images();

    return 0;
}
