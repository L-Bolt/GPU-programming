#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"
#include "include/Matrix.hpp"
#include "include/Gui.h"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};

    Dataset dataset("../dataset/cifar-10-batches-bin");
    Gpu gpu(std::vector<std::string>{"../src/kernels/test.cl"});
    CNN cnn(input_dim, kernel_dim, pool_size, 196, 10);
    Gui gui("Image Classifier", &cnn, &dataset);

    if (gui.is_enabled()) {
        gui.run();
    }
    else {
        cnn.train(*dataset.get_training_set(), dataset.labels, 0.001, 12);
        std::cout << "validating: " << std::endl;
        cnn.validate(*dataset.get_test_set(), dataset.test_labels);
    }

    return 0;
}
