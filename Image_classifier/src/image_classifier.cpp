#include "include/Dataset.h"
#include "include/Gpu.h"
#include "include/Cnn.h"


int main() {

	Shape3D input_dim={32, 32, 3};
	Shape kernel_dim={5, 5};
	Shape pool_size={2, 2};

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../dataset/cifar-10-batches-bin");
    std::unique_ptr<Gpu> gpu = std::make_unique<Gpu>(std::vector<std::string>{"../src/kernels/test.cl"});
    CNN cnn(input_dim, kernel_dim, pool_size, 30, 10);

    cnn.train(dataset->training_set, dataset->labels, 0.01, 3);

    if (gpu->gpu_enabled()) {
        gpu->test();
    }

    dataset->display_all_images();
}
