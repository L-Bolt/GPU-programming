#include "include/Dataset.h"
#include "include/Gpu.h"


int main() {

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../dataset/cifar-10-batches-bin");
    std::unique_ptr<Gpu> gpu = std::make_unique<Gpu>(std::vector<std::string>{"../src/kernels/test.cl"});

    if (gpu->gpu_enabled()) {
        gpu->test();
    }

    Matrix2D<double>::test_matrix2D();

    dataset->display_all_images();
}
