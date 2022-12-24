#include <OpenCL/OpenCL.hpp>

#include "include/Dataset.h"
#include "include/Gpu.h"


int main() {

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../dataset/cifar-10-batches-bin");
    Gpu gpu("../src/kernels/test.cl");
    if (gpu.gpu_enabled()) {
        gpu.test();
    }
    dataset->display_all_images();

}
