#include <OpenCL/OpenCL.hpp>

#include "include/Dataset.h"

// #include "include/Matrix.h"


int main() {

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../dataset/cifar-10-batches-bin");
    dataset->display_all_images();

}
