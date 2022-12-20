
#include <iostream>

#include "Dataset.h"

int main() {

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("/home/lb/Desktop/GPU-programming/Image_classifier/data/cifar-10-batches-bin");
    dataset->display_all_images();


}
