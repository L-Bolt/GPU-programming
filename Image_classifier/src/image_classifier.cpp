
#include <iostream>
#include <memory>

#include "Dataset.h"

int main() {
    //std::cout << "helo team" << std::endl;


    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("/home/lb/Desktop/GPU-programming/Image_classifier/data/cifar-10-batches-bin");
    if (dataset->error) {
        std::cout << "dataset error" << std::endl;
        dataset->~Dataset();
        exit(1);
    }

}
