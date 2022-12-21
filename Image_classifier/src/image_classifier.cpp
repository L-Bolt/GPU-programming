#include "Dataset.h"


int main() {

    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("../data/cifar-10-batches-bin");
    dataset->display_all_images();

}
