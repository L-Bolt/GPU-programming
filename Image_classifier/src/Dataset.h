#ifndef DATASET
#define DATASET

#include <string>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <filesystem>

#include "util.h"
#include "Image.h"

#define CIFAR_CLASSES 10
#define CIFAR_IMAGE_COUNT 60000
#define CIFAR_FILES 6
#define CIFAR_IMAGES_PER_FILE 10000
#define CIFAR_BIN_FILE_SIZE 3073 * 10000

class Dataset {
    public:
        Dataset(std::string path);
        ~Dataset() = default;

        void display_all_images();
        void write_images_to_disk();
        std::vector<uint8_t> &get_image_data(int &index);

    private:
        std::vector<std::vector<uint8_t>> buffer;
        std::vector<std::vector<uint8_t>> training_set;
        std::vector<std::vector<uint8_t>> test_set;

        std::string files[CIFAR_FILES] = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
            "test_batch.bin"
        };

        std::string classes[CIFAR_CLASSES] {
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        };

        std::vector<std::vector<uint8_t>> make_buffer(std::string& path);
};

#endif
