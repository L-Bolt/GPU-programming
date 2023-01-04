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
        Image &get_image(int index);

        std::vector<Image> training_set;
        std::vector<std::vector<double>> labels;

    private:
        std::vector<std::vector<uint8_t>> buffer;
        std::vector<Image> image_buffer;
        std::vector<Image> test_set;

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
        std::vector<Image> make_images(std::vector<std::vector<uint8_t>> &buffer);
};

#endif
