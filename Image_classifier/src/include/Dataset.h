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
#include "Gpu.h"
#include "Matrix.hpp"

#define CIFAR_CLASSES 10
#define CIFAR_IMAGE_COUNT 60000
#define CIFAR_FILES 6
#define CIFAR_IMAGES_PER_FILE 10000
#define CIFAR_BIN_FILE_SIZE 3073 * 10000

class Dataset {
    public:
        Dataset() = default;
        Dataset(std::string path, Gpu &gpu, Matrix3D<double> &conv_kernel, Shape &pool_window);
        ~Dataset() = default;

        void display_all_images();
        void write_images_to_disk();
        Image &get_image(int index);

        std::vector<Image> *get_training_set() {return &training_set;};
        std::vector<Image> *get_test_set() {return &test_set;};

        std::vector<std::vector<double>> labels;
        std::vector<std::vector<double>> test_labels;
        std::vector<std::vector<uint8_t>>* get_buffer() {return &buffer;};

    private:
        std::vector<std::vector<uint8_t>> buffer;
        std::vector<Image> image_buffer;

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
        std::vector<Image> training_set;
        std::vector<Image> test_set;
        std::vector<Image> make_images(std::vector<std::vector<uint8_t>> &buffer, std::vector<Matrix2D<double>> &processed_buffer);
        Gpu gpu;
};

#endif
