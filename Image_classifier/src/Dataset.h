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

class Dataset {
    public:
        Dataset(std::string path);
        ~Dataset() = default;

        void display_all_images();
        void write_images_to_disk();
        std::vector<uint8_t> &get_image_data(int &index);

    private:
        size_t dataset_size = 10000 * 3073;
        size_t size_per_image = 3073;
        std::vector<std::vector<uint8_t>> buffer;
        std::vector<std::vector<uint8_t>> training_set;
        std::vector<std::vector<uint8_t>> test_set;

        std::string files[6] = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
            "test_batch.bin"
        };

        std::string classes[10] {
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
