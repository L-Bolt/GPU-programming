#include <string>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

#include "Image.h"


class Dataset {
    public:
        Dataset(std::string path);
        ~Dataset() = default;

        void display_all_images();

    private:
        std::string path;
        size_t dataset_size = 10000 * 3073;
        size_t size_per_image = 3073;
        std::vector<uint8_t> buffer;

        std::string files[5] = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"
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

        std::vector<uint8_t> make_buffer(std::string& path);
};
