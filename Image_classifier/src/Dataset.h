#include <string>
#include <fstream>
#include <string.h>
#include <iostream>
#include <bitset>

class Dataset {
    public:
        Dataset(std::string path);
        ~Dataset();

        bool error = false;

        void dprint(std::string print) {
            std::cout << print << std::endl;
        }


    private:
        std::string path;
        size_t dataset_size;
        size_t size_per_image;
        uint8_t *buffer;

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

        uint8_t *make_buffer(std::string& path);
};