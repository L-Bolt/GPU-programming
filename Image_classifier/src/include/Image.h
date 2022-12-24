#ifndef IMAGE
#define IMAGE

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/Matrix.h"

#define CIFAR_IMAGE_SIZE 32
#define CIFAR_IMAGE_COLOR_CHANNELS 3

class Image {
    public:
        Image(): classifier{255} {};
        Image(std::vector<uint8_t> *data);
        ~Image() = default;

        void display_image(std::string window_name);
        void save_image(std::string &path);
        short get_class();

    private:
        std::vector<uint8_t> *data;
        Matrix3D matrix;
        uint8_t classifier;

        cv::Mat array_to_cv_mat();

};

#endif
