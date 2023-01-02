#ifndef IMAGE
#define IMAGE

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/Matrix.hpp"

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

        Matrix3D<double> normalize() {return matrix.normalize();};
        int get_size() const {return CIFAR_IMAGE_SIZE;};

    private:
        Matrix3D<uint8_t> matrix;
        uint8_t classifier;

        cv::Mat array_to_cv_mat();

};

#endif
