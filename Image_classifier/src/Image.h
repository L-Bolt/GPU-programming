#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class Image {
    public:
        uint8_t classifier;
        Image(std::vector<uint8_t> &data);

        static std::vector<std::vector<uint8_t>> split_vector(std::vector<uint8_t> &vec, size_t n);
        void display_image();

    private:
        std::vector<uint8_t> data;

};