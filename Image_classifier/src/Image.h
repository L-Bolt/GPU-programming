#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class Image {
    public:
        Image(std::vector<uint8_t> &data);
        ~Image() = default;

        static std::vector<std::vector<uint8_t>> split_vector(std::vector<uint8_t> &vec, size_t n);
        void display_image();
        short get_class();

    private:
        std::vector<uint8_t> data;
        uint8_t classifier;

};
