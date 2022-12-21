#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class Image {
    public:
        Image(std::vector<uint8_t> &data);
        ~Image() = default;

        void display_image(std::string window_name);
        void save_image(std::string &path);
        short get_class();

    private:
        std::vector<uint8_t> data;
        uint8_t classifier;

};
