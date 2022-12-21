#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class Image {
    public:
        Image(std::vector<uint8_t> &data);

        void display_image();
        void save_image(std::string &path);
        short get_class();

    private:
        std::vector<uint8_t> data;
        uint8_t classifier;

};