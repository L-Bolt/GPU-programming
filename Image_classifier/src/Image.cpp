#include "Image.h"


Image::Image(std::vector<uint8_t> &data) {
    this->classifier = data[0];
    data.erase(data.begin());
    this->data = data;
}

/**
 * @brief Displats an image.
 *
 */
void Image::display_image() {
    cv::Mat src(32, 32, CV_8U, this->data.data());

    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::imshow("Display window", src);

    int k = cv::waitKey(0);
}

void Image::save_image(std::string &path) {
    cv::Mat src(32, 32, CV_8U, this->data.data());
    cv::imwrite(path, src);
}

short Image::get_class() {
    return (short) this->classifier;
}
