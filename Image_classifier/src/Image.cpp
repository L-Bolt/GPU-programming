#include "Image.h"


Image::Image(std::vector<uint8_t> &data) {
    this->classifier = data[0];
    data.erase(data.begin());
    this->data = data;
}

/**
 * @brief Displays an image.
 */
void Image::display_image() {
    cv::Mat src(32, 32, CV_8U, this->data.data());

    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::imshow("Display window", src);

    int k = cv::waitKey(0);
}

/**
 * @brief Saves an image to the disk.
 *
 * @param path Path to save the image at.
 */
void Image::save_image(std::string &path) {
    cv::Mat src(32, 32, CV_8U, this->data.data());
    cv::imwrite(path, src);
}

/**
 * @brief Returns the class the image belongs to.
 *
 * @return short 0-9 indicating the class.
 */
short Image::get_class() {
    return (short) this->classifier;
}
