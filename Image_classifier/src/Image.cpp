#include "Image.h"


Image::Image(std::vector<uint8_t> &data) {
    this->classifier = data[0];
    data.erase(data.begin());
    this->data = data;
}

/**
 * @brief Displays an image.
 */
void Image::display_image(std::string window_name) {
    cv::Mat mat(32, 32, CV_8UC3);
    int pixel = 0;

    // Copy the data from the image buffer into the cv::mat matrix.
    // TODO dit is kaulo aids, miss kan t met iterators. maar cv mat is sws ait aids.
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 3; k++) {
                mat.at<cv::Vec3b>(i, j)[k] = this->data[(2048 - k * 1024) + pixel];
            }
        pixel++;
        }
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, mat);

    int k = cv::waitKey(0);
    cv::destroyWindow(window_name);
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
