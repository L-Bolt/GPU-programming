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

short Image::get_class() {
    return (short) this->classifier;
}

/**
 * @brief Split a vector in n equal parts.
 *
 * @param vec Vector to split
 * @param n Split the vector in n parts.
 * @return std::vector<std::vector<uint8_t>> Vector containing all the split vectors.
 */
std::vector<std::vector<uint8_t>> Image::split_vector(std::vector<uint8_t> &vec, size_t n) {
    std::vector<std::vector<uint8_t>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i) {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<uint8_t>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
}
