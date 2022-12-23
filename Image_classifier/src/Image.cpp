#include "include/Image.h"


Image::Image(std::vector<uint8_t> *data) {
    this->classifier = data->at(0);
    data->erase(data->begin());
    this->data = data;

    this->matrix = Matrix3D(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_COLOR_CHANNELS, data);
}

/**
 * @brief Displays an image.
 */
void Image::display_image(std::string window_name) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, array_to_cv_mat());

    int k = cv::waitKey(0);
    cv::destroyWindow(window_name);
}

/**
 * @brief Converts the array to a cv::Mat.
 *
 * @return cv::Mat Converted array to cv::Mat.
 */
cv::Mat Image::array_to_cv_mat() {
    cv::Mat mat(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CV_8UC3);
    int pixel = 0;

    // Copy the data from the image buffer into the cv::mat matrix.
    // TODO dit is kaulo aids, miss kan t met iterators. maar cv mat is sws aids.
    // Ook fking inefficient omdat t alles kopieert. Maar dat moet vgm omdat een cv::mat anders random data heeft.
    for (int i = 0; i < CIFAR_IMAGE_SIZE; i++) {
        for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
            for (int k = 0; k < 3; k++) {
                mat.at<cv::Vec3b>(i, j)[k] = this->data->at((2048 - k * 1024) + pixel); //TODO maak een algemene functie die een i,j,k 3d coordinaat kan omzetten naar een 1d coordinaat voor de cifar dataset.
            }
            pixel++;
        }
    }

    return mat;
}

/**
 * @brief Saves an image to the disk.
 *
 * @param path Path to save the image at.
 */
void Image::save_image(std::string &path) {
    cv::Mat src = array_to_cv_mat();
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
