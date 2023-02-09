#include "include/Image.h"


/**
 * Constructor of the Image that takes a vector of uint_8 as input.
 * Create a matrix that contains the input data in a 3D coordinate system.
 */
Image::Image(std::vector<uint8_t> *data) {
    this->classifier = data->at(0);

    this->matrix = Matrix3D<uint8_t>(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_COLOR_CHANNELS, data);
}

/**
 * Constructor of the Image that takes both a vector of uint_8 as input, as well
 * as a matrix2D of doubles that are the result of the convolutional layer on the GPU.
 */
Image::Image(std::vector<uint8_t> *data, Matrix2D<double>* preprocessed_data) {
    this->classifier = data->at(0);

    this->matrix = Matrix3D<uint8_t>(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_COLOR_CHANNELS, data);
    this->preprocessed_data = preprocessed_data->flatten_to_vector(0);
    this->processed = true;
}

/**
 * @brief Displays an image.
 */
void Image::display_image(std::string window_name) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, array_to_cv_mat());

    cv::waitKey(0);
    cv::destroyWindow(window_name);
}

/**
 * @brief Converts the array to a cv::Mat.
 *
 * @return cv::Mat Converted array to cv::Mat.
 */
cv::Mat Image::array_to_cv_mat() {
    cv::Mat mat(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CV_8UC3);

    // Copy the data from the image buffer into the cv::mat matrix.
    for (int i = 0; i < CIFAR_IMAGE_SIZE; i++) {
        for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
            for (int k = 0; k < CIFAR_IMAGE_COLOR_CHANNELS; k++) {
                mat.at<cv::Vec3b>(i, j)[k] = this->matrix.get(i, j, k);
            }
        }
    }

    // Convert our matrix' RGB representation to the BGR representation needed
    // to display OpenCV matrices.
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

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
