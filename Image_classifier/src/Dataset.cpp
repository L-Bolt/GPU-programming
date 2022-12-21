#include "Dataset.h"

/**
 * Constructor for the dataset class.
 * Calls the make_buffer function to store the dataset in memory
 */
Dataset::Dataset(std::string path) {
    this->buffer = make_buffer(path);

    this->training_set = std::vector(this->buffer.begin(), this->buffer.end() - CIFAR_IMAGES_PER_FILE);
    this->test_set = std::vector(this->buffer.end() - CIFAR_IMAGES_PER_FILE, this->buffer.end());
}

/**
 * Creates a buffer with enough space for all the images in the entire
 * dataset.
 */
std::vector<std::vector<uint8_t>> Dataset::make_buffer(std::string& path) {
    // Allocate buffer vector
    std::vector<uint8_t> buffer(CIFAR_FILES * CIFAR_BIN_FILE_SIZE);

    // Read all 6 binary files containing the dataset.
    for (int i = 0; i < CIFAR_FILES; i++) {
        size_t offset = i * CIFAR_BIN_FILE_SIZE;
        std::string binary_file_path = path + '/' + this->files[i];

        std::ifstream file(binary_file_path, std::ios::binary);
        if (!file) {
            std::cout << "Error opening file: " << path << std::endl;
            exit(1);
        }

        file.read(reinterpret_cast<char*>(buffer.data() + offset), CIFAR_BIN_FILE_SIZE);
    }

    return util::split_vector(buffer, CIFAR_IMAGE_COUNT);
}

/**
 * @brief Returns a reference to the data of one image in the buffer.
 *
 * @param index The index of the image in the buffer.
 * @return std::vector<uint8_t>& Reference to the data of the image.
 */
std::vector<uint8_t> &Dataset::get_image_data(int &index) {
    return this->buffer[index];
}

/**
 * Creates images out of all the data in the buffer and calls the function to
 * display the image.
 */
void Dataset::display_all_images() {
    for (std::vector<uint8_t> img_data : this->buffer) {
        Image img(img_data);
        std::cout << this->classes[img.get_class()] << std::endl;
        img.display_image(this->classes[img.get_class()]);
    }
}

/**
 * @brief Debug function that writes all the images in the dataset to the disk.
 */
void Dataset::write_images_to_disk() {
    for (int i = 0; i < CIFAR_FILES; i++) {
        std::string folder = std::to_string(i + 1);
        if (std::filesystem::is_directory(folder)) {
            std::filesystem::remove_all(folder);
        }

        std::filesystem::create_directory(folder);
        for (int j = 0; j < CIFAR_IMAGES_PER_FILE; j++) {
            std::string image_name = std::to_string(j);
            std::string path = folder + '/' + image_name + ".jpg";

            std::vector<uint8_t> img_data = this->buffer[i * CIFAR_IMAGES_PER_FILE + j];
            Image img(img_data);
            img.save_image(path);
        }
    }
}
