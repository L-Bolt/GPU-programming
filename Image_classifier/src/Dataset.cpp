#include "Dataset.h"

/**
 * Constructor for the dataset class.
 * Calls the make_buffer function to store the dataset in memory
 */
Dataset::Dataset(std::string path) {
    this->path = path;

    this->buffer = make_buffer(this->path);
}

/**
 * Creates a buffer with enough space for all the images in the entire
 * dataset.
 */
std::vector<uint8_t> Dataset::make_buffer(std::string& path) {
    // Allocate buffer vector
    std::vector<uint8_t> buffer(5 * this->dataset_size);

    // Read all 5 binary files containing the dataset.
    for (int i = 0; i < 5; i++) {
        size_t offset = i * this->dataset_size;
        std::string binary_file_path = this->path + '/' + this->files[i];

        std::ifstream file(binary_file_path, std::ios::binary);
        file.read(reinterpret_cast<char*>(buffer.data() + offset), this->dataset_size);
    }

    return buffer;
}

/**
 * Creates images out of all the data in the buffer and calls the function to
 * display the image.
 */
void Dataset::display_all_images() {
    std::vector<std::vector<uint8_t>> splitted = Image::split_vector(this->buffer, 5 * 10000);
    size_t splitted_size = splitted.size();

    for (std::vector<uint8_t> img_data : splitted) {
        size_t img_data_size = img_data.size();
        Image img(img_data);
        std::cout << this->classes[img.get_class()] << std::endl;
        img.display_image();
    }
}
