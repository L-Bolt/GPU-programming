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
std::vector<std::vector<uint8_t>> Dataset::make_buffer(std::string& path) {
    // Allocate buffer vector
    std::vector<uint8_t> buffer(5 * this->dataset_size);

    // Read all 5 binary files containing the dataset.
    for (int i = 0; i < 5; i++) {
        size_t offset = i * this->dataset_size;
        std::string binary_file_path = this->path + '/' + this->files[i];

        std::ifstream file(binary_file_path, std::ios::binary);
        file.read(reinterpret_cast<char*>(buffer.data() + offset), this->dataset_size);
    }

    return split_vector(buffer, 5 * 10000);
}

/**
 * Creates images out of all the data in the buffer and calls the function to
 * display the image.
 */
void Dataset::display_all_images() {
    for (std::vector<uint8_t> img_data : this->buffer) {
        size_t img_data_size = img_data.size();
        Image img(img_data);
        std::cout << this->classes[img.get_class()] << std::endl;
        img.display_image();
    }
}

/**
 * @brief Debug function that writes all the images in the dataset to the disk.
 */
void Dataset::write_images_to_disk() {
    for (int i = 0; i < 5; i++) {
        std::string folder = std::to_string(i + 1);
        if (std::filesystem::is_directory(folder)) {
            std::filesystem::remove_all(folder);
        }

        std::filesystem::create_directory(folder);
        for (int j = 0; j < 10000; j++) {
            std::string image_name = std::to_string(j);
            std::string path = folder + '/' + image_name + ".jpg";

            std::vector<uint8_t> img_data = this->buffer[i * 10000 + j];
            Image img(img_data);
            img.save_image(path);
        }
    }
}

/**
 * @brief Split a vector in n equal parts.
 *
 * @param vec Vector to split
 * @param n Split the vector in n parts.
 * @return std::vector<std::vector<uint8_t>> Vector containing all the split vectors.
 */
std::vector<std::vector<uint8_t>> Dataset::split_vector(std::vector<uint8_t> &vec, size_t n) {
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
