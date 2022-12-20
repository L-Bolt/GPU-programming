#include "Dataset.h"

/**
 * Constructor for the dataset class.
 * Calls the make_buffer function to store the dataset in memory
 */
Dataset::Dataset(std::string path) {
    this->path = path;
    this->dataset_size = 10000 * 3073;
    this->size_per_image = 3073;

    this->buffer = make_buffer(this->path);
}

/**
 * Destructor for the dataset.
 * Automatically frees the buffer when the dataset goes out of scope or when
 * the program terminates.
 */
Dataset::~Dataset() {
    free(this->buffer);
    std::cout << "freed dataset buffer" << std::endl;
}

/**
 * Creates a large buffer with enough space for all the images in the entire
 * dataset.
 */
uint8_t *Dataset::make_buffer(std::string& path) {
    // Allocate the arena in which the data of all images will be stored.
    // Buffer needs to be initialized to a char* because the ifstream.read()
    // function does not overload for unsigned chars (1 byte unsigned integers).
    char* buffer = (char*) malloc(5 * this->dataset_size);
    memset(buffer, 5 * this->dataset_size, 0);

    if (!buffer) {
        this->error = true;
        return NULL;
    }

    // Read all 5 binary files containing the dataset.
    for (int i = 0; i < 5; i++) {
        size_t offset = i * dataset_size;

        std::string binary_file_path = this->path + '/' + this->files[i];
        std::ifstream input(binary_file_path, std::ios::binary);

        if (!input) {
            this->error = true;
            return NULL;
        }

        // Copy the contents from the file to the buffer at offset.
        input.read(&buffer[offset], this->dataset_size);
    }

    return (uint8_t*) buffer;
}
