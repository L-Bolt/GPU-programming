#include "include/Dataset.h"


/**
 * Constructor for the dataset class.
 * Calls the make_buffer function to store the dataset in memory
 */
Dataset::Dataset(std::string path, Gpu &gpu, Matrix3D<double> &conv_kernel, Shape &pool_window) {
    try {
        this->buffer = make_buffer(path);
        this->gpu = gpu;

        if (this->gpu.gpu_enabled()) {
            std::cout << "Preprocessing images on GPU..." << std::endl;
            std::vector<Matrix2D<double>> processed_images = this->gpu.preprocess(&this->buffer, conv_kernel, 32, 32, 3, pool_window);
            this->image_buffer = make_images(this->buffer, processed_images);
        }
        else {
            this->image_buffer = make_images(this->buffer);
        }

        std::cout << "\033[1;32mDataset has been read into memory.\033[0m\n" <<std::endl;

        this->training_set = std::vector<Image>(this->image_buffer.begin(), this->image_buffer.end() - CIFAR_IMAGES_PER_FILE);
        this->test_set = std::vector<Image>(this->image_buffer.end() - CIFAR_IMAGES_PER_FILE, this->image_buffer.end());

        std::cout << "Images in training set: " << this->training_set.size() << '\n';
        std::cout << "Images in test set: " << this->test_set.size() << '\n' << std::endl;
    }
    catch (const std::string &msg) {
        std::cout << msg << std::endl;
        std::cout << "Have you tried downloading the dataset by running \033[1;36m./download_dataset.sh\033[0m?" << std::endl;
        exit(1);
    }
    catch(...) {
        std::cout << "\033[1;31mCould not allocate memory for image buffer\033[0m" << std::endl;
        exit(1);
    }

    for (Image &img : this->training_set) {
        std::vector<double> label(CIFAR_CLASSES, 0);
        label[img.get_class()] = 1.0;
        this->labels.push_back(label);
    }

    for (Image &img : this->test_set) {
        std::vector<double> test_label(CIFAR_CLASSES, 0);
        test_label[img.get_class()] = 1.0;
        this->test_labels.push_back(test_label);
    }
}

/**
 * Creates a buffer with enough space for all the images in the entire
 * dataset.
 */
std::vector<std::vector<uint8_t>> Dataset::make_buffer(std::string& path) {
    // Allocate buffer vector
    std::cout << "Allocating memory for dataset buffer..." << std::endl;
    std::vector<uint8_t> buffer(CIFAR_FILES * CIFAR_BIN_FILE_SIZE);
    printf("\033[1;32mSuccesfully allocated %ld bytes for image buffer.\033[0m\n\n", sizeof(std::vector<uint8_t>) + (sizeof(uint8_t) * buffer.size()));

    // Read all 6 binary files containing the dataset.
    for (int i = 0; i < CIFAR_FILES; i++) {
        size_t offset = i * CIFAR_BIN_FILE_SIZE;
        std::string binary_file_path = path + '/' + this->files[i];

        std::cout << "Reading dataset file: " << binary_file_path << " ..." << std::endl;
        std::ifstream file(binary_file_path, std::ios::binary);
        if (!file) {
            throw("\033[1;31mError opening file: " + path + "\033[0m");
        }

        file.read(reinterpret_cast<char*>(buffer.data() + offset), CIFAR_BIN_FILE_SIZE);
        file.close();
    }

    return util::split_vector(buffer, CIFAR_IMAGE_COUNT);
}

/**
 * Create an Image object for every block of image data in the dataset buffer.
 * Store the Image object in the image_buffer and return the image_buffer.
 */
std::vector<Image> Dataset::make_images(std::vector<std::vector<uint8_t>> &buffer) {
    std::vector<Image> image_buffer(CIFAR_IMAGE_COUNT);
    for (int i = 0; i < CIFAR_IMAGE_COUNT; i++) {
        image_buffer[i] = Image(&buffer[i]);
    }

    return image_buffer;
}

std::vector<Image> Dataset::make_images(std::vector<std::vector<uint8_t>> &buffer, std::vector<Matrix2D<double>> &processed_buffer) {
    std::vector<Image> image_buffer(CIFAR_IMAGE_COUNT);
    for (int i = 0; i < CIFAR_IMAGE_COUNT; i++) {
        image_buffer[i] = Image(&buffer[i], &processed_buffer[i]);
    }

    return image_buffer;
}

/**
 * @brief Returns a reference to the data of one image in the buffer.
 *
 * @param index The index of the image in the buffer.
 * @return std::vector<uint8_t>& Reference to the data of the image.
 */
Image &Dataset::get_image(int index) {
    return this->image_buffer.at(index);
}

/**
 * Creates images out of all the data in the buffer and calls the function to
 * display the image.
 */
void Dataset::display_all_images() {
    for (Image img : this->image_buffer) {
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

            Image img = get_image(i * CIFAR_IMAGES_PER_FILE + j);
            std::cout << "saving image: " << (j + (CIFAR_IMAGES_PER_FILE * (i)) + 1) <<std::endl;
            img.save_image(path);
        }
    }
}
