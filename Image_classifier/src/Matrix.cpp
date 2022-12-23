#include "include/Matrix.h"

// Matrix3D::Matrix3D(): array(data) {};

Matrix3D::Matrix3D(int rows, int columns, int channels, std::vector<uint8_t> *data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->array = data;
}

void Matrix3D::set(int row, int column, int channel, uint8_t value) {
    this->array->at(coordinate_to_index3D(row, column, channel)) = value;
}

uint8_t Matrix3D::get(int row, int column, int channel) {
    return this->array->at(coordinate_to_index3D(row, column, channel));
}

int Matrix3D::coordinate_to_index3D(int i, int j, int k) {
    return (i * this->columns * this->channels) + (j * this->channels) + k;
}

std::vector<int> Matrix3D::index_to_coordinate3D(int index) {
    std::vector<int> coordinates{-1, -1, -1};

    if (index < 0 || index > (this->rows * this->columns * this->channels) - 1) {
        return coordinates;
    }

    int pixels_per_channel = this->rows * this->columns;
    int channel = index % pixels_per_channel;
    int row = (index - channel * pixels_per_channel) % this->rows;
    int column = (index - row * this->columns);

    coordinates[0] = row;
    coordinates[1] = column;
    coordinates[2] = channel;

    return coordinates;
}

void Matrix3D::print() {
    for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
            for (int k = 0; k < channels; k++) {
			    std::cout << (int) this->array->at(coordinate_to_index3D(i, j, k)) << " " ;
            }
            std::cout << "-|- ";
		}
		std::cout << "\n\n";
	}
    std::cout << std::endl;
}
