#include "include/Matrix.h"

Matrix3D::Matrix3D(int rows, int columns, int channels, std::vector<uint8_t> &data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->array = data;
}

void Matrix3D::set(int row, int column, int channel, uint8_t value) {
    this->array[coordinate_to_index3D(row, column, channel)] = value;
}

uint8_t Matrix3D::get(int row, int column, int channel) {
    return this->array[coordinate_to_index3D(row, column, channel)];
}

int Matrix3D::coordinate_to_index3D(int i, int j, int k) {
    return (i * this->columns * this->channels) + (j * this->channels) + k;
}

void Matrix3D::print() {
    for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
            for (int k = 0; k < channels; k++) {
			    std::cout << this->array[coordinate_to_index3D(i, j, k)] << " " ;
            }
            std::cout << " - ";
		}
		std::cout << '\n';
	}
    std::cout << std::endl;
}
