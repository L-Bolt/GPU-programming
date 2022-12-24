#include "include/Matrix.h"

Matrix3D::Matrix3D(int rows, int columns, int channels, std::vector<uint8_t> *data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->array = data;
}

Matrix3D::Matrix3D(int rows, int columns, int channels) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;

    //TODO is dit optimaal?
    this->array = new std::vector<uint8_t>((rows * columns * channels) + 1);
}

/**
 * Sets the value at the coordinate (row, column, channel) to the given value.
 */
void Matrix3D::set(int row, int column, int channel, uint8_t value) {
    this->array->at(coordinate_to_index3D(row, column, channel)) = value;
}

/**
 * Returns the value at (row, column, channel).
 */
uint8_t Matrix3D::get(int row, int column, int channel) {
    return this->array->at(coordinate_to_index3D(row, column, channel));
}

/**
 * Transforms the coordinate (i, j, k) to the 1D coordinate used in the array.
 */
int Matrix3D::coordinate_to_index3D(int i, int j, int k) {
    return 1 + ((k * this->rows * this->columns) + (i * this->rows) + j);
}

//TODO: Check of deze functie ook echt werkt. Wss hebben we m niet nodig tho
// dus check pas wanneer je m gaat gebruiken want is wss fout.
/**
 * Transforms a 1D array index to a 3D coordinate.
*/
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

/**
 * Prints the matrix.
 */
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

//TODO kan maybe beter met een transform functie
/**
 * Adds two 3D matrices together.
 */
Matrix3D operator+(Matrix3D& m1, Matrix3D& m2) {
    Matrix3D plus(m1.rows, m1.columns, m1.channels);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                uint new_value = m1.get(i, j, k) + m2.get(i, j, k);
                new_value > 255 ? 255 : new_value;
                plus.set(i, j, k, (uint8_t) new_value);
            }
        }
    }

    return plus;
}
