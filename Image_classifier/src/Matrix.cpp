#include "include/Matrix.h"


Matrix3D::Matrix3D(int rows, int columns, int channels, std::vector<uint8_t> *data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->array = data;
    this->offset = data->size() % (rows * columns * channels);
}

Matrix3D::Matrix3D(int rows, int columns, int channels) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->dynamic = true;

    this->array = new std::vector<uint8_t>((rows * columns * channels));
    std::fill(this->array->begin(), this->array->end(), 0);
}

/**
 * Matrix2D deconstructor. Frees the array pointer if this matrix allocated this
 * pointer itself without getting it passed to it by a parameter.
 */
Matrix3D::~Matrix3D() {
    if (this->dynamic) {
        delete this->array;
    }
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
    return offset + ((k * this->rows * this->columns) + (i * this->rows) + j);
}

//TODO: Check of deze functie ook echt werkt. Wss hebben we m niet nodig tho
// dus check pas wanneer je m gaat gebruiken want is wss fout.
/**
 * Transforms a 1D array index to a 3D coordinate.
 */
const std::vector<int> Matrix3D::index_to_coordinate3D(int index) {
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
			    std::cout << (int) get(i, j, k) << " " ;
            }
            std::cout << "-|- ";
		}
		std::cout << "\n\n";
	}
    std::cout << std::endl;
}

/**
 * Adds two 3D matrices together.
 */
Matrix3D operator+(Matrix3D& m1, Matrix3D& m2) {
    Matrix3D plus(m1.rows, m1.columns, m1.channels);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                uint new_value = m1.get(i, j, k) + m2.get(i, j, k);
                new_value = new_value > 255 ? 255 : new_value;
                plus.set(i, j, k, (uint8_t) new_value);
            }
        }
    }

    return plus;
}

/**
 * Adds scalar to every index of m1. Accounts for overflow by limiying the
 * result to 255.
 */
Matrix3D operator+(Matrix3D& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                int new_value = m1.get(i, j, k) + scalar;
                new_value = new_value > 255 ? 255 : new_value;
                m1.set(i, j, k, (uint8_t) new_value);
            }
        }
    }

    return m1;
}

/**
 * Subtracts m2 from m1.
 */
Matrix3D operator-(Matrix3D& m1, Matrix3D& m2) {
    Matrix3D minus(m1.rows, m1.columns, m1.channels);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                int new_value = m1.get(i, j, k) - m2.get(i, j, k);
                new_value = new_value < 0 ? 0 : new_value;
                minus.set(i, j, k, (uint8_t) new_value);
            }
        }
    }

    return minus;
}

/**
 * Subtracts scalar from every index in m1. Accounts for underflow by limiting the
 * result to 255.
 */
Matrix3D operator-(Matrix3D& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                int new_value = m1.get(i, j, k) - scalar;
                new_value = new_value < 0 ? 0 : new_value;
                m1.set(i, j, k, (uint8_t) new_value);
            }
        }
    }

    return m1;
}

/**
 * Multiply m1 by m2.
 * This is a mathematically valid matrix multiplication algorithm but that is
 * not necesary for this application. We only want to multiply the third
 * dimension of the two matrices which are the RGB color values.
 * This could be seen as calculating the gain of m1 and m2.
 * That is why it returns a Matrix2D and not a Matrix3D object.
 */
Matrix2D operator*(Matrix3D& m1, Matrix3D& m2) {
    Matrix2D mult(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int sum = 0;
            for (int k = 0; k < m1.channels; k++) {
                sum += m1.get(i, j, k) * m2.get(i, j, k);
            }
            sum = sum > 255 ? 255 : sum;
            mult.set(i, j, (uint8_t) sum);
        }
    }

    return mult;
}

/**
 * Multiplies every index of m1 by scalar. Accounts for overflow by limiting the
 * result  to 255.
 */
Matrix3D operator*(Matrix3D& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                int new_value = m1.get(i, j, k) * scalar;
                new_value = new_value > 255 ? 255 : new_value;
                m1.set(i, j, k, (uint8_t) new_value);
            }
        }
    }

    return m1;
}

Matrix2D::Matrix2D(int rows, int columns, std::vector<uint8_t> *data) {
    this->rows = rows;
    this->columns = columns;

    this->array = data;
}

Matrix2D::Matrix2D(int rows, int columns) {
    this->rows = rows;
    this->columns = columns;
    this->dynamic = true;

    this->array = new std::vector<uint8_t>(this->rows * this->columns);
    std::fill(this->array->begin(), this->array->end(), 0);
}

/**
 * Matrix2D deconstructor. Frees the array pointer if this matrix allocated this
 * pointer itself without getting it passed to it by a parameter.
 */
Matrix2D::~Matrix2D() {
    if (this->dynamic) {
        delete this->array;
    }
}

/**
 * Sets the value of index (i, j) to value.
 */
void Matrix2D::set(int row, int column, uint8_t value) {
    this->array->at(coordinate_to_index2D(row, column)) = value;
}

/**
 * Gets the value stored at index (row, column);
 */
uint8_t Matrix2D::get(int row, int column) {
    return this->array->at(coordinate_to_index2D(row, column));
}

/**
 * Transforms a 2D coordinate (i, j) to a 1D index.
 */
int Matrix2D::coordinate_to_index2D(int i, int j) {
    return i * this->rows + j;
}

/**
 * Transforms a 1D index to a 2D coordinate.
 */
const std::vector<int> Matrix2D::index_to_coordinate2D(int index) {
    std::vector<int> coordinate(2);

    int row = index % this->rows;
    int column = index - row * this->rows;

    coordinate[0] = row;
    coordinate[1] = column;

    return coordinate;
}

/**
 * Adds m2 to m1. Accounts for overflow by limiting the result value to 255.
 */
Matrix2D operator+(Matrix2D& m1, Matrix2D& m2) {
    Matrix2D plus(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) + m2.get(i, j);
            new_value = new_value > 255 ? 255 : new_value;
            plus.set(i, j, (uint8_t) new_value);
        }
    }

    return plus;
}

/**
 * Adds scalar to every index in m1. Accounts for overflow by limiting the result
 * to 255.
 */
Matrix2D operator+(Matrix2D& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) + scalar;
            new_value = new_value > 255 ? 255 : new_value;
            m1.set(i, j, (uint8_t) new_value);
        }
    }

    return m1;
}

/**
 * Subtracts m2 from m1. Accounts for underflow by limiting the result value to 0.
 */
Matrix2D operator-(Matrix2D& m1, Matrix2D& m2) {
    Matrix2D minus(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) - m2.get(i, j);
            new_value = new_value < 0 ? 0 : new_value;
            minus.set(i, j, (uint8_t) new_value);
        }
    }

    return minus;
}

/**
 * Subtracts scalar to every index in m1. Accounts for underflow by limiting the
 * result to 0.
 */
Matrix2D operator-(Matrix2D& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) - scalar;
            new_value = new_value < 0 ? 0 : new_value;
            m1.set(i, j, (uint8_t) new_value);
        }
    }

    return m1;
}

/**
 * Multiply m1 by m2. Accounts for overflow by limiting the value to 255.
 */
Matrix2D operator*(Matrix2D& m1, Matrix2D& m2) {
    assert(m1.rows == m2.columns);
    Matrix2D mult(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; ++i) {
        for (int j = 0; j < m2.columns; ++j) {
            int sum = 0;
            for (int k = 0; k < m1.columns; ++k) {
                sum += m1.get(i, k) * m2.get(k, j);
            }
            sum = sum > 255 ? 255 : sum;
            mult.set(i, j, (uint8_t) sum);
        }
    }

    return mult;
}

/**
 * Multiplies every index of m1 by scalar. Accounts for overflow by limiting the
 * result  to 255.
 */
Matrix2D operator*(Matrix2D& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) * scalar;
            new_value = new_value > 255 ? 255 : new_value;
            m1.set(i, j, (uint8_t) new_value);
        }
    }

    return m1;
}

/**
 * Prints the matrix.
 */
void Matrix2D::print() {
    int cols = this->columns;
    for (long unsigned int i = 0; i < this->array->size(); i++) {
        if ((i + 1) % cols == 0) {
            std::cout << (int) this->array->at(i) << '\n';
        }
        else {
            std::cout << (int) this->array->at(i) << " , ";
        }
    }
}

void Matrix2D::test_matrix2D() {
    std::vector<uint8_t> a = {1, 2, 3, 4};
    std::vector<uint8_t> b = {5, 6, 7, 8};

    Matrix2D matA(2, 2, &a);
    Matrix2D matB(2, 2, &b);
    std::cout << "testing addition" << std::endl;
    Matrix2D plus = matA + matB;
    plus.print();
    std::cout << '\n';

    std::cout << "testing subtraction" << std::endl;
    Matrix2D minus = matA - matB;
    minus.print();
    std::cout << '\n';

    std::cout << "testing multiplication" << std::endl;
    Matrix2D mult = matA * matB;
    mult.print();
    std::cout << std::endl;
}

void Matrix3D::test_matrix3D() {
    std::vector<uint8_t> a = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    std::vector<uint8_t> b = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};

    Matrix3D matA(2, 2, 3, &a);
    Matrix3D matB(2, 2, 3, &b);
    std::cout << "testing addition" << std::endl;
    Matrix3D plus = matA + matB;
    plus.print();
    std::cout << '\n';

    std::cout << "testing subtraction" << std::endl;
    Matrix3D minus = matA - matB;
    minus.print();
    std::cout << '\n';

    std::cout << "testing multiplication" << std::endl;
    Matrix2D mult = (matA * matB);
    mult.print();
    std::cout << '\n';
}
