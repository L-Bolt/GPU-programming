#ifndef MATRIX
#define MATRIX

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>

// Todo: Fix inheritance van een algemene Matrix class. En dan Matrix1D en Matrix3D
// met override functies (virtual functies).
template<typename T>
class Matrix2D {
    public:
        Matrix2D(): rows{0}, columns{0}, array{NULL} {};
        Matrix2D(int rows, int columns);
        Matrix2D(int rows, int columns, std::vector<T> &data);
        ~Matrix2D() = default;

        int get_rows() const {return rows;};
        int get_columns() const {return columns;};

        T get(int row, int column);
        void set(int row, int column, T value);
        void print(double floating_point=false);
        void reshape(int rows, int columns);
        Matrix2D<double> normalize(double mean, double stdev);
        Matrix2D<double> normalize();
        void flatten() {rows = 1; columns = array.size();};
        static void test_matrix2D();

        template<typename T2> friend Matrix2D<T2> operator+(Matrix2D<T2> &m1, Matrix2D<T2> &m2);
        template<typename T2> friend Matrix2D<T2> operator-(Matrix2D<T2> &m1, Matrix2D<T2> &m2);
        template<typename T2> friend Matrix2D<T2> operator*(Matrix2D<T2> &m1, Matrix2D<T2> &m2);

        template<typename T2> friend Matrix2D<T2> operator+(Matrix2D<T2> &m1, int &scalar);
        template<typename T2> friend Matrix2D<T2> operator-(Matrix2D<T2> &m1, int &scalar);
        template<typename T2> friend Matrix2D<T2> operator*(Matrix2D<T2> &m1, int &scalar);

    private:
        int rows;
        int columns;
        std::vector<T> array;

        bool dynamic = false;

        int coordinate_to_index2D(int i, int j);
        const std::vector<int> index_to_coordinate2D(int index);
};

template<typename T>
class Matrix3D {
    public:
        Matrix3D(): rows{0}, columns{0}, channels{0}, array{NULL} {};
        Matrix3D(int rows, int columns, int channels);
        Matrix3D(int rows, int columns, int channels, std::vector<T> *data);
        template<typename T2>
        Matrix3D(int rows, int columns, int channels, std::vector<T2> *data);
        ~Matrix3D();

        int get_rows() const {return rows;};
        int get_columns() const {return columns;};
        int get_channels() const {return channels;};

        T get(int row, int column, int channel);
        void set(int row, int column, int channel, T value);
        void print(double floating_point=false);
        Matrix3D<int> to_int();
        Matrix3D<double> normalize(double mean, double stdev);
        static void test_matrix3D();

        template<typename T2> friend Matrix3D<T2> operator+(Matrix3D<T2> &m1, Matrix3D<T2> &m2);
        template<typename T2> friend Matrix3D<T2> operator-(Matrix3D<T2> &m1, Matrix3D<T2> &m2);
        template<typename T2> friend Matrix2D<T> operator*(Matrix3D<T2> &m1, Matrix3D<T2> &m2);

        template<typename T2> friend Matrix3D<T2> operator+(Matrix3D<T2> &m1, int &scalar);
        template<typename T2> friend Matrix3D<T2> operator-(Matrix3D<T2> &m1, int &scalar);
        template<typename T2> friend Matrix3D<T2> operator*(Matrix3D<T2> &m1, int &scalar);

    private:
        int rows;
        int columns;
        int channels;
        std::vector<T> *array;

        bool dynamic = false;
        int offset = 0;

        int coordinate_to_index3D(int i, int j, int k);
        const std::vector<int> index_to_coordinate3D(int index);
};


template<typename T>
Matrix2D<T>::Matrix2D(int rows, int columns, std::vector<T> &data) {
    this->rows = rows;
    this->columns = columns;
    this->array = data;
}

template<typename T>
Matrix2D<T>::Matrix2D(int rows, int columns) {
    this->rows = rows;
    this->columns = columns;
    this->dynamic = true;

    this->array = std::vector<T>(this->rows * this->columns);
    std::fill(this->array.begin(), this->array.end(), 0);
}

template<typename T>
Matrix2D<double> Matrix2D<T>::normalize(double mean, double stdev) {
    Matrix2D<double> normalized_matrix(this->rows, this->columns);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            int index = coordinate_to_index2D(i, j);
            double value = ((double) (this->get(i, j) / (double) UINT8_MAX) - mean) / stdev;
            normalized_matrix.set(i, j, value);
        }
    }

    return normalized_matrix;
}

template<typename T>
Matrix2D<double> Matrix2D<T>::normalize() {

    unsigned int sum = 0;
    for (size_t i = 0; i < this->array.size(); i++) {
        sum += this->array.at(i);
    }
    assert(sum > 0);

    Matrix2D<double> normalized_matrix(this->rows, this->columns);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            double value = (double) this->get(i, j) / (double) sum;
            normalized_matrix.set(i, j, value);
        }
    }

    return normalized_matrix;
}

/**
 * Matrix2D deconstructor. Frees the array pointer if this matrix allocated this
 * pointer itself without getting it passed to it by a parameter.
 */
// template<typename T>
// Matrix2D<T>::~Matrix2D() {
//     if (this->dynamic) {
//         delete this->array;
//     }
// }

/**
 * Sets the value of index (i, j) to value.
 */
template<typename T>
void Matrix2D<T>::set(int row, int column, T value) {
    this->array.at(coordinate_to_index2D(row, column)) = value;
}

/**
 * Gets the value stored at index (row, column);
 */
template<typename T>
T Matrix2D<T>::get(int row, int column) {
    return this->array.at(coordinate_to_index2D(row, column));
}

/**
 * Transforms a 2D coordinate (i, j) to a 1D index.
 */
template<typename T>
int Matrix2D<T>::coordinate_to_index2D(int i, int j) {
    return i * this->columns + j;
}

/**
 * Transforms a 1D index to a 2D coordinate.
 */
template<typename T>
const std::vector<int> Matrix2D<T>::index_to_coordinate2D(int index) {
    std::vector<int> coordinate(2);

    int row = index % this->rows;
    int column = index - row * this->rows;

    coordinate[0] = row;
    coordinate[1] = column;

    return coordinate;
}

//TODO: iets beter testen.
/**
 * Reshapes the matrix.
 */
template<typename T>
void Matrix2D<T>::reshape(int rows, int columns) {
    assert(rows * columns == (int) this->array.size());

    this->rows = rows;
    this->columns = columns;
}

/**
 * Adds m2 to m1. Accounts for overflow by limiting the result value to 255.
 */
template<typename T>
Matrix2D<T> operator+(Matrix2D<T>& m1, Matrix2D<T>& m2) {
    assert (m1.rows == m2.rows && m1.columns == m2.columns);
    Matrix2D<T> plus(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) + m2.get(i, j);
            plus.set(i, j, new_value);
        }
    }

    return plus;
}

/**
 * Adds scalar to every index in m1. Accounts for overflow by limiting the result
 * to 255.
 */
template<typename T>
Matrix2D<T> operator+(Matrix2D<T>& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) + scalar;
            m1.set(i, j, new_value);
        }
    }

    return m1;
}

/**
 * Subtracts m2 from m1. Accounts for underflow by limiting the result value to 0.
 */
template<typename T>
Matrix2D<T> operator-(Matrix2D<T>& m1, Matrix2D<T>& m2) {
    assert (m1.rows == m2.rows && m1.columns == m2.columns);
    Matrix2D<T> minus(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) - m2.get(i, j);
            minus.set(i, j, new_value);
        }
    }

    return minus;
}

/**
 * Subtracts scalar to every index in m1. Accounts for underflow by limiting the
 * result to 0.
 */
template<typename T>
Matrix2D<T> operator-(Matrix2D<T>& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) - scalar;
            m1.set(i, j, new_value);
        }
    }

    return m1;
}

/**
 * Multiply m1 by m2. Accounts for overflow by limiting the value to 255.
 */
template<typename T>
Matrix2D<T> operator*(Matrix2D<T>& m1, Matrix2D<T>& m2) {
    assert(m1.rows == m2.columns);
    Matrix2D<T> mult(m1.rows, m2.columns);

    for (int i = 0; i < m1.rows; ++i) {
        for (int j = 0; j < m2.columns; ++j) {
            int sum = 0;
            for (int k = 0; k < m1.columns; ++k) {
                sum += m1.get(i, k) * m2.get(k, j);
            }
            mult.set(i, j, sum);
        }
    }

    return mult;
}

/**
 * Multiplies every index of m1 by scalar. Accounts for overflow by limiting the
 * result  to 255.
 */
template<typename T>
Matrix2D<T> operator*(Matrix2D<T>& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) * scalar;
            m1.set(i, j, new_value);
        }
    }

    return m1;
}

/**
 * Prints the matrix.
 */
template<typename T>
void Matrix2D<T>::print(double floating_point) {
    int cols = this->columns;
    if (!floating_point) {
        for (long unsigned int i = 0; i < this->array.size(); i++) {
            if ((i + 1) % cols == 0) {
                printf("%d\n", (int) this->array.at(i));
            }
            else {
                printf("%d , ", (int) this->array.at(i));
            }
        }
    }
    else {
        for (long unsigned int i = 0; i < this->array.size(); i++) {
            if ((i + 1) % cols == 0) {
                printf("%f\n", (double) this->array.at(i));
            }
            else {
                printf("%f , ", (double) this->array.at(i));
            }
        }
    }
}


template<typename T>
template<typename T2>
Matrix3D<T>::Matrix3D(int rows, int columns, int channels, std::vector<T2> *data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;

    this->array = new std::vector<T>(data->begin(), data->end());
    this->dynamic = true;

    this->offset = data->size() % (rows * columns * channels);
}

template<typename T>
Matrix3D<T>::Matrix3D(int rows, int columns, int channels, std::vector<T> *data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;

    this->array = data;

    this->offset = data->size() % (rows * columns * channels);
}

template<typename T>
Matrix3D<T>::Matrix3D(int rows, int columns, int channels) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->dynamic = true;

    this->array = new std::vector<T>((rows * columns * channels) + 1);
}

/**
 * Matrix2D deconstructor. Frees the array pointer if this matrix allocated this
 * pointer itself without getting it passed to it by a parameter.
 */
template<typename T>
Matrix3D<T>::~Matrix3D() {
    if (this->dynamic) {
        delete this->array;
    }
}

/**
 * Sets the value at the coordinate (row, column, channel) to the given value.
 */
template<typename T>
void Matrix3D<T>::set(int row, int column, int channel, T value) {
    this->array->at(coordinate_to_index3D(row, column, channel)) = value;
}

/**
 * Returns the value at (row, column, channel).
 */
template<typename T>
T Matrix3D<T>::get(int row, int column, int channel) {
    return this->array->at(coordinate_to_index3D(row, column, channel));
}

/**
 * Transforms the coordinate (i, j, k) to the 1D coordinate used in the array.
 */
template<typename T>
int Matrix3D<T>::coordinate_to_index3D(int i, int j, int k) {
    return offset + ((k * this->rows * this->columns) + (i * this->rows) + j);
}

//TODO: Check of deze functie ook echt werkt. Wss hebben we m niet nodig tho
// dus check pas wanneer je m gaat gebruiken want is wss fout.
/**
 * Transforms a 1D array index to a 3D coordinate.
 */
template<typename T>
const std::vector<int> Matrix3D<T>::index_to_coordinate3D(int index) {
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

template<typename T>
Matrix3D<double> Matrix3D<T>::normalize(double mean, double stdev) {
    Matrix3D<double> normalized_matrix(this->rows, this->columns, this->channels);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            for (int k = 0; k < this->channels; k++) {
                double value = ((double) (this->get(i, j, k) / (double) UINT8_MAX) - mean) / stdev;
                normalized_matrix.set(i, j, k, value);
            }
        }
    }

    return normalized_matrix;
}

template<typename T>
Matrix3D<int> Matrix3D<T>::to_int() {
    Matrix3D<int> int_matrix(this->rows, this->columns, this->channels);
    for (size_t i = 0; i < this->array->size(); i++) {
        int_matrix.array->at(i) = this->array->at(i);
    }

    return int_matrix;
}

/**
 * Prints the matrix.
 */
template<typename T>
void Matrix3D<T>::print(double floating_point) {
    for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
            for (int k = 0; k < channels; k++) {
                if (!floating_point) {
			        std::cout << (int) get(i, j, k) << " " ;
                }
                else {
                    std::cout << (double) get(i, j, k) << " " ;
                }
            }
            std::cout << "| ";
		}
		std::cout << "\n\n";
	}
    std::cout << std::endl;
}

/**
 * Adds two 3D matrices together.
 */
template<typename T>
Matrix3D<T> operator+(Matrix3D<T>& m1, Matrix3D<T>& m2) {
    assert (m1.rows == m2.rows && m1.columns == m2.columns && m1.channels == m2.channels);
    Matrix3D<T> plus(m1.rows, m1.columns, m1.channels);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                T new_value = m1.get(i, j, k) + m2.get(i, j, k);
                plus.set(i, j, k, new_value);
            }
        }
    }

    return plus;
}

/**
 * Adds scalar to every index of m1. Accounts for overflow by limiying the
 * result to 255.
 */
template<typename T>
Matrix3D<T> operator+(Matrix3D<T>& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                T new_value = m1.get(i, j, k) + scalar;
                m1.set(i, j, k, new_value);
            }
        }
    }

    return m1;
}

/**
 * Subtracts m2 from m1.
 */
template<typename T>
Matrix3D<T> operator-(Matrix3D<T>& m1, Matrix3D<T>& m2) {
    assert (m1.rows == m2.rows && m1.columns == m2.columns && m1.channels == m2.channels);
    Matrix3D<T> minus(m1.rows, m1.columns, m1.channels);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                T new_value = m1.get(i, j, k) - m2.get(i, j, k);
                minus.set(i, j, k, new_value);
            }
        }
    }

    return minus;
}

/**
 * Subtracts scalar from every index in m1. Accounts for underflow by limiting the
 * result to 255.
 */
template<typename T>
Matrix3D<T> operator-(Matrix3D<T>& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                T new_value = m1.get(i, j, k) - scalar;
                m1.set(i, j, k, new_value);
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
template<typename T>
Matrix2D<T> operator*(Matrix3D<T>& m1, Matrix3D<T>& m2) {
    Matrix2D mult(m1.rows, m1.columns);

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int sum = 0;
            for (int k = 0; k < m1.channels; k++) {
                sum += m1.get(i, j, k) * m2.get(i, j, k);
            }
            mult.set(i, j, sum);
        }
    }

    return mult;
}

/**
 * Multiplies every index of m1 by scalar. Accounts for overflow by limiting the
 * result  to 255.
 */
template<typename T>
Matrix3D<T> operator*(Matrix3D<T>& m1, int &scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            for (int k = 0; k < m1.channels; k++) {
                T new_value = m1.get(i, j, k) * scalar;
                m1.set(i, j, k, new_value);
            }
        }
    }

    return m1;
}

/**
 * Debug function to test Matrix2D implementation.
 */
template<typename T>
void Matrix2D<T>::test_matrix2D() {
    std::vector<uint8_t> a = {1, 2, 3, 4};
    std::vector<uint8_t> b = {5, 6, 7, 8};
    std::vector<uint8_t> k = {5, 6, 7, 8};
    std::vector<uint8_t> c = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint8_t> d = {1, 2, 3, 4, 5, 6, 7, 8};

    Matrix2D<uint8_t> matA(4, 1, a);
    Matrix2D<uint8_t> matB(4, 1, b);
    Matrix2D<uint8_t> matK(1, 4, k);
    Matrix2D<uint8_t> matC(4, 2, c);
    Matrix2D<uint8_t> matD(8, 1, d);

    std::cout << "resizing matrix" << std::endl;
    matC.print();
    std::cout << "\nreshaping to (2, 4) \n";
    matC.reshape(2, 4);
    matC.print();
    std::cout << "\nreshaping to (8,1)" << std::endl;
    matC.reshape(8, 1);
    matC.print();
    std::cout << std::endl;

    std::cout << "testing addition" << std::endl;
    Matrix2D<uint8_t> plus = matA + matB;
    plus.print();
    std::cout << '\n';

    std::cout << "testing subtraction" << std::endl;
    Matrix2D<uint8_t> minus = matA - matB;
    minus.print();
    std::cout << '\n';

    std::cout << "testing multiplication" << std::endl;
    Matrix2D<uint8_t> mult = matA * matK;
    mult.print();
    std::cout << std::endl;

    std::cout << "testing flatten" << std::endl;
    mult.flatten();
    mult.print();
    std::cout << std::endl;

    std::cout << "testing normalization" << std::endl;
    Matrix2D<double> normie = matD.normalize(0.5, 0.5);
    normie.print(true);
    std::cout << std::endl;
}

/**
 * Debug function to test Matrix3D implementation.
 */
template<typename T>
void Matrix3D<T>::test_matrix3D() {
    std::vector<uint8_t> a = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    std::vector<uint8_t> b = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};

    std::vector<double> c = {1.5f, 1.5f, 2.6f, 1.5f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    std::vector<double> d = {4.4f, 4.3f, 78.5f, 4.0f, 5.0f, 5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f, 6.0f};

    Matrix3D<int> matA(2, 2, 3, &a);
    Matrix3D<int> matB(2, 2, 3, &b);
    std::cout << "testing addition" << std::endl;
    Matrix3D<int> plus = matA + matB;
    plus.print(false);
    std::cout << '\n';

    Matrix3D<double> matC(2, 2, 3, &c);
    Matrix3D<double> matD(2, 2, 3, &d);
    std::cout << "testing subtraction" << std::endl;
    Matrix3D<double> minus = matC - matD;
    minus.print(true);
    std::cout << '\n';

    std::cout << "testing normalization" << std::endl;
    Matrix3D<double> normie = matA.normalize(0.5, 0.5);
    normie.print(true);
    std::cout << '\n';
}

#endif