#ifndef MATRIX
#define MATRIX

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <float.h>

#include "util.h"

template<typename T>
class Matrix2D {
    public:
        Matrix2D(): rows{0}, columns{0} {};
        Matrix2D(int rows, int columns, bool init=false);
        Matrix2D(int rows, int columns, std::vector<T> &data);
        Matrix2D(std::vector<T> &v1, std::vector<T> &v2, int slice=0);
        ~Matrix2D() = default;

        int get_rows() const {return rows;};
        int get_columns() const {return columns;};

        T get(int row, int column);
        void set(int row, int column, T value);
        void print(bool floating_point=false);
        void reshape(int rows, int columns);
        Matrix2D<T> transpose();
        Matrix2D<double> normalize(double mean, double stdev);
        Matrix2D<double> normalize();
        Matrix2D<double> applyFunction(double (*active_fn)(double));
        Matrix2D<double> max_pooling(Shape &pooling_window);
        std::vector<T> dot(std::vector<T> &v, int slice=0);
        Matrix2D<T> dot(Matrix2D<T>& m2);
        T element_sum();

        void flatten() {rows = 1; columns = array.size();};
        std::vector<T> flatten_to_vector(int extra = 1);
        static void test_matrix2D();

        Matrix2D<T> subtract(Matrix2D<T> & m2);

        template<typename T2> friend Matrix2D<T2> operator+(Matrix2D<T2> &m1, Matrix2D<T2> &m2);
        template<typename T2> friend Matrix2D<T2> operator-(Matrix2D<T2> &m1, Matrix2D<T2> &m2);
        template<typename T2> friend Matrix2D<T2> operator*(Matrix2D<T2> &m1, Matrix2D<T2> &m2);

        template<typename T2> friend Matrix2D<T2> operator+(Matrix2D<T2> &m1, T2 scalar);
        template<typename T2> friend Matrix2D<T2> operator-(Matrix2D<T2> &m1, T2 scalar);
        template<typename T2> friend Matrix2D<T2> operator*(Matrix2D<T2> &m1, T2 scalar);
        std::vector<T> array;

    private:
        int rows;
        int columns;

        bool dynamic = false;

        int coordinate_to_index2D(int i, int j);
        const std::vector<int> index_to_coordinate2D(int index);
};

template<typename T>
class Matrix3D {
    public:
        Matrix3D(): rows{0}, columns{0}, channels{0} {};
        Matrix3D(int rows, int columns, int channels, bool init=false);
        Matrix3D(int rows, int columns, int channels, std::vector<T> *data);
        template<typename T2>
        Matrix3D(int rows, int columns, int channels, std::vector<T2> *data);
        ~Matrix3D() = default;

        int get_rows() const {return rows;};
        int get_columns() const {return columns;};
        int get_channels() const {return channels;};

        T get(int row, int column, int channel);
        void set(int row, int column, int channel, T value);
        void print(bool floating_point=false);
        Matrix3D<double> normalize(double mean, double stdev);
        Matrix3D<double> normalize();
        Matrix2D<double> convolve(Matrix3D<double> &kernel, double bias=1);
        Matrix2D<T> get_plane(int channel);
        static void test_matrix3D();

        template<typename T2> friend Matrix3D<T2> operator+(Matrix3D<T2> &m1, Matrix3D<T2> &m2);
        template<typename T2> friend Matrix3D<T2> operator-(Matrix3D<T2> &m1, Matrix3D<T2> &m2);
        template<typename T2> friend Matrix2D<T> operator*(Matrix3D<T2> &m1, Matrix3D<T2> &m2);

        template<typename T2> friend Matrix3D<T2> operator+(Matrix3D<T2> &m1, T2 &scalar);
        template<typename T2> friend Matrix3D<T2> operator-(Matrix3D<T2> &m1, T2 &scalar);
        template<typename T2> friend Matrix3D<T2> operator*(Matrix3D<T2> &m1, T2 &scalar);
        std::vector<T> array;


    private:
        int rows;
        int columns;
        int channels;

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
Matrix2D<T>::Matrix2D(int rows, int columns, bool init) {
    this->rows = rows;
    this->columns = columns;
    this->dynamic = true;

    this->array = std::vector<T>(this->rows * this->columns);
    if (init) {
        std::uniform_real_distribution<double> unif(-1, 1);
        for (size_t i = 0; i < this->array.size(); i++) {
            double a = unif(random_engine);
            this->array.at(i) = a;
        }
    }
    else {
        std::fill(this->array.begin(), this->array.end(), 0);
    }
}

template<typename T>
Matrix2D<T>::Matrix2D(std::vector<T> &v1, std::vector<T> &v2, int slice) {
    this->rows = (int) v1.size();
    this->columns = (int) v2.size() - slice;

    for (int i = 0; i < this->rows; i++){
        for (int j = 0; j< this->columns; j++){
            // this->array.at(i * this->rows + j) = v1.at(i) * v2.at(j);
            this->array.push_back(v1.at(i) * v2.at(j));
        }
    }

    // this->array = data;
}

template<typename T>
Matrix2D<double> Matrix2D<T>::normalize(double mean, double stdev) {
    Matrix2D<double> normalized_matrix(this->rows, this->columns);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
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

template<typename T>
Matrix2D<double> Matrix2D<T>::applyFunction(double (*active_fn)(double)) {
    Matrix2D<double> m3(this->get_rows(), this->get_columns());
    for (int i = 0; i < this->get_rows(); i++){
        for (int j = 0; j < this->get_columns(); j++){
            double ret = (*active_fn)(this->get(i, j));
            if (std::isnan(ret)) {
                m3.set(i, j, 0);
            }
            else {
                m3.set(i, j, ret);
            }
        }
    }

    return m3;
}

template<typename T>
Matrix2D<double> Matrix2D<T>::max_pooling(Shape &pooling_window) {
    Matrix2D<double> pooled_mat(this->rows / pooling_window.rows, this->columns / pooling_window.columns);

    for (int i = 0; i < this->rows; i += pooling_window.rows) {
        for (int j = 0; j < this->columns; j += pooling_window.columns) {
            double max = -DBL_MAX;
            for (int x = i; x < i + pooling_window.rows; x++) {
                for (int y = j; y < j + pooling_window.columns; y++) {
                    double val_at_index = this->get(x, y);
                    max = val_at_index > max ? val_at_index : max;
                }
            }
            pooled_mat.set(i / pooling_window.rows, j / pooling_window.columns, max);
        }
    }

    return pooled_mat;
}

template<typename T>
T Matrix2D<T>::element_sum() {
    T sum = 0;
    for (size_t i = 0; i < this->array.size(); i++) {
        sum += this->array.at(i);
    }

    return sum;
}

template<typename T>
std::vector<T> Matrix2D<T>::flatten_to_vector(int extra) {
    std::vector<T> flattened_vec(this->rows * this->columns + extra);

    for (int i = 0; i < this->rows * this->columns; i++) {
        flattened_vec.at(i) = this->array.at(i);
    }

    return flattened_vec;
}

template<typename T>
Matrix2D<T> Matrix2D<T>::transpose() {
    Matrix2D<T> m3(this->columns, this->rows);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++){
            m3.set(j, i, this->get(i, j));
        }
    }

    return m3;
}

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

template<typename T>
std::vector<T> Matrix2D<T>::dot(std::vector<T> &v, int slice) {
		assert(this->columns == (int) v.size() - slice);
		std::vector<double> vr(this->rows);
		for (int i=0; i < this->rows; i++){
			double w = 0;
			for (int j=0; j < this->columns - slice; j++){
				w += (this->get(i,j) * v.at(j));
			}
            vr.at(i) = w;
		}
		return vr;
}


template<typename T>
Matrix2D<T> Matrix2D<T>::dot(Matrix2D<T>& m2) {
    assert(this->rows == m2.columns || this->columns == m2.rows);
    Matrix2D<T> mult(this->rows, m2.columns);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < m2.columns; ++j) {
            T sum = 0;
            for (int k = 0; k < this->columns; ++k) {
                sum += this->get(i, k) * m2.get(k, j);
            }
            mult.set(i, j, sum);
        }
    }

    return mult;
}



/**
 * Adds m2 to m1.
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
 * Adds scalar to every index in m1.
 */
template<typename T>
Matrix2D<T> operator+(Matrix2D<T>& m1, T scalar) {
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) + scalar;
            m1.set(i, j, new_value);
        }
    }

    return m1;
}

template<typename T>
Matrix2D<T> Matrix2D<T>::subtract(Matrix2D<T> & m2){
    assert(this->get_rows() == m2.get_rows() && this->get_columns() == m2.get_columns());

    Matrix2D<T> m3(this->get_rows(), this->get_columns(), false);
    for (int i = 0; i < this->get_rows(); i++){
        for (int j = 0; j < this->get_columns(); j++){
            m3.set(i, j, this->get(i, j) - m2.get(i, j));
        }
    }
    return m3;
}

/**
 * Subtracts m2 from m1.
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
 * Subtracts scalar to every index in m1.
 */
template<typename T>
Matrix2D<T> operator-(Matrix2D<T>& m1, T scalar) {
    Matrix2D<T> min(m1.rows, m1.columns);
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) - scalar;
            min.set(i, j, new_value);
        }
    }

    return min;
}

/**
 * Multiply m1 by m2.
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
 * Multiplies every index of m1 by scalar.
 */
template<typename T>
Matrix2D<T> operator*(Matrix2D<T>& m1, T scalar) {
    Matrix2D<T> mult(m1.rows, m1.columns);
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.columns; j++) {
            int new_value = m1.get(i, j) * scalar;
            mult.set(i, j, new_value);
        }
    }

    return mult;
}

/**
 * Prints the matrix.
 */
template<typename T>
void Matrix2D<T>::print(bool floating_point) {
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

    this->array = std::vector<T>(data->begin(), data->end());

    this->offset = data->size() % (rows * columns * channels);
}

template<typename T>
Matrix3D<T>::Matrix3D(int rows, int columns, int channels, std::vector<T> *data) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;

    this->array = *(data);

    this->offset = data->size() % (rows * columns * channels);
}

template<typename T>
Matrix3D<T>::Matrix3D(int rows, int columns, int channels, bool init) {
    this->rows = rows;
    this->columns = columns;
    this->channels = channels;
    this->dynamic = true;
    this->offset = 0;

    this->array = std::vector<T>((rows * columns * channels));

    if (init) {
        std::uniform_real_distribution<double> unif(-0.5, 0.5);
        for (size_t i = 0; i < this->array.size(); i++) {
            double a = unif(random_engine);
            this->array.at(i) = a;
        }
    }
}

/**
 * Matrix2D deconstructor. Frees the array pointer if this matrix allocated this
 * pointer itself without getting it passed to it by a parameter.
 */
// template<typename T>
// Matrix3D<T>::~Matrix3D() {
//     if (this->dynamic) {
//         delete this->array;
//     }
// }

/**
 * Sets the value at the coordinate (row, column, channel) to the given value.
 */
template<typename T>
void Matrix3D<T>::set(int row, int column, int channel, T value) {
    this->array.at(coordinate_to_index3D(row, column, channel)) = value;
}

/**
 * Returns the value at (row, column, channel).
 */
template<typename T>
T Matrix3D<T>::get(int row, int column, int channel) {
    return this->array.at(coordinate_to_index3D(row, column, channel));
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
Matrix3D<double> Matrix3D<T>::normalize() {
    Matrix3D<double> normalized_matrix(this->rows, this->columns, this->channels);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            for (int k = 0; k < this->channels; k++) {
                double value = ((double) (this->get(i, j, k) / (double) UINT8_MAX) );
                normalized_matrix.set(i, j, k, value);
            }
        }
    }

    return normalized_matrix;
}

template<typename T>
Matrix2D<double> Matrix3D<T>::convolve(Matrix3D<double> &kernel, double bias) {
    Matrix2D<double> convolved_mat(this->rows - kernel.get_rows() + 1, this->columns - kernel.get_columns() + 1);

    for (int i = 0; i < convolved_mat.get_rows(); i++) {
        for (int j = 0; j < convolved_mat.get_columns(); j++) {
            double value = 0.0;
            for (int h = i; h < i + kernel.get_rows(); h++) {
                for (int w = j; w < j + kernel.get_columns(); w++) {
                    for (int channel = 0; channel < this->channels; channel++) {
                        value += kernel.get(h - i, w - j, channel) * this->get(h, w, channel);
                    }
                }
            }
            convolved_mat.set(i, j, value + bias);
        }
    }

    return convolved_mat;
}

template<typename T>
Matrix2D<T> Matrix3D<T>::get_plane(int channel) {
    assert(channel < this->channels && channel >= 0);

    Matrix2D<T> plane(this->rows, this->columns);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            plane.set(i, j, this->get(i, j, channel));
        }
    }

    return plane;
}

/**
 * Prints the matrix.
 */
template<typename T>
void Matrix3D<T>::print(bool floating_point) {
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
 * Adds scalar to every index of m1.
 */
template<typename T>
Matrix3D<T> operator+(Matrix3D<T>& m1, T &scalar) {
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
 * Subtracts scalar from every index in m1.
 */
template<typename T>
Matrix3D<T> operator-(Matrix3D<T>& m1, T &scalar) {
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
 * Multiplies every index of m1 by scalar.
 */
template<typename T>
Matrix3D<T> operator*(Matrix3D<T>& m1, T &scalar) {
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
    // std::vector<uint8_t> dub = {1, 2, 3, 4, 5, 6, 7, 8};

    Matrix2D<uint8_t> matA(4, 1, a);
    Matrix2D<uint8_t> matB(4, 1, b);
    Matrix2D<uint8_t> matK(1, 4, k);
    Matrix2D<uint8_t> matC(4, 2, c);
    Matrix2D<uint8_t> matD(8, 1, d);
    // Matrix2D<double> dubs(4, 2, d);

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
    // std::cout << "testing multiplication with constant" << std::endl;
    // dubs = dubs * 0.1;
    // dubs.print();

    std::cout << "testing flatten" << std::endl;
    mult.flatten();
    mult.print();
    std::cout << std::endl;

    std::cout << "testing normalization" << std::endl;
    Matrix2D<double> normie = matD.normalize(0.5, 0.5);
    normie.print(true);
    std::cout << std::endl;

    std::cout << "testing randomization" << std::endl;
    Matrix2D<double> rand(3, 3, true);
    Matrix2D<double> rand2(3, 3, true);
    rand.print(true);
    std::cout << std::endl;
    rand2.print(true);
    std::cout << std::endl;

    std::cout << "testing maxpooling" << std::endl;
    Matrix2D<double> img(4, 4, true);
    img.print(true);
    std::cout << std::endl;
    Shape test = {2,2};
    Matrix2D<double> pooled = img.max_pooling(test);
    pooled.print(true);
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

    std::cout << "testing randomizaion" << std::endl;
    Matrix3D<double> randie(3, 3, 3, true);
    randie.print(true);
    std::cout << '\n';

    std::cout << "testing convolution" << std::endl;
    std::vector<double> i = {1, 2, 3, 1, 2, 3,1, 2, 3,      1, 2, 3,1, 2, 3,1, 2, 3,        1, 2, 3,1, 2, 3,1, 2, 3};
    std::vector<double> kern = {1.5f, 1.5f, 2.6f,  1.5f,        2.0f, 2.0f, 2.0f, 2.0f,      3.0f, 3.0f, 3.0f, 3.0f};
    Matrix3D<double> img(3, 3, 3, &i);
    Matrix3D<double> kernel(2, 2, 3, &kern);
    Matrix2D<double> convolved = img.convolve(kernel);
    convolved.print(true);
    std::cout << '\n';
}

#endif
