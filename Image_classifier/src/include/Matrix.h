#ifndef MATRIX
#define MATRIX

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>

// Todo: Fix inheritance van een algemene Matrix class. En dan Matrix1D en Matrix3D
// met override functies (virtual functies).
class Matrix2D {
    public:
        Matrix2D(): rows{0}, columns{0}, array{NULL} {};
        Matrix2D(int rows, int columns);
        Matrix2D(int rows, int columns, std::vector<uint8_t> *data);
        ~Matrix2D();

        int get_rows() const {return rows;};
        int get_columns() const {return columns;};

        uint8_t get(int row, int column);
        void set(int row, int column, uint8_t value);
        void print();
        static void test_matrix2D();

        friend Matrix2D operator+(Matrix2D& m1, Matrix2D& m2);
        friend Matrix2D operator-(Matrix2D& m1, Matrix2D& m2);
        friend Matrix2D operator*(Matrix2D& m1, Matrix2D& m2);

    private:
        int rows;
        int columns;
        bool dynamic = false;
        std::vector<uint8_t> *array;

        int coordinate_to_index2D(int i, int j);
        const std::vector<int> index_to_coordinate2D(int index);
};

class Matrix3D {
    public:
        Matrix3D(): rows{0}, columns{0}, channels{0}, array{NULL} {};
        Matrix3D(int rows, int columns, int channels);
        Matrix3D(int rows, int columns, int channels, std::vector<uint8_t> *data);
        ~Matrix3D();

        int get_rows() const {return rows;};
        int get_columns() const {return columns;};
        int get_channels() const {return channels;};

        uint8_t get(int row, int column, int channel);
        void set(int row, int column, int channel, uint8_t value);
        void print();
        static void test_matrix3D();

        friend Matrix3D operator+(Matrix3D& m1, Matrix3D& m2);
        friend Matrix3D operator-(Matrix3D& m1, Matrix3D& m2);
        friend Matrix2D operator*(Matrix3D& m1, Matrix3D& m2);

    private:
        int rows;
        int columns;
        int channels;
        std::vector<uint8_t> *array;

        bool dynamic = false;
        int offset = 0;

        int coordinate_to_index3D(int i, int j, int k);
        const std::vector<int> index_to_coordinate3D(int index);
};

#endif
