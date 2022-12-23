#include <string>
#include <vector>
#include <iostream>

class Matrix3D {
    public:
        Matrix3D(int rows, int columns, int channels, std::vector<uint8_t> &data);
        ~Matrix3D() = default;

        int get_rows() {return rows;};
        int get_columns() {return columns;};
        int get_channels() {return channels;};

        uint8_t get(int row, int column, int channel);
        void set(int row, int column, int channel, uint8_t value);
        void print();

    private:
        std::vector<uint8_t> array;
        int columns;
        int rows;
        int channels;

        int coordinate_to_index3D(int i, int j, int k);
};