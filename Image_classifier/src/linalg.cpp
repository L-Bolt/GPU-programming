#include "include/linalg.h"

Matrix3D linalg::convolve(Matrix3D &mat, Matrix2D &kernel) {
    std::cout << "test" << std::endl;
    Matrix3D result(mat.get_rows(), mat.get_columns(), mat.get_channels());

    return result;
}

Matrix2D linalg::convolve(Matrix2D &mat, Matrix2D &kernel) {
    std::cout << "test" << std::endl;
    Matrix2D result(mat.get_rows(), mat.get_columns());

    return result;
}
