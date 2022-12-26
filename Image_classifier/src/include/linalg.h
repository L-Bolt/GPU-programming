#ifndef LINALG
#define LINALG

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>

#include "Matrix.h"


namespace linalg {
    Matrix3D convolve(Matrix3D &mat, Matrix2D &kernel);
    Matrix2D convolve(Matrix2D &mat, Matrix2D &kernel);
}

#endif
