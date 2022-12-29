#ifndef UTIL
#define UTIL

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <float.h>
#include <time.h>

#include "Matrix.hpp"


struct Shape{
	int rows;
	int columns;
};

extern std::default_random_engine random_engine;

namespace fns{
	double relu(double x);
	double sigmoid(double x);
	double tan(double x);
	double relu_gradient(double x);
	double sigmoid_gradient(double x);
	double tan_gradient(double x);
	double softmax(double x);
}

namespace util {
    std::vector<std::vector<uint8_t>> split_vector(std::vector<uint8_t> &vec, size_t n);
}

namespace np {
    std::vector<double> applyFunction(std::vector<double> &v, double (*function)(double));
    Matrix2D<double> applyFunction(Matrix2D<double> & m1, double (*function)(double));
}

#endif
