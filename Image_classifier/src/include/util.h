#ifndef UTIL
#define UTIL

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <time.h>
#include <assert.h>

extern std::default_random_engine random_engine;

struct Shape {
	int rows;
	int columns;
};

struct Shape3D {
	int rows;
	int columns;
	int channels;
};

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
	template <typename T>
    std::vector<std::vector<T>> split_vector(std::vector<T> &vec, size_t n);
}

/**
 * @brief Split a vector in n equal parts.
 *
 * @param vec Vector to split
 * @param n Split the vector in n parts.
 * @return std::vector<std::vector<uint8_t>> Vector containing all the split vectors.
 */
template <typename T>
std::vector<std::vector<T>> util::split_vector(std::vector<T> &vec, size_t n) {
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i) {
        end += (remain > 0) ? (length + !!(remain--)) : length;
        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));
        begin = end;
    }

    return outVec;
}

namespace np {
    std::vector<double> applyFunction(std::vector<double> &v, double (*function)(double));
	std::vector<double> normalize(std::vector<double> &v);
	std::vector<double> multiply(std::vector<double> &v1, std::vector<double> &v2);
	void multiply(std::vector<double> &v1, double val);
	double sum(std::vector<double> v1);
	std::vector<double> subtract(std::vector<double> & v1, std::vector<double> & v2);
	int get_max_class(std::vector<double> &prediction);
}

#endif
