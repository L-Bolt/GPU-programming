#include "include/util.h"


int seed = time(0);
std::default_random_engine random_engine(seed);

double fns::relu(double x){
    if (x > 0) return x;
    else return (double) 0;
}

double fns::sigmoid(double x){
    return (1.0 / (1.0 + exp(-x)));
}

double fns::tan(double x){
    return tanh(x);
}

double fns::relu_gradient(double x){
    if (x > 0) return (double) 1;
    else return (double) 0.2;
}

double fns::sigmoid_gradient(double x){
    return (x * (1 - x));
}

double fns::tan_gradient(double x){
    return (1 - (x * x));
}

double fns::softmax(double x){
    if (isnan(x)) return 0;
    return exp(x);
}

/**
 * @brief Split a vector in n equal parts.
 *
 * @param vec Vector to split
 * @param n Split the vector in n parts.
 * @return std::vector<std::vector<uint8_t>> Vector containing all the split vectors.
 */
std::vector<std::vector<uint8_t>> util::split_vector(std::vector<uint8_t> &vec, size_t n) {
    std::vector<std::vector<uint8_t>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i) {
        end += (remain > 0) ? (length + !!(remain--)) : length;
        outVec.push_back(std::vector<uint8_t>(vec.begin() + begin, vec.begin() + end));
        begin = end;
    }

    return outVec;
}

// apply a function to every element of the vector
std::vector<double> np::applyFunction(std::vector<double> &v, double (*active_fn)(double)){
    std::vector<double> vr;
    for(unsigned int i=0; i < v.size(); i++){
        if (!isnan(v.at(i))){
            double ret = (*active_fn)(v.at(i));
            if (isnan(ret))	{
                vr.push_back(0);
            }
            else {
                vr.push_back(ret);
            }
        }
        else {
            vr.push_back(0);
        }
    }

    return vr;
}

// apply a function to every element of the matrix
Matrix2D<double> np::applyFunction(Matrix2D<double> &m1, double (*active_fn)(double)){
    Matrix2D<double> m3(m1.get_rows(), m1.get_columns());
    for(unsigned int i=0; i < m1.get_rows(); i++){
        for(unsigned int j=0; j < m1.get_columns(); j++){
            double ret = (*active_fn)(m1.get(i, j));
            if (isnan(ret)) {
                m3.set(i, j, 0);
            }
            else {
                m3.set(i, j, ret);
            }
        }
    }
    return m3;
}
