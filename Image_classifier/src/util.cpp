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
    if (std::isnan(x)) return 0;
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
        if (!std::isnan(v.at(i))){
            double ret = (*active_fn)(v.at(i));
            if (std::isnan(ret))	{
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
