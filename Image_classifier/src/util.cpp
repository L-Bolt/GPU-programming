#include "include/util.h"


int seed = time(0);
std::default_random_engine random_engine(seed);

double fns::relu(double x){
    if (x > 0) return x;
    else return (double) 0.0;
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

// apply a function to every element of the vector
std::vector<double> np::applyFunction(std::vector<double> &v, double (*active_fn)(double)){
    std::vector<double> vr(v.size());

    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isnan(v.at(i))) {
            double ret = (*active_fn)(v.at(i));
            if (std::isnan(ret)) {
                vr.at(i) = 0;
            }
            else {
                vr.at(i) = ret;
            }
        }
        else {
            vr.at(i) = 0.0;
        }
    }

    return vr;
}

std::vector<double> np::normalize(std::vector<double> &v){
    std::vector<double> vr(v.size());

    double sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v.at(i);
    }
    assert(sum != 0);

    for (size_t i = 0; i < v.size(); i++) {
        if (sum == 0) {
            vr.at(i) = 0.0;
        }
        else {
            vr.at(i) = v.at(i) / sum;
        }
    }

    return vr;
}

std::vector<double> np::multiply(std::vector<double> &v1, std::vector<double> &v2){
    assert(v1.size() == v2.size());

    std::vector<double> vr(v1.size());
    for (size_t i = 0; i < v1.size(); i++){
        vr.at(i) = v1.at(i) * v2.at(i);
    }

    return vr;
}

void np::multiply(std::vector<double> &v1, double val) {

    for (size_t i = 0; i < v1.size(); i++){
        v1.at(i) = v1.at(i) * val;
    }

}

std::vector<double> np::subtract(std::vector<double> & v1, std::vector<double> & v2){
	assert(v1.size() == v2.size());

	std::vector<double> vr(v1.size());
	for (size_t i = 0; i < v1.size(); i++){
		vr.at(i) = v1.at(i) - v2.at(i);
	}

	return vr;
}

double np::sum(std::vector<double> v1) {
    double sum = 0.0;

    for (size_t i = 0; i < v1.size(); i++) {
        sum += v1.at(i);
    }

    return sum;
}

int np::get_max_class(std::vector<double> &prediction) {
    double max = 0.0;
    int index = 0;
    for (size_t i = 0; i < prediction.size(); i++) {
        if (prediction[i] > max) {
            max = prediction[i];
            index = i;
        }
    }

    return index;
}
