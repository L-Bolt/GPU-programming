#include "include/Cnn.h"


double multiply(Matrix2D<double> & m1, Matrix2D<double> & m2, int xslice, int yslice);



CNN::CNN(Shape3D input_dim, Shape kernel_size, Shape pool_size, int hidden_layer_nodes, int output_dim) {
	assert(input_dim.rows > kernel_size.rows && input_dim.columns > kernel_size.columns);
	assert(input_dim.rows - kernel_size.rows + 1 > pool_size.rows && input_dim.columns - kernel_size.columns + 1 > pool_size.columns);

	this->kernel = Matrix3D<double>(kernel_size.rows, kernel_size.columns, 3, true);
	this->pool_window = pool_size;

	int x = ((input_dim.rows - kernel_size.rows + 1) / pool_size.rows); // output rows of feature vector.
	int y = ((input_dim.columns - kernel_size.columns + 1) / pool_size.columns); // output cols of feature vector.

	Matrix2D<double> Weights0(x * y + 1, hidden_layer_nodes, true);
	Matrix2D<double> Weights1(hidden_layer_nodes + 1, output_dim, true);
	this->weights.push_back(Weights0);
	this->weights.push_back(Weights1);
}

void CNN::train(std::vector<Image> &Xtrain,
                std::vector<std::vector<double>> &Ytrain,
                double learning_rate, int epochs) {

	assert(Xtrain.size() == Ytrain.size());

	for (int epoch = 1; epoch <= epochs; epoch++) {
		double error = 0.0;
		for (size_t it = 0; it < Xtrain.size(); it++) {
			std::vector<Matrix2D<double>> conv_activations(2);
			std::vector<std::vector<double>> activations(3);

			forward_propagate(Xtrain[it], conv_activations, activations);
			error += cross_entropy(activations.back(), Ytrain[it]);

			std::vector<double> delta_L = np::subtract(activations.back(), Ytrain[it]);

			back_propagate(delta_L, conv_activations, activations, Xtrain[it], fns::relu_gradient, learning_rate);

			it += 1;
		}
		std::cout << "epoch: " << epoch << " error: " << (error / Xtrain.size()) << std::endl ;
	}
}

double CNN::validate(std::vector<Image> &Xval, std::vector<std::vector<double>> &Yval) {
	/*	Calculate the Validation error over the validation set.
	 *	So only do forward_propagate for each batch without updating weights
	 *	each iteration
	*/
	assert(Xval.size() == Yval.size());
	unsigned int it = 1;
	double error = 0;
	while (it <= Xval.size()){
		std::vector<Matrix2D<double>> conv_activations(2);
		std::vector<std::vector<double>> activations(3);

		forward_propagate(Xval[it], conv_activations, activations);
		error += cross_entropy(activations.back(), Yval[it]);

		it += 1;
	}
	std::cout << " error: " << (error / Xval.size()) << std::endl;

	return (error / Xval.size());
}

/*
 * Forward propagate the provided inputs through the Convolution Neural Network
 * and the outputs of each Dense layer is appended as vector to activations.
 * Output of convolution layer(matrix) is appended to conv_activations
 */
void CNN::forward_propagate(Image &input, std::vector<Matrix2D<double>> &conv_activations, std::vector<std::vector<double>> &activations) {
	// Convolve the normalized matrix of the input image and apply the ReLu activation function.
	// This transforms the 3D image into a 2D matrix.
	Matrix2D<double> convolved = input.normalize().convolve(this->kernel).applyFunction(fns::relu);

	// Take the max pooling of the convolved image.
	Matrix2D<double> pooled = convolved.max_pooling(this->pool_window);

	std::vector<double> flattened_pool = pooled.flatten_to_vector(1);
	//	append 1s to inputs and to output of every layer (for bias)
	flattened_pool[flattened_pool.size()] = 1;

	for (int i = 0; i < pooled.get_rows(); i++) {
		for (int j = 0; j < pooled.get_columns(); j++) {
			pooled.set(i, j, 1.0);
		}
	}

	// Output of convolution layer and pooling layer(matrix) is appended to conv_activations.
	conv_activations.push_back(pooled);

	//	hidden layer
	Matrix2D<double> W0 = weights[0].transpose();
	std::vector<double> hidden = W0.dot(flattened_pool);
	hidden = np::applyFunction(hidden, fns::relu);
	hidden.push_back(1);

	activations[0] = flattened_pool;
	// output layer
	Matrix2D<double> W1 = weights[1].transpose();
	std::vector<double> output = W1.dot(hidden);
	output = np::applyFunction(output, fns::softmax);
	output = np::normalize(output);

	activations[1] = std::move(hidden);
	activations[2] = std::move(output);
}

/*
 * Compute deltas of each layer and return the same.
 * delta_L: delta of the final layer, computed and passed as argument
 * activations: Output of each layer after applying activation function
 * Assume that all layers have same activation function except that of final layer.
 * active_fn_der: function pointer for the derivative of activation function,
 * which takes activation of the layer as input
 */
void CNN::back_propagate(std::vector<double> &delta_L, std::vector<Matrix2D<double>> &conv_activations, std::vector<std::vector<double>> &activations, Image &input, double (*active_fn_der)(double), double learning_rate) {

	std::vector<double> delta_h = weights[1].dot(delta_L);
	std::vector<double> active = np::applyFunction(activations[1], active_fn_der);

	delta_h = np::multiply(delta_h, active);

	// don't compute last layer
	std::vector<double> delta_x = weights[0].dot(delta_h, 1);

	active = np::applyFunction(activations[0], active_fn_der);
	delta_x = np::multiply(delta_x, active);

	Matrix2D<double> delta_conv(conv_activations[0].get_rows(), conv_activations[0].get_columns(), false);

	// some weird matrix2d construction.
	int counter = 0;
	for (int r = 0; r < conv_activations[0].get_rows(); r++) {
		for (int c = 0; c < conv_activations[0].get_columns(); c++) {
			if (conv_activations[0].get(r, c) == 1.0) {
				delta_conv.set(r, c, delta_x.at(counter));
				counter++;
			}
		}
	}

	Matrix2D<double> dW0(activations[0], delta_h, 1);
		// last column has to be sliced off
	Matrix2D<double> dW1(activations[1], delta_L);
	dW0 * learning_rate;
	dW1 * learning_rate;

	weights[0] - dW0;
	weights[1] - dW1;

	// TODO update kernel voor 3d image.
	for (int i = 0; i < kernel.get_rows(); i++){
		for (int j = 0; j < kernel.get_columns(); j++){
			for (int k = 0; k < 3; k++) {
				Matrix2D<double> plane = input.normalize().get_plane(k);
				kernel.set(i, j, k, multiply(delta_conv, plane, i, j));
			}
		}
	}
}

/*
 * Calculate cross entropy loss if the predictions and true values are given
 */
double CNN::cross_entropy(std::vector<double> &ypred, std::vector<double> &ytrue) {
	assert(ypred.size() == ytrue.size());
	std::vector<double> z = np::applyFunction(ypred, log);

	for (size_t i = 0; i < z.size(); i++) {
        z.at(i) = z.at(i) * ytrue.at(i);
    }

	double error = std::reduce(z.begin(), z.end());

	return (-error);
}


double multiply(Matrix2D<double> & m1, Matrix2D<double> & m2, int xslice, int yslice) {
	assert(m2.get_rows() >= m1.get_rows() && m2.get_columns() >= m1.get_columns());

	double accumulator = 0;
	for (int i = 0; i < m1.get_rows(); i++){
		for (int j = 0; j < m1.get_columns(); j++){
			accumulator += (m1.get(i,j) * m2.get(xslice + i, yslice + j));
		}
	}
	return accumulator;
}
