#include "include/Cnn.h"

CNN::CNN(Shape3D input_dim, Shape kernel_size, Shape pool_size, int hidden_layer_nodes, int output_dim, Matrix3D<double> &conv_kernel) {
	assert(input_dim.rows > kernel_size.rows && input_dim.columns > kernel_size.columns);
	assert(input_dim.rows - kernel_size.rows + 1 > pool_size.rows && input_dim.columns - kernel_size.columns + 1 > pool_size.columns);

	this->kernel = conv_kernel;
	this->pool_window = pool_size;
	this->output_dim = output_dim;
	this->hidden_layer_nodes = hidden_layer_nodes;

	int x = ((input_dim.rows - kernel_size.rows + 1) / pool_size.rows);			 // output rows of feature vector.
	int y = ((input_dim.columns - kernel_size.columns + 1) / pool_size.columns); // output cols of feature vector.

	Matrix2D<double> Weights0(x * y, hidden_layer_nodes, true);
	Matrix2D<double> Weights1(hidden_layer_nodes, output_dim, true);
	this->weights.push_back(Weights0);
	this->weights.push_back(Weights1);

	Matrix2D<double> Bias0(hidden_layer_nodes, 1, true);
	Matrix2D<double> Bias1(output_dim, 1, true);
	this->biases.push_back(Bias0);
	this->biases.push_back(Bias1);
}

float CNN::get_training_percentage() {
	int total = this->epochs * this->training_size;

	return (this->iteration / (float) total) * 100.0;
}

void CNN::train(std::vector<Image> &Xtrain, std::vector<std::vector<double>> &Ytrain, double learning_rate, int epochs) {
	assert(Xtrain.size() == Ytrain.size());

	this->training_size = (int) Xtrain.size();

	this->trained = false;
	this->epochs = epochs;

	for (int epoch = 1; epoch <= epochs; epoch++) {
		double error = 0.0;

		std::cout << "Running epoch: " << epoch << std::endl;
		for (size_t it = 0; it < Xtrain.size(); it++) {
			this->iteration = this->iteration + 1;

			if (this->stop) {
				std::cout << "stopped training" << std::endl;
				return;
			}

			std::vector<std::vector<double>> a;
			std::vector<std::vector<double>> z;

			forward_propagate(Xtrain[it], a, z);

			error += cross_entropy(a.at(1), Ytrain[it]);
			std::vector<double> dZ2 = np::subtract(a.at(1), Ytrain[it]);
			back_propagate(dZ2, a, z, Xtrain[it], fns::relu_gradient, learning_rate);

		}
		this->epoch = this->epoch + 1.0;
		std::cout << "epoch: " << epoch << " error: " << (error / Xtrain.size()) << std::endl;
	}
	this->trained = true;
}

/*	Calculate the Validation error over the validation set.
 *	So only do forward_propagate for each batch without updating weights
 *	each iteration
 */
double CNN::validate(std::vector<Image> &Xval, std::vector<std::vector<double>> &Yval) {
	assert(Xval.size() == Yval.size());

	this->validated = false;

	double error = 0;
	for (size_t it = 0; it < Xval.size(); it++) {
		if (this->stop) {
			return 0.0;
		}

		std::vector<std::vector<double>> a;
		std::vector<std::vector<double>> z;

		forward_propagate(Xval[it], a, z);
		std::vector<double> prediction = a.at(1);
		int classified = np::get_max_class(prediction);
		if (classified == Xval[it].get_class()) {
			this->correctly_classified++;
		}

		error += cross_entropy(a.at(1), Yval[it]);
	}

	std::cout << "Images correctly classified: " << this->correctly_classified << '\n';
	std::cout << " error: " << (error / Xval.size()) << std::endl;
	this->validated = true;

	return (error / Xval.size());
}

/*
 * Forward propagate the provided inputs through the Convolution Neural Network
 * and the outputs of each Dense layer is appended as vector to activations.
 * Output of convolution layer(matrix) is appended to conv_activations
 */
void CNN::forward_propagate(Image &input, std::vector<std::vector<double>> &a, std::vector<std::vector<double>> &z) {
	// Convolve the normalized matrix of the input image and apply the ReLu activation function.
	// This transforms the 3D image into a 2D matrix.
	std::vector<double> flattened_pool;
	if (!input.processed) {
		Matrix2D<double> convolved = input.normalize(0.5, 0.5).convolve(this->kernel);

		// Take the max pooling of the convolved image.
		Matrix2D<double> pooled = convolved.max_pooling(this->pool_window);
		flattened_pool = pooled.flatten_to_vector(0);
	}
	else {
		flattened_pool = input.preprocessed_data;
	}

	std::vector<double> Z1 = this->weights.at(0).transpose().dot(flattened_pool);
	assert((int)Z1.size() == this->biases.at(0).get_rows());
	for (size_t i = 0; i < Z1.size(); i++) {
		Z1.at(i) = Z1.at(i) + this->biases.at(0).get(i, 0);
	}

	std::vector<double> A1 = np::applyFunction(Z1, fns::relu);

	std::vector<double> Z2 = this->weights.at(1).transpose().dot(A1);
	assert((int)Z2.size() == this->biases.at(1).get_rows());
	for (size_t i = 0; i < Z2.size(); i++) {
		Z2.at(i) = Z2.at(i) + this->biases.at(1).get(i, 0);
	}

	std::vector<double> A2 = np::applyFunction(Z2, fns::softmax);

	for (size_t i = 0; i < A2.size(); i++) {
		if (std::isnan(A2[i])) {
			A2[i] = 1.0;
		}
	}
	A2 = np::normalize(A2);

	a.push_back(A1);
	a.push_back(A2);
	z.push_back(Z1);
	z.push_back(Z2);
}

/*
 * Compute deltas of each layer and return the same.
 * delta_L: delta of the final layer, computed and passed as argument
 * activations: Output of each layer after applying activation function
 * Assume that all layers have same activation function except that of final layer.
 * active_fn_der: function pointer for the derivative of activation function,
 * which takes activation of the layer as input
 */
void CNN::back_propagate(std::vector<double> &dZ2,
						 std::vector<std::vector<double>> &a,
						 std::vector<std::vector<double>> &z,
						 Image &input, double (*active_fn_der)(double), double learning_rate) {

	std::vector<double> X;
	if (!input.processed) {
		Matrix2D<double> convolved = input.normalize(0.5, 0.5).convolve(this->kernel);

		// Take the max pooling of the convolved image.
		Matrix2D<double> pooled = convolved.max_pooling(this->pool_window);
		X = pooled.flatten_to_vector(0);
	}
	else {
		X = input.preprocessed_data;
	}

	Matrix2D<double> dz2_mat(this->output_dim, 1, dZ2);
	Matrix2D<double> a1_mat(1, this->hidden_layer_nodes, a.at(0));
	Matrix2D<double> dW2 = dz2_mat.dot(a1_mat);

	double db2 = 0.0;
	for (size_t i = 0; i < dZ2.size(); i++) {
		db2 += dZ2[i];
	}
	db2 *= (1.0 / 10.0);
	db2 *= learning_rate;

	std::vector<double> dZ1 = this->weights.at(1).dot(dZ2);
	std::vector<double> dz_deriv = np::applyFunction(z.at(0), active_fn_der);
	dZ1 = np::multiply(dZ1, dz_deriv);

	Matrix2D<double> dz1_mat(dZ1.size(), 1, dZ1);
	Matrix2D<double> X_mat(1, X.size(), X);
	Matrix2D<double> dW1 = dz1_mat.dot(X_mat);
	for (int i = 0; i < dW1.get_rows(); i++) {
		for (int j = 0; j < dW1.get_columns(); j++) {
			dW1.set(i, j, dW1.get(i, j) * (1.0 / 10.0));
		}
	}

	double db1 = 0.0;
	for (size_t i = 0; i < dZ1.size(); i++) {
		db1 += dZ1[i];
	}
	db1 *= (1.0 / 10.0);
	db1 *= learning_rate;

	// Update weights and biases.
	dW1 = dW1.transpose();
	for (int i = 0; i < dW1.get_rows(); i++) {
		for (int j = 0; j < dW1.get_columns(); j++) {
			dW1.set(i, j, dW1.get(i, j) * learning_rate);
			this->weights.at(0).set(i, j, this->weights.at(0).get(i, j) - dW1.get(i, j));
		}
	}

	for (int i = 0; i < this->biases.at(0).get_rows(); i++) {
		for (int j = 0; j < this->biases.at(0).get_columns(); j++) {
			this->biases.at(0).set(i, j, this->biases.at(0).get(i, j) * db1);
		}
	}

	dW2 = dW2.transpose();
	for (int i = 0; i < dW2.get_rows(); i++) {
		for (int j = 0; j < dW2.get_columns(); j++) {
			dW2.set(i, j, dW2.get(i, j) * learning_rate);
			this->weights.at(1).set(i, j, this->weights.at(1).get(i, j) - dW2.get(i, j));
		}
	}

	for (int i = 0; i < this->biases.at(1).get_rows(); i++) {
		for (int j = 0; j < this->biases.at(1).get_columns(); j++) {
			this->biases.at(1).set(i, j, this->biases.at(1).get(i, j) * db2);
		}
	}
}

/*
 * Calculate cross entropy loss if the predictions and true values are given
 */
double CNN::cross_entropy(std::vector<double> &ypred, std::vector<double> &ytrue) {
	assert(ypred.size() == ytrue.size());

	std::vector<double> z = np::applyFunction(ypred, log);
	z = np::multiply(z, ytrue);

	double error = 0.0;
	for (size_t i = 0; i < z.size(); i++) {
		error += z[i];
	}

	return (-error);
}
