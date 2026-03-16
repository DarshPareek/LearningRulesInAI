#include "../include/SequentialLayer.h"
#include "../include/Functions.h"
SequentialLayer::SequentialLayer(int input, int output, int batch,
                                 double (*activation)(double), std::string t) {
  weights.rows = input;
  weights.cols = output;
  weights.allocate_mat();
  weights.fill_rand();
  grad_weights.rows = input;
  grad_weights.cols = output;
  grad_weights.allocate_mat();
  biases.rows = 1;
  biases.cols = output;
  biases.allocate_mat();
  biases.fill(0);
  grad_biases.rows = 1;
  grad_biases.cols = output;
  grad_biases.allocate_mat();
  hidden.rows = batch;
  hidden.cols = output;
  hidden.allocate_mat();
  outputs.rows = batch;
  outputs.cols = output;
  outputs.allocate_mat();
  grad_outputs.rows = batch;
  grad_outputs.cols = output;
  grad_outputs.allocate_mat();
  batches = batch;
  type = t;
  activation_funciton = activation;
}
void SequentialLayer::forward(Mat x) {
  hidden.mul(x, weights);
  hidden.add_column_wise(biases);
  outputs.add(hidden);
  outputs.apply_activation(activation_funciton);
}
void SequentialLayer::backward(Mat x) {
  hidden.apply_activation(&nn::grad_sigmoid);
  hidden.dot(grad_outputs);
  grad_biases.squish_rows(hidden);
  grad_weights.mul(x, hidden);
}
void SequentialLayer::update(std::vector<double> params) {
  double lr = params[0];
  grad_biases.mul(lr);
  grad_weights.mul(lr);
  biases.add(grad_biases);
  weights.add(grad_weights);
}
