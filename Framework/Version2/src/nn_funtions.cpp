#include "nn_framework.h"
#include <iostream>
#include <string>

// Essential Functions For Any Layer in the Neural Network
// Layer::Layer(int input, int output, int batch, std::string type = "") {
//  weights.rows = input;
//  weights.cols = output;
//  weights.allocate_mat();
//  weights.fill_rand();
//  grad_weights.rows = input;
//  grad_weights.cols = output;
//  grad_weights.allocate_mat();
//  biases.rows = 1;
//  biases.cols = output;
//  biases.allocate_mat();
//  biases.fill(0);
//  grad_biases.rows = 1;
//  grad_biases.cols = output;
//  grad_biases.allocate_mat();
//  hidden.rows = batch;
//  hidden.cols = output;
//  hidden.allocate_mat();
//  outputs.rows = batch;
//  outputs.cols = output;
//  outputs.allocate_mat();
//  grad_outputs.rows = batch;
//  grad_outputs.cols = output;
//  grad_outputs.allocate_mat();
//  batches = batch;
//  type = type;
//}

void Layer::print(int padding = 0) {
  int init_padding = padding;
  std::cout << std::setw(padding) << "Layer of type " + type << std::endl;

  std::cout << std::setw(padding) << "Weights: " << std::endl;
  weights.print(padding);

  std::cout << std::setw(padding) << "Biases: " << std::endl;
  biases.print(padding);

  std::cout << std::setw(padding) << "Outputs: " << std::endl;
  outputs.print(padding);
}
void Layer::reset() {
  grad_outputs.fill(0);
  hidden.fill(0);
  outputs.fill(0);
  grad_biases.fill(0);
  grad_weights.fill(0);
}

// Essential Functions For Any Sequential Layer In Neural Network
SequentialLayer::SequentialLayer(int input, int output, int batch,
                                 double (*activation)(double),
                                 std::string type = "Sequential") {
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
  type = type;
  activation_funciton = activation;
}
void SequentialLayer::forward(Mat x) {
  hidden.mul(x, weights);
  hidden.add_column_wise(biases);
  outputs.add(hidden);
  outputs.apply_activation(activation_funciton);
}
