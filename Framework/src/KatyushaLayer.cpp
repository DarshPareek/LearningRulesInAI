#include "../include/KatyushaLayer.h"
#include "../include/Functions.h"
KatyushaLayer::KatyushaLayer(int input, int output, int batch,
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
  // setting initial for x, y, z, and mu
  y_weights = weights;
  z_weights = weights;
  x_snapshot_weights = weights;
  temp_weights = weights;
  mu_snapshot_weights = grad_weights;
  y_biases = biases;
  z_biases = biases;
  x_snapshot_biases = biases;
  temp_biases = biases;
  mu_snapshot_biases = biases;
  y_avg_biases = biases;
  y_avg_weights = weights;
  y_avg_biases.fill(0);
  y_avg_weights.fill(0);
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
void KatyushaLayer::backward(Mat x) {
  hidden.apply_activation(&nn::grad_sigmoid);
  hidden.dot(grad_outputs);
  grad_biases.squish_rows(hidden);
  grad_weights.mul(x, hidden);
}
void KatyushaLayer::set_to_temp() {
  temp_weights.mat = weights.mat;
  temp_biases.mat = biases.mat;
}
void KatyushaLayer::set_to_y() {
  weights.mat = y_weights.mat;
  biases.mat = y_biases.mat;
}
void KatyushaLayer::set_to_x() {
  weights.mat = x_snapshot_weights.mat;
  biases.mat = x_snapshot_biases.mat;
}
void KatyushaLayer::set_to_z() {
  weights.mat = z_weights.mat;
  biases.mat = z_biases.mat;
}
void KatyushaLayer::set_mu_grad() {
  mu_snapshot_weights.mat = grad_weights.mat;
  mu_snapshot_biases.mat = grad_biases.mat;
}
void KatyushaLayer::set_normal() {
  weights.mat = temp_weights.mat;
  biases.mat = temp_biases.mat;
}
void KatyushaLayer::forward(Mat x) {
  hidden.mul(x, weights);
  hidden.add_column_wise(biases);
  outputs.add(hidden);
  outputs.apply_activation(activation_funciton);
}
void KatyushaLayer::update(std::vector<double> params) {
  double t1 = params[0];
  double t2 = params[1];
  double lr = params[2];
  double flag = params[3];
  if (flag == 0.0) {
    set_mu_grad();
    y_avg_biases.fill(0);
    y_avg_weights.fill(0);
  }
  if (flag == 1.0) {
    z_weights.mul(t1);
    x_snapshot_weights.mul(t2);
    y_weights.mul((1 - t1 - t2));
    temp_weights.fill(0);
    temp_weights.add(z_weights);
    temp_weights.add(y_weights);
    temp_weights.add(x_snapshot_weights);
    weights = temp_weights;
    z_weights.mul(1.0 / t1);
    x_snapshot_weights.mul(1.0 / t2);
    y_weights.mul(1.0 / (1 - t1 - t2));
    z_biases.mul(t1);
    x_snapshot_biases.mul(t2);
    y_biases.mul((1 - t1 - t2));
    temp_biases.fill(0);
    temp_biases.add(z_biases);
    temp_biases.add(y_biases);
    temp_biases.add(x_snapshot_biases);
    biases = temp_biases;
    z_biases.mul(1.0 / t1);
    x_snapshot_biases.mul(1.0 / t2);
    y_biases.mul(1.0 / (1 - t1 - t2));
  }
  if (flag == 2.0) {
    grad_weights.mul(-1 * lr);
    z_weights.add(grad_weights);
    grad_biases.mul(-1 * lr);
    z_biases.add(grad_biases);
    y_weights = weights;
    y_biases = biases;
    grad_weights.mul(-1.0 / lr);
    grad_biases.mul(-1.0 / lr);
    y_avg_weights.add(y_weights);
    y_avg_biases.add(y_biases);
  }
  if (flag == 3.0) {
    y_avg_weights.mul(1.0 / 4);
    y_avg_biases.mul(1.0 / 4);
    x_snapshot_weights = y_avg_weights;
    x_snapshot_biases = y_avg_biases;
  }
}
