#include "../include/FeedForward.h"
#include "../include/Functions.h"
#include <cstddef>
#include <iostream>
NeuralNet::NeuralNet(Mat data, std::vector<int> d_conf, double rate,
                     int batch) {
  batch = batch;
  lr = rate;
  dataset.rows = data.rows;
  dataset.cols = data.cols;
  dataset.mat = data.mat;
  // Loading Samples
  int inputs = d_conf[0];
  int outputs = d_conf[1];
  X.cols = inputs;
  Y.cols = outputs;
  X.rows = dataset.rows;
  Y.rows = dataset.rows;
  for (int i = 0; i < dataset.rows; i++) {
    std::vector<double> x;
    std::vector<double> y;
    for (int j = 0; j < dataset.cols; j++) {
      if (j < inputs)
        x.push_back(dataset.get(i, j));
      else
        y.push_back(dataset.get(i, j));
    }
    X.mat.push_back(x);
    Y.mat.push_back(y);
    x.clear();
    y.clear();
  }
  Cost.rows = 1;
  Cost.cols = d_conf[1];
  Cost.allocate_mat();
}
void NeuralNet::add(std::unique_ptr<Layer> l) {
  Layers.push_back(std::move(l));
}
void NeuralNet::print(int padding) {
  for (size_t i = 0; i < Layers.size(); i++) {
    Layers[i]->print(padding);
  }
}
void NeuralNet::reset() {
  for (size_t j = 0; j < Layers.size(); j++) {
    Layers[j]->reset();
  }
}
void NeuralNet::forward() {
  Layers[0]->forward(X);
  for (size_t i = 1; i < Layers.size(); i++) {
    Layers[i]->forward(Layers[i - 1]->outputs);
  }
}
void NeuralNet::cost_fn() {
  auto netowrk_output = Layers[Layers.size() - 1]->outputs;
  netowrk_output.mul(-1);
  netowrk_output.add(Y);
  Layers[Layers.size() - 1]->grad_outputs.add(netowrk_output);
  netowrk_output.apply_activation(&nn::square);
  Cost.squish_rows(netowrk_output);
  Cost.mul((double)1 / X.rows);
}
void NeuralNet::backward() {
  for (int k = Layers.size() - 1; k > -1; k--) {
    std::cout << k << "\n";
    Mat t;
    if (k != 0) {
      t = Layers[k - 1]->outputs.transpose();
    } else {
      t = X.transpose();
    }
    Layers[k]->backward(t);
    if (k != 0)
      Layers[k - 1]->grad_outputs.mul(Layers[k]->hidden,
                                      Layers[k]->weights.transpose());
  }
}
void NeuralNet::find_L() {
  Mat X_norm = X;
  X_norm.mul(1.00 / 255);
  Mat res(X_norm.cols, X.cols);
  res.mul(X_norm.transpose(), X);
  Mat v(X.cols, 1);
  v.fill(1);
  v.norm();
  Mat l_est_mat(1, 1);
  Mat w(X.cols, 1);
  for (int i = 0; i < 100; i++) {
    w.fill(0);
    w.mul(res, v);
    auto vT = v.transpose();
    l_est_mat.fill(0);
    l_est_mat.mul(vT, w);
    w.norm();
    v = w;
  }
  L = l_est_mat.mat[0][0];
}
void NeuralNet::set_params(int n) {
  tau1 = 2.0 / (n + 4);
  lr = 1.0 / (3 * tau1 * L);
  params.clear();
  params.push_back(tau1);
  params.push_back(tau2);
  params.push_back(lr);
  params.push_back(flag);
  params.push_back(L);
}
void NeuralNet::update() {
  for (size_t i = 0; i < Layers.size(); i++) {
    Layers[i]->update(params);
  }
}
