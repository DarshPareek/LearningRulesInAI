#include <iostream>
#include <vector>

class DeltaNetwork : public NeuralNet {
public:
  Mat X_t;
  Mat Cost;
  std::vector<Mat> grad_layers;
  std::vector<Mat> grad_biases;
  std::vector<Mat> grad_outputs;
  DeltaNetwork(std::vector<int> config, Mat data, std::vector<int> d_conf,
               double rate)
      : NeuralNet(config, data, d_conf, rate) {
    this->alloc_cost();
    this->alloc_inputs();
    for (int i = 0; i < config.size() - 1; i++) {
      auto matW = Mat(config[i], config[i + 1]);
      matW.fill(0);
      auto matB = Mat(data.rows, config[i + 1]);
      matB.fill(0);
      auto matO = Mat(data.rows, config[i + 1]);
      matO.fill(0);
      grad_layers.push_back(matW);
      grad_biases.push_back(matB);
      grad_outputs.push_back(matO);
    }
  }
  void alloc_inputs() {
    X_t.rows = X.cols;
    X_t.cols = X.rows;
    X_t.allocate_mat();
    for (int i = 0; i < X.rows; i++) {
      for (int j = 0; j < X.cols; j++) {
        X_t.set(j, i, X.mat[i][j]);
      }
    }
  }
  void alloc_cost() {
    Cost.rows = Y.rows;
    Cost.cols = Y.cols;
    Cost.allocate_mat();
    Cost.fill(0);
  }
  void forward() {
    for (int i = 0; i < Outputs.size(); i++) {
      Outputs[i].mul(X, Layers[i]);
      Outputs[i].add(Biases[i]);
      Outputs[i].apply_activation(&sigmoid);
    }
  }
  void cost_fn() {
    Cost.add(Outputs[Outputs.size() - 1]);
    Cost.mul(-1);
    Cost.add(Y);
    Cost.dot(Cost);
    Cost.squish_rows();
    Cost.mul((double)1 / Y.rows);
  }
  void backward() {
    // Cost.fill(0);
    // Cost.add(Outputs[Outputs.size() - 1]);
    // Cost.mul(-1);
    // Cost.add(Y);
    for (int i = Outputs.size() - 1; i > -1; i--) {
      grad_outputs[i].add(Outputs[i]);
      grad_outputs[i].mul(-1);
      grad_outputs[i].add(Y);
      grad_outputs[i].mul(2);
      Outputs[i].apply_activation(&grad_sigmoid);
      grad_outputs[i].dot(Outputs[i]);
      grad_outputs[i].mul(lr);
      grad_outputs[i].squish_rows();
      X_t.squish_columns();
      X_t.mul(X_t, grad_outputs[i]);
      Layers[i].add(X_t);
      Biases[i].add(grad_outputs[i].mat[0]);
    }
    this->reset();
  }
  void reset() {
    this->alloc_cost();
    for (int i = Outputs.size() - 1; i > -1; i--) {
      Outputs[i].fill(0);
      grad_outputs[i].rows = Outputs[i].rows;
      grad_outputs[i].cols = Outputs[i].cols;
      grad_outputs[i].allocate_mat();
      grad_outputs[i].fill(0);
    }
  }
};
