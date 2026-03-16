#include "nn_framework.h"
#include <iostream>
#include <ostream>
#include <vector>

class HebbianNetwork : public NeuralNet {
public:
  Mat X_t;
  HebbianNetwork() {}
  HebbianNetwork(std::vector<int> config, Mat data, std::vector<int> d_conf,
                 double rate)
      : NeuralNet(config, data, d_conf, rate) {
    X_t.rows = X.cols;
    X_t.cols = X.rows;
    std::vector<double> temp;
    for (int i = 0; i < X_t.rows; i++) {
      for (int j = 0; j < X_t.cols; j++) {
        temp.push_back(0);
      }
      X_t.mat.push_back(temp);
      temp.clear();
    }
    for (int i = 0; i < X.rows; i++) {
      for (int j = 0; j < X.cols; j++) {
        X_t.set(j, i, X.mat[i][j]);
      }
    }
  }
  void forward() {
    for (int i = 0; i < Outputs.size(); i++) {
      Outputs[i].mul(X, Layers[i]);
      Outputs[i].add(Biases[i]);
      Outputs[i].apply_activation(&sigmoid);
    }
  }
  void update() {
    Mat delta_weights(Layers[0].rows, Layers[0].cols);
    // Mat delta_bias(Biases[0].rows, Biases[0].cols);
    delta_weights.mul(X_t, Outputs[0]);
    delta_weights.mul(lr);
    Outputs[0].mul(lr);
    Biases[0].add(Outputs[0]);
    Layers[0].add(delta_weights);
    Outputs[0].fill(0);
  }
};
