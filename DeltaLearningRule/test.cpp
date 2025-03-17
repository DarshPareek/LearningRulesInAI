#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

class NN {
public:
  int num_layers;
  std::vector<std::vector<double>> dataset;
  std::vector<std::vector<double>> weights;
  double lr;
  std::vector<double> layer_outs;
  NN(std::vector<int> config, std::vector<std::vector<double>> _dataset,
     double _lr) {
    int i = 0;
    dataset = _dataset;
    lr = _lr;
    std::vector<double> layer;
    num_layers = config.size();
    for (i = 0; i < num_layers; i++) {
      weights.push_back(layer);
    }
    i = 0;
    for (auto &layer : weights) {
      for (int j = 0; j < config[i]; j++) {
        layer.push_back(0.000);
      }
      i++;
    }
  }
  void info() {
    std::cout << "Network has " << num_layers << " layers.\n";
    std::cout << "Printing Weights\n";
    for (int i = 0; i < num_layers; i++) {
      std::cout << "Printing Weights of Layer " << i + 1 << "\n";
      for (auto &w : weights[i]) {
        std::cout << w << "\t";
      }
      std::cout << "\n";
    }
  }
  void forward(int i) {
    // Not Implementing Batches so we dont need matrix multiplication
    // 'i' is for traversing the dataset
    // Layer 1
    double layer1_out =
        weights[0][0] * dataset[i][0] + weights[0][1] * dataset[i][1] +
        weights[0][2] * dataset[i][2] + weights[0][3] * dataset[i][3];
    layer_outs.push_back(layer1_out);
    weights[1][0] = layer1_out;
    std::cout << "Output: " << layer1_out << "\n";
  }
  void backward(int i) {
    double error = pow(layer_outs[i] - dataset[i][4], 2);
    weights[0][0] += lr * (layer_outs[i] - dataset[i][4]) * dataset[i][0];
    weights[0][1] += lr * (layer_outs[i] - dataset[i][4]) * dataset[i][1];
    weights[0][2] += lr * (layer_outs[i] - dataset[i][4]) * dataset[i][2];
    weights[0][3] += lr * (layer_outs[i] - dataset[i][4]) * dataset[i][3];
    std::cout << "error: " << error << "\n";
  }
};
int main() {
  std::vector<std::vector<double>> dataXY = {{1.0, -1.0, 1.0, -1.0, -1.0},
                                             {1.0, 1.0, 1.0, 1.0, 1.0},
                                             {1.0, 1.0, 1.0, -1.0, -1.0},
                                             {1.0, -1.0, -1.0, 1.0, -1.0}};
  NN net({4, 1}, dataXY, 0.25);
  net.info();
  // for (int j = 0; j < 1; j++) {
  for (int i = 0; i < 4; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    net.forward(i);
    net.backward(i);
    // std::cout << "\033[2J\033[1;1H";
    net.info();
  }
  // }
  std::cout << net.weights[0][0] * -1 + net.weights[0][1] * -1 +
                   net.weights[0][2] * 1 + net.weights[0][3] * 1
            << "\n";
  return 0;
}
