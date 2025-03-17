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
    weights[0][0] += dataset[i][0] * weights[0][0];
    weights[0][1] += dataset[i][1] * weights[0][1];
    double layer1_out = std::max(0.0, (weights[0][0] + weights[0][1]));
    layer_outs.push_back(layer1_out);
    // Layer 2
    weights[1][0] += layer1_out * weights[1][0];
    weights[1][1] += layer1_out * weights[1][1];
    double layer2_out = std::max(0.0, (weights[1][0] + weights[1][1]));
    layer_outs.push_back(layer2_out);
    // Layer 3
    weights[2][0] += layer2_out * weights[2][0];
    weights[2][1] += layer2_out * weights[2][1];
    weights[2][2] += layer2_out * weights[2][2];
    weights[2][3] += layer2_out * weights[2][3];
    double layer3_out = std::max(
        0.0, (weights[2][0] + weights[2][1] + weights[2][2] + weights[2][3]));
    layer_outs.push_back(layer3_out);
    // Output
    weights[3][0] += layer3_out += weights[3][0];
    double layer4_out = std::max(0.0, weights[3][0]);
    layer_outs.push_back(layer4_out);
  }
  void backward(int i) {
    // Layer 4
    double deltaW_4 = lr * pow((layer_outs[3] - dataset[i][2]), 2);
    weights[3][0] += deltaW_4 * layer_outs[2];
    double layer4_out = std::max(0.0, weights[3][0]);
    // Layer 3
    double deltaW_3 = lr * pow((layer_outs[2] - layer4_out), 2);
    weights[2][0] += deltaW_3 * layer_outs[1];
    weights[2][1] += deltaW_3 * layer_outs[1];
    weights[2][2] += deltaW_3 * layer_outs[1];
    weights[2][3] += deltaW_3 * layer_outs[1];
    double layer3_out = std::max(
        0.0, (weights[2][0] + weights[2][1] + weights[2][2] + weights[2][3]));
    // Layer 2
    double deltaW_2 = lr * pow((layer_outs[1] - layer3_out), 2);
    weights[1][0] += deltaW_2 * layer_outs[0];
    weights[1][1] += deltaW_2 * layer_outs[0];
    double layer2_out = std::max(0.0, (weights[1][0] + weights[1][1]));
    // Layer 1
    double deltaW_1 = lr * pow((layer_outs[0] - layer2_out), 2);
    weights[0][0] += deltaW_1 * dataset[i][0];
    weights[0][1] += deltaW_1 * dataset[i][1];
  }
};
int main() {
  std::vector<std::vector<double>> dataXY = {
      {1.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};
  NN net({2, 2, 4, 1}, dataXY, 0.01);
  net.info();
  for (int j = 0; j < 5; j++) {
    for (int i = 0; i < 4; i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      net.forward(i);
      net.backward(i);
      std::cout << "\033[2J\033[1;1H";
      net.info();
    }
  }
  return 0;
}
