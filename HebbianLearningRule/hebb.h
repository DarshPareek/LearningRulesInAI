#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

inline int dot(std::vector<int> arr1, int *arr2, int rows, int columns) {
  int i = 0, res = 0;
  for (i = 0; i < columns - 1; i++) {
    res += arr1[i] * arr2[i];
  }
  return res;
}

class Perceptron {
public:
  int **dataset, rows, columns, output, bias;
  std::vector<int> input_layer;
  Perceptron(int **_dataset, int _rows, int _columns) {
    dataset = _dataset;
    rows = _rows;
    columns = _columns;
    bias = 0;
    std::vector<int> _input_layer(columns - 1);
    input_layer = _input_layer;
    fill(input_layer.begin(), input_layer.end(), 0);
    int output = 0;
  };
  void train() {
    for (int i = 0; i < rows; i++) {
      std::cout << "Weights: ";
      for (int j = 0; j < columns - 1; j++) {
        input_layer[j] += dataset[i][j] * dataset[i][columns - 1];
        std::cout << input_layer[j] << " ";
      }
      std::cout << std::endl;
      bias += dataset[i][columns - 1];
    }
  }
  void train_new() {
    for (int i = 0; i < rows; i++) {
      std::cout << "Weights: ";
      int pred = dataset[i][0] * input_layer[0] +
                 dataset[i][1] * input_layer[1] + bias;
      if (pred > 0)
        pred = 1;
      else
        pred = -1;
      input_layer[0] += 1 * dataset[i][0] * (dataset[i][columns - 1] - pred);
      input_layer[1] += 1 * dataset[i][1] * (dataset[i][columns - 1] - pred);
      bias += 1 * (dataset[i][columns - 1] - pred);
      std::cout << input_layer[0] << " " << input_layer[1] << " " << bias << " "
                << pred;
      std::cout << std::endl;
    }
  }
  void printWeights() {
    std::cout << "Weights: ";
    for (int i = 0; i < columns - 1; i++) {
      std::cout << input_layer[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Bias: " << bias << std::endl;
  }
  int predict(int *input) {
    int pred = dot(input_layer, input, rows, columns);
    pred += bias;
    if (pred > 0) {
      std::cout << "Prediction: " << 1 << std::endl;
      return 1;
    } else {
      std::cout << "Prediction: " << -1 << std::endl;
      return -1;
    }
  }
  int predict_new(int *input) {
    int pred = dot(input_layer, input, rows, columns);
    pred += bias;
    if (pred > 0) {
      std::cout << "Prediction: " << pred << std::endl;
      return 1;
    } else if (pred < 0) {
      std::cout << "Prediction: " << pred << std::endl;
      return -1;
    }
    return 0;
  }
};
