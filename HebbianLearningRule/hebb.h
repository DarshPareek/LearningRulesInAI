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
    // Training
    for (int i = 0; i < rows; i++) {
      std::cout << "Weights: ";
      for (int j = 0; j < columns - 1; j++) {
        input_layer[j] += dataset[i][j] * dataset[i][columns - 1];
        std::cout << input_layer[j] << " ";
      }
      std::cout << std::endl;
      bias += dataset[i][columns - 1];
    }
  };
  void printWeights() {
    std::cout << "Weights: ";
    for (int i = 0; i < columns - 1; i++) {
      std::cout << input_layer[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Bias: " << bias << std::endl;
  }
  void predict(int *input) {
    int pred = dot(input_layer, input, rows, columns);
    pred += bias;
    if (pred > 0)
      std::cout << "Prediction: " << 1 << std::endl;
    else
      std::cout << "Prediction: " << -1 << std::endl;
  }
};
