#include "hebb.h"
int main() {
  int data[4][3] = {
      {1, 1, 1}, {1, -1, 1}, {-1, 1, 1}, {-1, -1, -1}}; // Basic AND gate
  int input_layer[] = {0, 0}, input_size = 2, output_layer[] = {0},
      output_size = 1, bias = 0;
  int dataset_rows = 4, dataset_cols = 3,
      i = 0; // Hebbian Rule states that neurons that wire together fire
  int **dataset = new int *[dataset_rows];
  for (i = 0; i < dataset_rows; i++) {
    dataset[i] = new int[dataset_cols];
  }
  for (int i = 0; i < dataset_rows; i++) {
    for (int j = 0; j < dataset_cols; j++) {
      dataset[i][j] = data[i][j];
    }
  }
  Perceptron model = Perceptron(dataset, 4, 3);
  model.printWeights();
  model.train_new();
  int arr[2] = {-1, 1};
  model.predict(arr);
  return 0;
}
