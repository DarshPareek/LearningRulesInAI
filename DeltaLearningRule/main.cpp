#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <vector>
double cost(double tru, double pred) { return tru - pred; }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double gradSigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
int main() {
  double w1, w2, w3, w4, w5, w6, w7, w8, w9, w10,
      b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, l1, l2, l3, l4, l5, loss = 0,
      total_loss = 0, lr = 0.1;
  w1 = (double)rand() / (RAND_MAX);
  w2 = (double)rand() / (RAND_MAX);
  w3 = (double)rand() / (RAND_MAX);
  w4 = (double)rand() / (RAND_MAX);
  w5 = (double)rand() / (RAND_MAX);
  w6 = (double)rand() / (RAND_MAX);
  w7 = (double)rand() / (RAND_MAX);
  w8 = (double)rand() / (RAND_MAX);
  w9 = (double)rand() / (RAND_MAX);
  w10 = (double)rand() / (RAND_MAX);
  std::vector<std::pmr::vector<int>> inputs = {
      {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}};
  for (int i = 0; i < 100 * 1000; i++) {
    std::cout << "\033[2J\033[1;1H";
    // FORWARD PASS FOR ALL
    total_loss = 0;
    for (int j = 0; j < 4; j++) {
      l1 = sigmoid(inputs[j][0] * w1 + inputs[j][1] * w2 + b1);
      l2 = sigmoid(inputs[j][0] * w3 + inputs[j][1] * w4 + b2);
      l3 = sigmoid(l1 * w5 + l2 * w6 + b3);
      l4 = sigmoid(l1 * w7 + l2 * w8 + b4);
      l5 = sigmoid(l3 * w9 + l4 * w10 + b5);
      loss = cost(inputs[j][2], l5);
      total_loss += loss * loss;
      std::cout << "Forward Pass Loss: " << loss << std::endl;
      // BACKWARD
      double dl_dl5 = (inputs[j][2] - l5);

      double db5_dl5 = gradSigmoid(l3 * w9 + l4 * w10 + b5);
      double dw10_dl5 = gradSigmoid(l3 * w9 + l4 * w10 + b5) * l4;
      double dw9_dl5 = gradSigmoid(l3 * w9 + l4 * w10 + b5) * l3;

      double db4_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4);
      double dw8_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * l2;
      double dw7_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * l1;

      double db3_dl5 = w9 * gradSigmoid(l3 * w9 + l4 * w10 + b5) *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3);
      double dw6_dl5 = w9 * gradSigmoid(l3 * w9 + l4 * w10 + b5) *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) * l2;
      double dw5_dl5 = w9 * gradSigmoid(l3 * w9 + l4 * w10 + b5) *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) * l1;

      double db2_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) * w8 *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * w9 *
                       gradSigmoid(l3 * w9 + l4 * w10 + b5) * w6 *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) *
                       gradSigmoid(inputs[j][0] * w3 + inputs[j][1] * w4 + b2);
      double dw3_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) * w8 *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * w9 *
                       gradSigmoid(l3 * w9 + l4 * w10 + b5) * w6 *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) *
                       gradSigmoid(inputs[j][0] * w3 + inputs[j][1] * w4 + b2) *
                       inputs[j][0];
      double dw4_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) * w8 *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * w9 *
                       gradSigmoid(l3 * w9 + l4 * w10 + b5) * w6 *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) *
                       gradSigmoid(inputs[j][0] * w3 + inputs[j][1] * w4 + b2) *
                       inputs[j][1];
      double db1_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) * w7 *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * w9 *
                       gradSigmoid(l3 * w9 + l4 * w10 + b5) * w5 *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) *
                       gradSigmoid(inputs[j][0] * w1 + inputs[j][1] * w2 + b1);
      double dw1_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) * w7 *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * w9 *
                       gradSigmoid(l3 * w9 + l4 * w10 + b5) * w5 *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) *
                       gradSigmoid(inputs[j][0] * w1 + inputs[j][1] * w2 + b1) *
                       inputs[j][0];
      double dw2_dl5 = w10 * gradSigmoid(l3 * w9 + l4 * w10 + b5) * w7 *
                       gradSigmoid(l1 * w7 + l2 * w8 + b4) * w9 *
                       gradSigmoid(l3 * w9 + l4 * w10 + b5) * w5 *
                       gradSigmoid(l1 * w5 + l2 * w6 + b3) *
                       gradSigmoid(inputs[j][0] * w1 + inputs[j][1] * w2 + b1) *
                       inputs[j][1];
      w1 += lr * dw1_dl5 * dl_dl5;
      w2 += lr * dw2_dl5 * dl_dl5;
      w3 += lr * dw3_dl5 * dl_dl5;
      w4 += lr * dw4_dl5 * dl_dl5;
      w5 += lr * dw5_dl5 * dl_dl5;
      w6 += lr * dw6_dl5 * dl_dl5;
      w7 += lr * dw7_dl5 * dl_dl5;
      w8 += lr * dw8_dl5 * dl_dl5;
      w9 += lr * dw9_dl5 * dl_dl5;
      w10 += lr * dw10_dl5 * dl_dl5;
      b1 += lr * db1_dl5 * dl_dl5;
      b2 += lr * db2_dl5 * dl_dl5;
      b3 += lr * db3_dl5 * dl_dl5;
      b4 += lr * db4_dl5 * dl_dl5;
      b5 += lr * db5_dl5 * dl_dl5;
    }
    std::cout << "Mean Sqaured Loss: " << total_loss / 4 << std::endl;
  }
  int x1, x2, y;
  x1 = 1, x2 = 0, y = 1;
  l1 = sigmoid(x1 * w1 + x2 * w2 + b1);
  l2 = sigmoid(x1 * w3 + x2 * w4 + b2);
  l3 = sigmoid(l1 * w5 + l2 * w6 + b3);
  l4 = sigmoid(l1 * w7 + l2 * w8 + b4);
  l5 = sigmoid(l3 * w9 + l4 * w10 + b5);
  std::cout << "Pass Result: " << (l5 > 0.5 ? 1 : 0) << std::endl;
  std::cout << "Pass Loss: " << y - l5 << std::endl;
  return 0;
}
