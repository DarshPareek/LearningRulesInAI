#include "nn_framework.h"
#include <bits/stdc++.h>
#include <cassert>
#include <cstdlib>
#include <vector>
double cost(double tru, double pred) { return tru - pred; }
double square(double x) { return x * x; }
double w1 = rand_val(0, 1);
double w2 = rand_val(0, 1);
double w3 = rand_val(0, 1);
double w4 = rand_val(0, 1);
double w5 = rand_val(0, 1);
double w6 = rand_val(0, 1);
double w7 = rand_val(0, 1);
double w8 = rand_val(0, 1);
double w9 = rand_val(0, 1);
double w10 = rand_val(0, 1);

void making_smaller() {
  double b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, l1, l2, l3, l4, l5, loss = 0,
         total_loss = 0, lr = 0.01;
  std::vector<Mat> Layers;
  std::vector<std::pmr::vector<int>> inputs = {
      {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}};

  Mat X(4, 2);
  X.mat = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};

  Mat Y(4, 1);
  Y.mat = {{0}, {1}, {1}, {0}};
  Mat lay1(2, 2);
  lay1.set(0, 0, w1);
  lay1.set(1, 0, w2);
  lay1.set(0, 1, w3);
  lay1.set(1, 1, w4);
  Mat lay2(2, 1);
  lay2.set(0, 0, w5);
  lay2.set(1, 0, w6);
  Mat bias1(1, 2);
  bias1.set(0, 0, b1);
  bias1.set(0, 1, b2);
  Mat bias2(1, 1);
  bias2.set(0, 0, b3);

  // Defining
  Mat hid2(4, 1);
  Mat hid1(4, 2);
  Mat out2(4, 1);
  Mat out1(4, 2);
  Mat Cost(1, out2.cols);
  Mat grad_bias2(bias2.rows, bias2.cols);
  Mat grad_bias1(bias1.rows, bias1.cols);
  Mat grad_lay2(lay2.rows, lay2.cols);
  Mat grad_lay1(lay1.rows, lay1.cols);
  Mat grad_loss2(out2.rows, out2.cols);
  Mat grad_loss1(out1.rows, out1.cols);
  for (int i = 0; i < 500 * 1000; i++) {

    // Forward Pass
    hid1.fill(0);
    out1.fill(0);
    hid2.fill(0);
    out2.fill(0);
    hid1.mul(X, lay1);
    hid1.add_column_wise(bias1);
    out1.add(hid1);
    out1.apply_activation(&sigmoid);
    hid2.mul(out1, lay2);
    hid2.add_column_wise(bias2);
    out2.add(hid2);
    out2.apply_activation(&sigmoid);

    // Cost
    grad_loss2.fill(0);
    Cost.fill(0);
    out2.mul(-1);
    out2.add(Y);
    grad_loss2.add(out2);
    out2.apply_activation(&square);
    Cost.squish_rows(out2);
    Cost.mul(0.25);
    Cost.print();

    // BACKWARD
    grad_bias2.fill(0);
    grad_lay2.fill(0);
    grad_bias1.fill(0);
    grad_lay1.fill(0);
    hid2.apply_activation(&grad_sigmoid);
    hid2.dot(grad_loss2);
    grad_bias2.squish_rows(hid2);
    grad_bias2.mul(0.25);
    grad_bias2.mul(lr);
    bias2.add(grad_bias2);
    auto temp = out1.transpose();
    grad_lay2.mul(temp, hid2);
    grad_lay2.mul(0.25);
    grad_lay2.mul(lr);
    lay2.add(grad_lay2);
    hid1.apply_activation(&grad_sigmoid);
    grad_loss1.mul(hid2, lay2.transpose());
    hid1.dot(grad_loss1);
    grad_bias1.squish_rows(hid1);
    grad_bias1.mul(0.25);
    grad_bias1.mul(lr);
    bias1.add(grad_bias1);
    temp = X.transpose();
    grad_lay1.mul(temp, hid1);
    grad_lay1.mul(0.25 * lr);
    lay1.add(grad_lay1);
  }

  // Testing
  out1.fill(0);
  out2.fill(0);
  out1.mul(X, lay1);
  out1.add_column_wise(bias1);
  out1.apply_activation(&sigmoid);
  out2.mul(out1, lay2);
  out2.add_column_wise(bias2);
  out2.apply_activation(&sigmoid);
  out2.print();

  out2.mul(-1);
  lay2.print();
  bias2.print();

  out2.add(Y);
  out2.print();

  Cost.fill(0);
  Cost.squish_rows(out2);
  Cost.print();
}
int main(int argc, char *argv[]) {
  making_smaller();
  return 0;
}
