#include "../include/Functions.h"
#include <cmath>
#include <random>
namespace nn {
double square(double x) { return x * x; }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double grad_sigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
double rand_val(int low, int high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distr(low, high);
  return (double)distr(gen);
}
} // namespace nn
