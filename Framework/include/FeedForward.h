#ifndef FEED_FORWARD_H
#include "Layer.h"
#include "Matrix.h"
#include <memory>
class NeuralNet {
public:
  std::vector<std::unique_ptr<Layer>> Layers;
  Mat Cost;
  Mat dataset;
  Mat X;
  Mat Y;
  int batch;
  int flag = 0;
  double tau1, L, lr, tau2 = 0.5;
  std::vector<double> params;
  void find_L();
  void set_params(int n);
  void update();
  NeuralNet(Mat data, std::vector<int> d_conf, double rate, int batch);
  void add(std::unique_ptr<Layer> l);
  void print(int padding = 0);
  void reset();
  void forward();
  void cost_fn();
  void backward();
};
#endif // !FEED_FORWARD_H
