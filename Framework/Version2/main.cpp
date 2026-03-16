#include "src/gnuplot-iostream.h"
#include "src/nn_framework.h"
#include <bits/stdc++.h>
#include <cassert>
#include <iostream>
#include <pybind11/embed.h> // For py::scoped_interpreter
#include <pybind11/numpy.h> // If you are using numpy arrays
#include <pybind11/pybind11.h> // Essential for all pybind11 types (this covers the /*--ignore*/ part)
#include <pybind11/stl.h>
#include <vector>
std::vector<int> config = {784, 36, 16, 8, 10};
std::vector<int> d_conf = {784, 10};
int batch = 2000;
double compress_cost(Mat x) {
  assert(x.rows == 1);
  double s = 0;
  for (int j = 0; j < x.cols; j++) {
    s += x.mat[0][j];
  }
  return s;
}
namespace py = pybind11;
int main(int argc, char *argv[]) {
  gnuplotio::Gnuplot gp;
  gp << "set terminal wxt persist size 800, 600\n";
  gp << "set title 'Loss Plot'\n";
  gp << "set xlabel 'Epoch'\n";
  gp << "set ylabel 'Loss'\n";
  gp << "set grid\n";
  py::scoped_interpreter guard{};
  py::module_ read_data_module = py::module_::import("read_data");
  MnistData mnist(read_data_module,
                  "/home/darsh/devel/intWork/PID008/LearningRulesInAI/"
                  "Framework/pyFiles/mnist_train_clean.csv");
  std::cout << mnist.csv_data.rows << " " << mnist.csv_data.cols << "\n";
  NeuralNet delta(config, mnist.csv_data, d_conf, 1e-1, batch);
  std::cout << delta.X.rows << " " << delta.X.cols << "\n";
  std::cout << delta.Layers[0].rows << " " << delta.Layers[0].cols << "\n";
  std::vector<std::pair<double, double>> epoch_loss;
  delta.Outputs[delta.Outputs.size() - 1].print();
  delta.reset();
  for (int i = 0; i < 5000; i++) {
    delta.reset();
    delta.forward();
    delta.cost_fn();
    double temp = compress_cost(delta.Cost);
    epoch_loss.push_back({i, temp});
    delta.backward();
    std::cout << temp << "\n";
    gp << "plot '-' with lines title 'Live Data'\n";
    gp.send1d(epoch_loss);
    if (epoch_loss.size() > 1) {
      gp << "set xrange [" << epoch_loss.front().first << ":"
         << epoch_loss.back().first << "]\n";
    }
    gp << std::flush;
  }
  delta.forward();
  delta.Outputs[delta.Outputs.size() - 1].print();
  return 0;
}
