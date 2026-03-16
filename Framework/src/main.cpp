#include "../include/DataLoader.h"
#include "../include/FeedForward.h"
#include "../include/Functions.h"
#include "../include/KatyushaLayer.h"
#include "../include/gnuplot-iostream.h"
#include <bits/stdc++.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <pybind11/embed.h> // For py::scoped_interpreter
#include <pybind11/numpy.h> // If you are using numpy arrays
#include <pybind11/pybind11.h> // Essential for all pybind11 types (this covers the /*--ignore*/ part)
#include <pybind11/stl.h>
#include <utility>
#include <vector>
std::vector<int> config = {784, 2, 2, 2, 10};
std::vector<int> d_conf = {784, 10};
int batch = 400;
double compress_cost(Mat x) {
  assert(x.rows == 1);
  double s = 0;
  for (int j = 0; j < x.cols; j++) {
    s += x.mat[0][j];
  }
  return s;
}
namespace py = pybind11;
double sigmoidT(double x) { return 1 / (1 + exp(-x)); }

int main() {
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

  auto l1 = std::make_unique<KatyushaLayer>(784, 360, batch, &nn::sigmoid,
                                            "FeedForward");
  auto l2 = std::make_unique<KatyushaLayer>(360, 160, batch, &nn::sigmoid,
                                            "FeedForward");
  auto l3 = std::make_unique<KatyushaLayer>(160, 80, batch, &nn::sigmoid,
                                            "FeedForward");
  auto l4 = std::make_unique<KatyushaLayer>(80, 40, batch, &nn::sigmoid,
                                            "FeedForward");
  auto l5 = std::make_unique<KatyushaLayer>(40, 10, batch, &nn::sigmoid,
                                            "FeedForward");
  NeuralNet FFN(mnist.csv_data, d_conf, 1e-3, batch);
  FFN.add(std::move(l1));
  FFN.add(std::move(l2));
  FFN.add(std::move(l3));
  FFN.add(std::move(l4));
  FFN.add(std::move(l5));
  std::cout << "Calculating L for dataset" << "\n";
  FFN.find_L();
  FFN.print();
  std::vector<std::pair<double, double>> epoch_loss;
  std::cout << "Starting Training:\n";
  // for (int i = 0; i < 500; i++) {
  //   // FFN.set_params(i);
  //   FFN.reset();
  //   FFN.forward();
  //   FFN.cost_fn();
  //   auto temp = compress_cost(FFN.Cost);
  //   std::cout << temp << "\n";
  //   epoch_loss.push_back({i, temp});
  //   FFN.backward();
  //   FFN.update();
  //   gp << "plot '-' with lines title 'Live Data'\n";
  //   gp.send1d(epoch_loss);
  //   if (epoch_loss.size() > 1) {
  //     gp << "set xrange [" << epoch_loss.front().first << ":"
  //        << epoch_loss.back().first << "]\n";
  //   }
  //   gp << std::flush;
  // }
  // FFN.print();
  // return 0;
  FFN.flag = 0.0;
  FFN.X.mul(1.0 / 255);
  for (int i = 0; i < 500; i++) {
    FFN.set_params(0);
    FFN.reset();
    FFN.forward();
    FFN.cost_fn();
    auto temp = compress_cost(FFN.Cost);
    std::cout << temp << "\n";
    epoch_loss.push_back({i, temp});
    gp << "plot '-' with lines title 'Live Data'\n";
    gp.send1d(epoch_loss);
    if (epoch_loss.size() > 1) {
      gp << "set xrange [" << epoch_loss.front().first << ":"
         << epoch_loss.back().first << "]\n";
    }
    gp << std::flush;
    std::cout << "BACK\n";
    FFN.backward();
    std::cout << "BACK\n";

    FFN.update();
    for (int j = 0; j < 4; j++) {
      FFN.flag = 1.0;
      FFN.set_params(i);
      FFN.update();
      FFN.reset();
      FFN.forward();
      FFN.cost_fn();
      FFN.backward();
      FFN.flag = 2.0;
      FFN.set_params(i);
      FFN.update();
    }
    FFN.flag = 3.0;
    FFN.set_params(i);
    FFN.update();
  }
  FFN.print();
  return 0;
}
