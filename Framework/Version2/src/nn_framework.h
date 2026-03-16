#include <assert.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <regex>
#include <string>
#include <typeinfo>
#include <vector>
#pragma once
namespace pybind11 {
class module_;
class tuple;
class list;
} // namespace pybind11
inline double rand_val(int low, int high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distr(low, high);
  return (double)distr(gen);
}
inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
inline double grad_sigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline double square(double x) { return x * x; }
class Mat {

public:
  Mat() {}
  std::vector<std::vector<double>> mat;
  int rows;
  int cols;
  Mat(int mrows, int mcols) {
    rows = mrows;
    cols = mcols;
    std::vector<double> temp;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        temp.push_back(0);
      }
      this->mat.push_back(temp);
      temp.clear();
    }
  }
  void allocate_mat() {
    assert(rows > 0);
    assert(cols > 0);
    mat.clear();
    std::vector<double> temp;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        temp.push_back(0);
      }
      mat.push_back(temp);
      temp.clear();
    }
  }
  void fill(double val) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat[i][j] = val;
      }
    }
  }
  void fill_rand(int low = 0, int high = 1) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat[i][j] = rand_val(low, high);
      }
    }
  }
  void print(int w = 0) {
    std::cout << std::setw(w / 2) << "[" << std::endl;
    for (int i = 0; i < rows; i++) {
      std::cout << std::setw(w);
      for (int j = 0; j < cols; j++) {
        std::cout << this->mat[i][j] << "    ";
      }
      std::cout << std::endl;
    }
    std::cout << std::setw(w / 2) << "]\n" << std::endl;
  }
  void apply_activation(double (*f)(double)) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->mat[i][j] = f(this->mat[i][j]);
      }
    }
  }
  void add(Mat x) {
    assert(this->rows == x.rows);
    assert(this->cols == x.cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->mat[i][j] += x.mat[i][j];
      }
    }
  }
  void add_column_wise(Mat x) {
    assert(x.rows == 1);
    assert(this->cols == x.cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->mat[i][j] += x.mat[0][j];
      }
    }
  }
  void add(std::vector<double> x) {
    // assert(this->rows == x.rows);
    // assert(this->cols == x.cols);
    // No assert because I trust myself to use this without errors
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->mat[i][j] += x[j];
      }
    }
  }

  void add(double x) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->mat[i][j] += x;
      }
    }
  }
  void mul(Mat x, Mat y) {
    assert(x.cols == y.rows);
    assert(this->rows == x.rows && this->cols == y.cols);
    this->fill(0);
    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->cols; j++) {
        for (int k = 0; k < x.cols; k++) {
          this->mat[i][j] += x.mat[i][k] * y.mat[k][j];
        }
      }
    }
  }
  void set(int i, int j, double val) {
    assert(i < rows && j < cols);
    this->mat[i][j] = val;
  }
  double get(int i, int j) {
    assert(i < rows && j < cols);
    return this->mat[i][j];
  }
  void mul(double val) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->mat[i][j] *= val;
      }
    }
  }
  Mat transpose() {
    Mat temp(cols, rows);
    for (int i = 0; i < temp.rows; i++) {
      for (int j = 0; j < temp.cols; j++) {
        temp.set(i, j, mat[j][i]);
      }
    }
    return temp;
  }
  void dot(Mat x) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat[i][j] *= x.mat[i][j];
      }
    }
  }
  void squish_rows(Mat x) {
    assert(rows == 1);
    assert(cols = x.cols);
    this->fill(0);
    for (int i = 0; i < x.rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat[0][j] += x.mat[i][j];
      }
    }
  }
  void squish_columns() {
    for (int i = 0; i < rows; i++) {
      for (int j = 1; j < cols; j++) {
        mat[i][0] += mat[i][j];
      }
    }
    cols = 1;
  }
};

class Layer {
public:
  Mat weights;
  Mat grad_weights;
  Mat biases;
  Mat grad_biases;
  Mat hidden;
  Mat outputs;
  Mat grad_outputs;
  int batches;
  std::string type;
  // Layer(int input_neurons, int output_neurons, int samples, std::string
  // type);
  void print(int padding);
  void reset();
};
class SequentialLayer : public Layer {
public:
  double (*activation_funciton)(double);
  Mat Cost;
  SequentialLayer(int input, int output, int samples,
                  double (*activation)(double), std::string type);
  void forward(Mat X);
};

class NeuralNet {
public:
  std::vector<Mat> Layers;
  std::vector<Mat> Biases;
  std::vector<Mat> Hidden;
  std::vector<Mat> Outputs;
  std::vector<Mat> gOutputs;
  std::vector<Mat> gBiases;
  std::vector<Mat> gLayers;
  Mat Cost;
  Mat dataset;
  Mat X;
  Mat Y;
  std::vector<int> config;
  int batch;
  double lr;
  NeuralNet() {}
  NeuralNet(std::vector<int> config, Mat data, std::vector<int> d_conf,
            double rate, int batch) {
    config = config;
    batch = batch;
    //     // Managing The dataset
    lr = rate;
    dataset.rows = data.rows;
    dataset.cols = data.cols;
    dataset.mat = data.mat;
    //     // Loading Samples
    int inputs = d_conf[0];
    int outputs = d_conf[1];
    X.cols = inputs;
    Y.cols = outputs;
    X.rows = dataset.rows;
    Y.rows = dataset.rows;
    for (int i = 0; i < dataset.rows; i++) {
      std::vector<double> x;
      std::vector<double> y;
      for (int j = 0; j < dataset.cols; j++) {
        if (j < inputs)
          x.push_back(dataset.get(i, j));
        else
          y.push_back(dataset.get(i, j));
      }
      X.mat.push_back(x);
      Y.mat.push_back(y);
      x.clear();
      y.clear();
    }
    Outputs.push_back(X);
    for (int b = 0; b < config.size() - 1; b++) {
      Mat tempL(config[b], config[b + 1]);
      tempL.fill_rand();
      Layers.push_back(tempL);
      gLayers.push_back(tempL);
      Mat tempB(1, config[b + 1]);
      tempB.fill(0);
      Biases.push_back(tempB);
      gBiases.push_back(tempB);
      Mat tempH(batch, config[b + 1]);
      Hidden.push_back(tempH);
      Outputs.push_back(tempH);
      gOutputs.push_back(tempH);
    }
    Cost.rows = 1;
    Cost.cols = config[config.size() - 1];
    Cost.allocate_mat();
  }
  void print(int padding = 0) {
    int init_padding = padding;
    std::cout << std::setw(padding) << "Weights: " << std::endl;
    for (auto i : Layers) {
      padding += 10;
      i.print(padding);
    }
    padding = init_padding;
    for (auto i : Biases) {
      padding += 10;
      i.print(padding);
    }
    padding = init_padding;
    for (auto i : Outputs) {
      padding += 10;
      i.print(padding);
    }
  }
  void reset() {
    for (int i = 0; i < Outputs.size() - 1; i++) {
      Hidden[i].fill(0);
      Outputs[i + 1].fill(0);
      gLayers[i].fill(0);
      gBiases[i].fill(0);
      gOutputs[i].fill(0);
    }
  }
  void forward() {
    for (int j = 0; j < Outputs.size() - 1; j++) {
      Hidden[j].mul(Outputs[j], Layers[j]);
      Hidden[j].add_column_wise(Biases[j]);
      Outputs[j + 1].add(Hidden[j]);
      Outputs[j + 1].apply_activation(&sigmoid);
    }
  }
  void cost_fn() {
    Outputs[Outputs.size() - 1].mul(-1);
    Outputs[Outputs.size() - 1].add(Y);
    gOutputs[gOutputs.size() - 1].add(Outputs[Outputs.size() - 1]);
    Outputs[Outputs.size() - 1].apply_activation(&square);
    // std::cout << Cost.rows << " " << Outputs[Outputs.size() - 1].rows
    //<< " Hello\n";
    Cost.squish_rows(Outputs[Outputs.size() - 1]);
    // std::cout << "Hello\n";
    Cost.mul((double)(1.00 / X.rows));
  }
  void backward() {
    for (int k = gOutputs.size() - 1; k > -1; k--) {
      Hidden[k].apply_activation(&grad_sigmoid);
      Hidden[k].dot(gOutputs[k]);
      gBiases[k].squish_rows(Hidden[k]);
      gBiases[k].mul((1.00 / X.rows) * lr);
      Biases[k].add(gBiases[k]);
      auto t = Outputs[k].transpose();
      gLayers[k].mul(t, Hidden[k]);
      gLayers[k].mul((1.00 / X.rows) * lr);
      Layers[k].add(gLayers[k]);
      if (k != 0)
        gOutputs[k - 1].mul(Hidden[k], Layers[k].transpose());
    }
  }
};
class MnistData {
public:
  Mat csv_data;
  int rows;
  int cols;
  std::string data_path;
  MnistData(pybind11::module_ read_data_module, std::string data_filepath);
  std::vector<std::string> get_columns(pybind11::list names, int len);
  Mat get_data(pybind11::list data, int rows, int columns);
};
