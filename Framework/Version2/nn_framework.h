#include <assert.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <regex>
#include <typeinfo>
#include <vector>
inline double rand_val(int low, int high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distr(low, high);
  return (double)distr(gen);
}
inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
inline double grad_sigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
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

class NeuralNet {
public:
  std::vector<Mat> Layers;
  std::vector<Mat> Biases;
  std::vector<Mat> Outputs;
  std::vector<int> config;
  int batch;
  Mat dataset;
  Mat X;
  Mat Y;
  Mat cost;
  double lr;
  NeuralNet() {}
  NeuralNet(std::vector<int> config, Mat data, std::vector<int> d_conf,
            double rate) {
    config = config;
    // Allocating Layers and Biases
    for (int i = 0; i < config.size() - 1; i++) {
      auto matW = Mat(config[i], config[i + 1]);
      matW.fill_rand();
      auto matB = Mat(data.rows, config[i + 1]);
      matB.fill(rand_val(0, 1));
      auto matO = Mat(data.rows, config[i + 1]);
      matO.fill(0);
      Layers.push_back(matW);
      Biases.push_back(matB);
      Outputs.push_back(matO);
    }
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
  //   void forward() {
  //     for (int i = 0; i < Outputs.size(); i++) {
  //       Outputs[i].mul(X, Layers[i]);
  //       Outputs[i].add(Biases[i]);
  //       Outputs[i].apply_activation(&sigmoid);
  //     }
  //   }
  //   void cost_fn() {
  //     cost.rows = dataset.rows;
  //     cost.cols = Y.cols;
  //     cost.fill(0);
  //     cost.add(Y); /* HACK Using Add in a non-intended way */
  //     Outputs[0].dot_mul(-1);
  //     cost.add(Outputs[0]);
  //     Outputs[0].dot_mul(-1);
  //   }
  //   void backward() {
  //     Outputs[0].mat = cost.mat;
  //     cost.dot_mul(lr);
  //     Outputs[0].apply_activation(&grad_sigmoid);
  //     Outputs[0].dot(cost);
  //     Biases[0].add(Outputs[0]);
  //     Outputs[0].transpose();
  //     Mat temp(Outputs[0].rows, X.cols);
  //     temp.fill(0);
  //     temp.mul(Outputs[0], X);
  //     temp.transpose();
  //     Layers[0].add(temp);
  //     Outputs[0].transpose();
  //     Outputs[0].mat.clear();
  //     Outputs[0].fill(0);
  //     cost.mat.clear();
  //     cost.fill(0);
  //   }
};
