#include "../include/Matrix.h"
#include "../include/Functions.h"
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

Mat::Mat() {}
Mat::Mat(int mrows, int mcols) {
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
Mat &Mat::operator=(const Mat &other) {
  this->rows = other.rows;
  this->cols = other.cols;
  this->mat = other.mat;
  return *this;
}
void Mat::allocate_mat() {
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
void Mat::fill(double val) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat[i][j] = val;
    }
  }
}
void Mat::fill_rand(int low, int high) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat[i][j] = nn::rand_val(low, high);
    }
  }
}
void Mat::print(int w) {
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
void Mat::apply_activation(double (*f)(double)) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->mat[i][j] = f(this->mat[i][j]);
    }
  }
}
// TODO: Implement an add function where it takes two matrices and return a new
// one.
void Mat::add(Mat x) {
  assert(this->rows == x.rows);
  assert(this->cols == x.cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->mat[i][j] += x.mat[i][j];
    }
  }
}
void Mat::add(std::vector<double> x) {
  // assert(this->rows == x.rows);
  // assert(this->cols == x.cols);
  // No assert because I trust myself to use this without errors
  // FIX: Fix this function
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->mat[i][j] += x[j];
    }
  }
}
void Mat::add(double x) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->mat[i][j] += x;
    }
  }
}
void Mat::add_column_wise(Mat x) {
  assert(x.rows == 1);
  assert(this->cols == x.cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->mat[i][j] += x.mat[0][j];
    }
  }
}
void Mat::mul(Mat x, Mat y) {
  // TODO: Make this function accept only one matrix and return the result
  // matrix
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
void Mat::mul(double val) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->mat[i][j] *= val;
    }
  }
}
void Mat::set(int i, int j, double val) {
  assert(i < rows && j < cols);
  this->mat[i][j] = val;
}
double Mat::get(int i, int j) {
  assert(i < rows && j < cols);
  return this->mat[i][j];
}
Mat Mat::transpose() {
  Mat temp(cols, rows);
  for (int i = 0; i < temp.rows; i++) {
    for (int j = 0; j < temp.cols; j++) {
      temp.set(i, j, mat[j][i]);
    }
  }
  return temp;
}
void Mat::dot(Mat x) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat[i][j] *= x.mat[i][j];
    }
  }
}
void Mat::squish_rows(Mat x) {
  assert(rows == 1);
  // assert(cols == x.cols);
  this->fill(0);
  for (int i = 0; i < x.rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat[0][j] += x.mat[i][j];
    }
  }
}
void Mat::squish_columns() {
  for (int i = 0; i < rows; i++) {
    for (int j = 1; j < cols; j++) {
      mat[i][0] += mat[i][j];
    }
  }
  cols = 1;
}
void Mat::norm() {
  assert(cols == 1); // Must be a column vector!! This function will only
                     // compute normalisation for column vectors
  double sum = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sum += mat[i][j] * mat[i][j];
    }
  }
  sum = sqrt(sum);
  mul(1.0 / sum);
}
