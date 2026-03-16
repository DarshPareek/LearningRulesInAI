#include <vector>
#ifndef Matrix_H
#define Matrix_H
class Mat {
public:
  std::vector<std::vector<double>> mat;
  int rows;
  int cols;
  Mat();
  Mat(int rows, int cols);
  Mat &operator=(const Mat &other);
  ~Mat() = default;
  Mat(const Mat &other) = default;
  void allocate_mat();
  void fill(double val);
  void fill_rand(int low = 0, int high = 1);
  void print(int w = 0);
  void apply_activation(double (*f)(double));
  void add(Mat x);
  void add(std::vector<double> x);
  void add(double x);
  void add_column_wise(Mat x);
  void mul(Mat x, Mat y);
  void set(int i, int j, double val);
  double get(int i, int j);
  void mul(double val);
  Mat transpose();
  void dot(Mat x);
  void squish_rows(Mat x);
  void squish_columns();
  void norm();
};
#endif
