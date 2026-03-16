#ifndef LAYER_H // Unique guard for Layer.h
#define LAYER_H
#include "Matrix.h"
#include <string>
class Layer {
protected:
  Layer()
      : weights(), grad_weights(), biases(), grad_biases(), hidden(), outputs(),
        grad_outputs(), batches(0), type("") {}

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
  void print(int padding = 0);
  void reset();
  virtual void forward(Mat x) = 0;
  virtual void backward(Mat x) = 0;
  virtual void update(std::vector<double> params = {}) = 0;
};

#endif // !LAYER_H
