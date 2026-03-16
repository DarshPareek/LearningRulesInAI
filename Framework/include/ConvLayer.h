#include "Layer.h"
#ifndef SEQUENTIAL_LAYER_H
class Conv2d : public Layer {
public:
  Mat kernel;
  double (*activation_funciton)(double);
  Conv2d(int input, int output, int kernel, int samples,
         double (*activation)(double), std::string type = "Convolution");
  void forward(Mat X);
  void backward(Mat X);
  void update(std::vector<double> params = {});
};
#endif // !SEQUENTIAL_LAYER_H
