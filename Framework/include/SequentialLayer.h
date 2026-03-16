#include "Layer.h"
#ifndef SEQUENTIAL_LAYER_H
class SequentialLayer : public Layer {
public:
  double (*activation_funciton)(double);
  SequentialLayer(int input, int output, int samples,
                  double (*activation)(double),
                  std::string type = "Sequential");
  void forward(Mat X) override;
  void backward(Mat X) override;
  void update(std::vector<double> params = {}) override;
};
#endif // !SEQUENTIAL_LAYER_H
