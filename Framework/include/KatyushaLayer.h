#ifndef KATYUSHA_LAYER_H
#define KATYUSHA_LAYER_H
#include "Layer.h"
class KatyushaLayer : public Layer {
public:
  Mat temp_weights, y_weights, z_weights, x_snapshot_weights,
      mu_snapshot_weights, y_avg_weights;
  Mat temp_biases, y_biases, z_biases, x_snapshot_biases, mu_snapshot_biases,
      y_avg_biases;
  double (*activation_funciton)(double);
  KatyushaLayer(int input, int output, int samples,
                double (*activation)(double), std::string type = "Katyusha");
  void forward(Mat X) override;
  void backward(Mat X) override;
  void set_to_temp();
  void set_to_y();
  void set_to_x();
  void set_to_z();
  void set_mu_grad();
  void set_normal();
  void set_params(int num_epoch);
  void update(std::vector<double> params = {}) override;
};
#endif // !KATYUSHA_LAYER_H
