#include "../include/Layer.h"
#include <iomanip>
#include <iostream>
void Layer::print(int padding) {
  std::cout << std::setw(padding) << "Layer of type " << this->type
            << std::endl;
  std::cout << std::setw(padding) << "Weights: " << std::endl;
  weights.print(padding);

  std::cout << std::setw(padding) << "Biases: " << std::endl;
  biases.print(padding);

  std::cout << std::setw(padding) << "Outputs: " << std::endl;
  outputs.print(padding);
}
void Layer::reset() {
  grad_outputs.fill(0);
  hidden.fill(0);
  outputs.fill(0);
  grad_biases.fill(0);
  grad_weights.fill(0);
}
