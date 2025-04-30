#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// (Layer and NeuralNetwork structures remain the same as in the previous
// response)
// Basic Neural Network Layer structure
struct Layer {
  int numInputs;
  int numOutputs;
  std::vector<std::vector<double>> weights;
  std::vector<double> biases;

  Layer(int inputs, int outputs)
      : numInputs(inputs), numOutputs(outputs),
        weights(inputs, std::vector<double>(outputs)), biases(outputs) {}

  // Initialize weights and biases with random values
  void initialize(std::mt19937 &generator,
                  std::uniform_real_distribution<> &distribution) {
    for (int i = 0; i < numInputs; ++i) {
      for (int j = 0; j < numOutputs; ++j) {
        weights[i][j] = distribution(generator);
      }
    }
    for (int i = 0; i < numOutputs; ++i) {
      biases[i] = distribution(generator);
    }
  }

  // Forward pass
  std::vector<double> forward(const std::vector<double> &inputs) const {
    std::vector<double> outputs(numOutputs, 0.0);
    for (int i = 0; i < numOutputs; ++i) {
      for (int j = 0; j < numInputs; ++j) {
        outputs[i] += inputs[j] * weights[j][i];
      }
      outputs[i] += biases[i];
      outputs[i] = sigmoid(outputs[i]); // Using sigmoid activation
    }
    return outputs;
  }

private:
  // Sigmoid activation function
  double sigmoid(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
};

// Neural Network structure
struct NeuralNetwork {
  std::vector<Layer> layers;

  NeuralNetwork() {}

  void addLayer(Layer layer) { layers.push_back(layer); }

  void initialize(std::mt19937 &generator,
                  std::uniform_real_distribution<> &distribution) {
    for (auto &layer : layers) {
      layer.initialize(generator, distribution);
    }
  }

  std::vector<double> predict(const std::vector<double> &inputs) const {
    std::vector<double> currentOutput = inputs;
    for (const auto &layer : layers) {
      currentOutput = layer.forward(currentOutput);
    }
    return currentOutput;
  }

  // Get all weights and biases as a single vector for genetic algorithm
  std::vector<double> getParameters() const {
    std::vector<double> params;
    for (const auto &layer : layers) {
      for (const auto &weightRow : layer.weights) {
        params.insert(params.end(), weightRow.begin(), weightRow.end());
      }
      params.insert(params.end(), layer.biases.begin(), layer.biases.end());
    }
    return params;
  }

  // Set weights and biases from a single parameter vector
  void setParameters(const std::vector<double> &params) {
    int paramIndex = 0;
    for (auto &layer : layers) {
      for (int i = 0; i < layer.numInputs; ++i) {
        for (int j = 0; j < layer.numOutputs; ++j) {
          layer.weights[i][j] = params[paramIndex++];
        }
      }
      for (int i = 0; i < layer.numOutputs; ++i) {
        layer.biases[i] = params[paramIndex++];
      }
    }
  }

  int getNumParameters() const {
    int count = 0;
    for (const auto &layer : layers) {
      count += layer.numInputs * layer.numOutputs; // Weights
      count += layer.numOutputs;                   // Biases
    }
    return count;
  }
};

// Function to evaluate the fitness of a neural network (policy)
double evaluateNetwork(const NeuralNetwork &network) {
  double fitness = 0.0;
  // Truth table for AND gate: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1
  if (std::round(network.predict({0.0, 0.0})[0]) == 0)
    fitness += 1.0;
  if (std::round(network.predict({0.0, 1.0})[0]) == 0)
    fitness += 1.0;
  if (std::round(network.predict({1.0, 0.0})[0]) == 0)
    fitness += 1.0;
  if (std::round(network.predict({1.0, 1.0})[0]) == 1)
    fitness += 1.0;
  return fitness;
}

// Function to create a random neural network (policy)
NeuralNetwork createRandomNetwork(int inputSize, int hiddenSize,
                                  int outputSize) {
  NeuralNetwork network;
  network.addLayer(Layer(inputSize, hiddenSize));
  network.addLayer(Layer(hiddenSize, outputSize));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(-1.0, 1.0);
  network.initialize(gen, distrib);
  return network;
}

// Function to mutate the parameters of a neural network (policy)
NeuralNetwork mutateNetwork(const NeuralNetwork &network, double mutationRate,
                            double mutationStep) {
  NeuralNetwork mutatedNetwork;
  for (const auto &layer : network.layers) {
    mutatedNetwork.addLayer(Layer(layer.numInputs, layer.numOutputs));
  }
  mutatedNetwork.setParameters(
      network.getParameters()); // Copy existing parameters

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(-mutationStep, mutationStep);
  std::uniform_real_distribution<> probDistrib(0.0, 1.0);

  std::vector<double> params = mutatedNetwork.getParameters();
  for (double &param : params) {
    if (probDistrib(gen) < mutationRate) {
      param += distrib(gen);
    }
  }
  mutatedNetwork.setParameters(params);
  return mutatedNetwork;
}

int main() {
  int populationSize = 50;
  int numGenerations = 20000; // May need more generations for convergence
  double mutationRate = 0.05;
  double mutationStep = 0.1;
  int inputSize = 2;  // For the two inputs of the AND gate
  int hiddenSize = 2; // Number of neurons in the hidden layer
  int outputSize = 1; // Single output for the AND result

  std::vector<NeuralNetwork> population(populationSize);
  for (int i = 0; i < populationSize; ++i) {
    population[i] = createRandomNetwork(inputSize, hiddenSize, outputSize);
  }

  NeuralNetwork bestNetworkSoFar = population[0];
  double bestFitnessSoFar = evaluateNetwork(bestNetworkSoFar);

  for (int generation = 0; generation < numGenerations; ++generation) {
    std::vector<std::pair<double, int>> fitnesses;
    for (int i = 0; i < populationSize; ++i) {
      double fitness = evaluateNetwork(population[i]);
      fitnesses.push_back({fitness, i});
      if (fitness > bestFitnessSoFar) {
        bestFitnessSoFar = fitness;
        bestNetworkSoFar = population[i];
      }
    }

    std::sort(fitnesses.rbegin(), fitnesses.rend());

    std::cout << "Generation " << generation + 1
              << ", Best Fitness: " << fitnesses[0].first << std::endl;

    std::vector<NeuralNetwork> newPopulation;
    int eliteSize = populationSize / 10;
    for (int i = 0; i < eliteSize; ++i) {
      newPopulation.push_back(population[fitnesses[i].second]);
    }

    while (newPopulation.size() < populationSize) {
      int parentIndex = std::rand() % eliteSize;
      newPopulation.push_back(
          mutateNetwork(population[fitnesses[parentIndex].second], mutationRate,
                        mutationStep));
    }

    population = newPopulation;

    if (bestFitnessSoFar == 4.0) {
      std::cout << "Perfect AND gate network found in generation "
                << generation + 1 << "!" << std::endl;
      break;
    }
  }

  // Output the best network found
  std::cout << "\nBest network performance:" << std::endl;
  std::cout << "Input (0,0): "
            << std::round(bestNetworkSoFar.predict({0.0, 0.0})[0])
            << " (Target: 0)" << std::endl;
  std::cout << "Input (0,1): "
            << std::round(bestNetworkSoFar.predict({0.0, 1.0})[0])
            << " (Target: 0)" << std::endl;
  std::cout << "Input (1,0): "
            << std::round(bestNetworkSoFar.predict({1.0, 0.0})[0])
            << " (Target: 0)" << std::endl;
  std::cout << "Input (1,1): "
            << std::round(bestNetworkSoFar.predict({1.0, 1.0})[0])
            << " (Target: 1)" << std::endl;

  std::cout << "\nParameters of the best network:" << std::endl;
  std::vector<double> bestParams = bestNetworkSoFar.getParameters();
  for (size_t i = 0; i < bestParams.size(); ++i) {
    std::cout << bestParams[i] << " ";
    if ((i + 1) % (hiddenSize * inputSize + hiddenSize) == 0) {
      std::cout << "| ";
    }
    if ((i + 1) % (outputSize * hiddenSize + outputSize +
                   hiddenSize * inputSize + hiddenSize) ==
        0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  return 0;
}
