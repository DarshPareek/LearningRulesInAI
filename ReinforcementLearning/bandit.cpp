#include <cmath> // For std::log (needed for UCB later)
#include <iostream>
#include <limits>  // For std::numeric_limits
#include <numeric> // For std::iota
#include <random>
#include <stdexcept> // For std::runtime_error
#include <vector>

// --- Random Number Generation Setup ---
// Use a high-quality random number generator
std::random_device rd;        // Seed
std::mt19937 generator(rd()); // Mersenne Twister engine

// --- BanditArm Class ---
class BanditArm {
private:
  double
      true_probability; // The actual (unknown to the algo) success probability
  int pulls;
  double total_reward;

public:
  // Constructor: Initializes the arm with its true probability
  BanditArm(double prob) : true_probability(prob), pulls(0), total_reward(0.0) {
    if (prob < 0.0 || prob > 1.0) {
      throw std::runtime_error("Probability must be between 0.0 and 1.0");
    }
  }

  // Simulate pulling the arm (returns 1 for reward, 0 for no reward)
  // This uses the *true* probability for simulation purposes.
  // In a real scenario, this reward would come from an external system.
  double pull() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double reward = (distribution(generator) < true_probability) ? 1.0 : 0.0;

    // Update internal stats AFTER determining reward for this pull
    pulls++;
    total_reward += reward;
    return reward;
  }

  // Get the estimated average reward based on past pulls
  double get_estimated_reward() const {
    if (pulls == 0) {
      // If never pulled, return a default value.
      // Returning 0 is common, or sometimes positive infinity initially
      // to encourage pulling each arm at least once. Let's use 0.
      return 0.0;
    }
    return total_reward / pulls;
  }

  // Getters for analysis (optional)
  int get_pull_count() const { return pulls; }
  double get_total_reward() const { return total_reward; }
  double get_true_probability() const {
    return true_probability;
  } // For comparing results
};

// --- EpsilonGreedyBandit Class ---
class EpsilonGreedyBandit {
private:
  std::vector<BanditArm> arms;
  double epsilon;   // Exploration rate (0.0 to 1.0)
  int total_trials; // Track total pulls across all arms

public:
  // Constructor
  EpsilonGreedyBandit(const std::vector<double> &probabilities, double eps)
      : epsilon(eps), total_trials(0) {
    if (eps < 0.0 || eps > 1.0) {
      throw std::runtime_error("Epsilon must be between 0.0 and 1.0");
    }
    arms.reserve(probabilities.size()); // Pre-allocate memory
    for (double prob : probabilities) {
      arms.emplace_back(prob); // Create BanditArm objects
    }
    if (arms.empty()) {
      throw std::runtime_error("Must have at least one arm.");
    }
  }

  // Select an arm based on the Epsilon-Greedy strategy
  int select_arm() {
    total_trials++; // Increment total trial count

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_value = distribution(generator);

    if (random_value < epsilon) {
      // --- Explore ---
      // Choose a random arm uniformly
      std::uniform_int_distribution<int> arm_distribution(0, arms.size() - 1);
      return arm_distribution(generator);
    } else {
      // --- Exploit ---
      // Find the arm with the highest estimated reward
      int best_arm_index = 0;
      double max_estimated_reward = -1.0; // Start below possible reward range

      for (int i = 0; i < (int)arms.size(); ++i) {
        double current_estimate = arms[i].get_estimated_reward();
        // If an arm hasn't been pulled, its estimate is 0 (based on
        // get_estimated_reward). To ensure each arm is tried initially, some
        // implementations force pulling each arm once before starting
        // epsilon-greedy, or assign an infinite initial estimate. Our current
        // simple version might take a while to try an arm if others quickly get
        // rewards.

        if (current_estimate > max_estimated_reward) {
          max_estimated_reward = current_estimate;
          best_arm_index = i;
        }
        // Optional: Handle ties randomly (e.g., collect all tied indices and
        // pick one randomly) For simplicity here, we just take the first one
        // found.
      }
      return best_arm_index;
    }
  }

  // Run the bandit simulation for a number of trials
  void run(int num_trials) {
    double cumulative_reward = 0.0;

    std::cout << "Running Epsilon-Greedy with epsilon = " << epsilon << " for "
              << num_trials << " trials." << std::endl;

    for (int i = 0; i < num_trials; ++i) {
      int chosen_arm_index = select_arm(); // Decide which arm to pull
      double reward =
          arms[chosen_arm_index].pull(); // Pull the arm and get reward (updates
                                         // arm's internal stats)
      cumulative_reward += reward;

      // Optional: Print progress periodically
      // if ((i + 1) % (num_trials / 10) == 0) {
      //     std::cout << "Trial " << (i + 1) << ": Chose arm " <<
      //     chosen_arm_index
      //               << ", Reward: " << reward << ", Cumulative Reward: " <<
      //               cumulative_reward << std::endl;
      // }
    }

    std::cout << "\nSimulation Complete." << std::endl;
    std::cout << "Total Reward: " << cumulative_reward << std::endl;
    std::cout << "Overall Average Reward: "
              << (num_trials > 0 ? cumulative_reward / num_trials : 0.0)
              << std::endl;

    // Print final statistics for each arm
    std::cout << "\nArm Statistics:" << std::endl;
    double best_true_prob = 0.0;
    for (int i = 0; i < (int)arms.size(); ++i) {
      best_true_prob = std::max(best_true_prob, arms[i].get_true_probability());
      std::cout << "  Arm " << i
                << ": True Prob = " << arms[i].get_true_probability()
                << ", Estimated Reward = " << arms[i].get_estimated_reward()
                << ", Pulls = " << arms[i].get_pull_count() << std::endl;
    }
    std::cout << "Best possible average reward (optimal arm): "
              << best_true_prob << std::endl;
  }

  // Getters
  size_t get_num_arms() const { return arms.size(); }
  const BanditArm &get_arm(int index) const {
    return arms.at(index);
  } // .at() provides bounds checking
  int get_total_trials() const { return total_trials; }
};

// --- Main Function ---
int main() {
  // Define the true reward probabilities for each arm
  std::vector<double> probabilities = {
      0.1, 0.3, 0.8, 0.5, 0.2}; // Arm 2 (index 2) is the best (0.8)

  // Set algorithm parameters
  double epsilon = 0.9;
  double epsilon1 = 0.1; // Explore 10% of the time
  int num_trials = 1000; // Number of times to pull an arm

  try {
    // Create and run the bandit
    EpsilonGreedyBandit bandit(probabilities, epsilon);
    bandit.run(num_trials);
    EpsilonGreedyBandit bandit1(probabilities, epsilon1);
    bandit1.run(num_trials);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
