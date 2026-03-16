// class NN_Config {
// public:
//   std::vector<int> neurons_in_layers;
//   void parse_config(std::string config, int size) {
//     std::stringstream ss;
//     ss << config.substr(1, size);
//     std::string temp;
//     int found;
//     while (!ss.eof()) {
//       ss >> temp;
//       if (std::stringstream(temp) >> found) {
//         neurons_in_layers.push_back(found);
//       }
//       temp = "";
//     }
//   }
//   void print_nn_info() {
//     std::cout << "Number of Hidden Layers " << neurons_in_layers.size()
//               << std::endl;
//     for (const int &i : neurons_in_layers)
//       std::cout << "i = " << i << std::endl;
//   }
// };

// std::string config = argv[1];
// std::cout << "Hello World" << std::endl;
// std::cout << "We will be having " << argc - 2 << " layers" << std::endl;
// std::cout << "We have " << config << " as our layer config " << std::endl;

// NN_Config nn;
// int conf_size = config.size();
// nn.parse_config(config, conf_size);
// nn.print_nn_info();
//
