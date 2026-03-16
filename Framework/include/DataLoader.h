#include "Matrix.h"
#include <pybind11/pybind11.h>
#include <string>
#pragma once
namespace pybind11 {
class module_;
class tuple;
class list;
} // namespace pybind11
#ifndef DATALOADER_H
class MnistData {
public:
  Mat csv_data;
  int rows;
  int cols;
  std::string data_path;
  MnistData(pybind11::module_ read_data_module, std::string data_filepath);
  std::vector<std::string> get_columns(pybind11::list names, int len);
  Mat get_data(pybind11::list data, int rows, int columns);
};
#endif // !DATALOADER_H
