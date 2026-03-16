#include "../include/DataLoader.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

MnistData::MnistData(py::module_ read_data_module, std::string data_filepath) {
  py::tuple result = read_data_module.attr("genData")(data_filepath);
  py::tuple size = result[0];
  py::list names = result[1];
  py::list data = result[2];
  int num_rows = size[0].cast<int>();
  int num_columns = size[1].cast<int>();
  rows = num_rows;
  cols = num_columns;
  std::vector<std::string> column_names = get_columns(names, num_columns);
  Mat temp = get_data(data, num_rows, num_columns);
  csv_data.rows = temp.rows;
  csv_data.cols = temp.cols;
  csv_data.mat = temp.mat;
}
std::vector<std::string> MnistData::get_columns(py::list names, int len) {
  std::vector<std::string> result;
  for (int i = 0; i < len; i++) {
    result.push_back(names[i].cast<std::string>());
  }
  return result;
}
Mat MnistData::get_data(py::list data, int rows, int columns) {
  Mat result(rows, columns);
  for (int i = 0; i < rows; i++) {
    py::list _temp = data[i];
    for (int j = 0; j < columns; j++) {
      result.set(i, j, _temp[j].cast<double>());
    }
  }
  return result;
}
