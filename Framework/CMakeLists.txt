cmake_minimum_required(VERSION 3.10)
project(framework CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Use modern C++ practices

# ----------------------------------------------------------------------
# 1. Find all required external dependencies
# ----------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python includes: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Pybind11 version: ${pybind11_VERSION}")

# Using modern Boost FindPackage. The warning will persist if CMake is older than 3.24
# and you haven't explicitly set policy CMP0167 to NEW.
# If you want to suppress the warning, add `cmake_policy(SET CMP0167 NEW)` near the top.
# -------------------------------------------------------
set(Boost_NO_BOOST_CMAKE ON)
set(Boost_USE_STATIC_LIBS ON) 

# 2. Tell the legacy module *exactly* where the headers are.
#    (This is the root directory that contains the 'boost/' folder)
set(BOOST_INCLUDEDIR "/home/darsh/Downloads/boost_1_70_0")

# 3. Tell the legacy module *exactly* where the compiled libs are.
set(BOOST_LIBRARYDIR "/home/darsh/Downloads/boost_1_70_0/stage/lib")
#
# -------------------------------------------------------

# This command is now *forced* to use the legacy module and
# has no choice but to use the exact paths we gave it.
find_package(Boost 1.70.0 COMPONENTS iostreams filesystem system REQUIRED)

message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
# Optional: Common compiler flags for better code quality
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Werror> # Treat warnings as errors for GCC/Clang
)

# ----------------------------------------------------------------------
# 2. Define the source files for your main C++ application
# Explicitly listing is safer than GLOB, especially for the main executable.
# Add all your .cpp files here that make up the executable.
# ----------------------------------------------------------------------
add_executable(framework
    src/main.cpp
    src/FeedForward.cpp        # Assuming your NeuralNet class is implemented here
    src/Layer.cpp            # Assuming your Layer base class is implemented here
    src/KatyushaLayer.cpp    # Assuming KatyushaLayer implementation
    src/Functions.cpp        # Where sigmoid and other nn functions are
    src/DataLoader.cpp       # Assuming MnistData is implemented here
    src/Matrix.cpp           # Assuming your Mat class implementation
    # Add any other .cpp files your C++ application uses
)

# ----------------------------------------------------------------------
# 3. Set properties for the 'framework' executable
# These commands must come AFTER add_executable()
# ----------------------------------------------------------------------

# Define where CMake should look for header files for the 'framework' target
target_include_directories(framework PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include # Your project's custom headers
    ${Boost_INCLUDE_DIRS}               # Boost headers
    ${Python_INCLUDE_DIRS}              # Python headers
    ${pybind11_INCLUDE_DIRS}            # pybind11 headers (often implicitly handled, but explicit is fine)
)

# Link necessary libraries to the 'framework' executable
target_link_libraries(framework PRIVATE
    Boost::iostreams
    Boost::filesystem
    Boost::system
    ${Python_LIBRARIES} # For Python embedding
    pybind11::embed    # For embedding Python interpreter
)

# ----------------------------------------------------------------------
# 4. Define your pybind11 Python module (if you have one separate from main app)
# If read_data.py is a Python script you import, you don't build it with CMake.
# If src/read_csv_bind.cpp is a C++ file that creates a Python module, then define it.
# ----------------------------------------------------------------------
# Example if you have a separate C++ file that creates a Python module:
# pybind11_add_module(read_data_module # This would be the name of your Python module (e.g., import read_data_module)
#     src/read_csv_bind.cpp
# )
#
# target_link_libraries(read_data_module PRIVATE
#     # Link to any of your core C++ libraries if this module uses them
#     # E.g., if read_csv_bind.cpp uses your Mat class, you'd define a Mat library:
#     # MatLib # Assuming you created 'add_library(MatLib ...)' earlier
# )
#
# # If the Python module needs to include your custom headers:
# target_include_directories(read_data_module PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
