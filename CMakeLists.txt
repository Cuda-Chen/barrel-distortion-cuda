cmake_minimum_required(VERSION 3.1)

project(barrel_distort_cuda)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_BUILD_TYPE Debug)

# force C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV stuff
find_package(OpenCV REQUIRED)

# CUDA stuff
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(barrel_distort_cuda main.cpp barrel_distortion.cuh barrel_distortion.cu)

target_link_libraries(barrel_distort_cuda ${OpenCV_LIBS})
