cmake_minimum_required(VERSION 3.1)

project(barrel_distort_cpp)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_BUILD_TYPE Debug)

# force C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV stuff
find_package(OpenCV REQUIRED)

add_executable(barrel_distort_cpp main.cpp barrel_distortion.hpp barrel_distortion.cpp)

target_link_libraries(barrel_distort_cpp ${OpenCV_LIBS})
