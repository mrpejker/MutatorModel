cmake_minimum_required(VERSION 3.5)
project(mutator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("${PROJECT_SOURCE_DIR}/include/cfd")
include_directories("${PROJECT_SOURCE_DIR}/include/eigen3")

set(SOURCE_FILES src/main.cpp)
add_executable(mutator ${SOURCE_FILES})
