cmake_minimum_required(VERSION 3.19)
project(Ex_2 C)

set(CMAKE_C_STANDARD 90)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(Ex_2 kmeans.c)
target_link_libraries(Ex_2 ${PYTHON_LIBRARIES})