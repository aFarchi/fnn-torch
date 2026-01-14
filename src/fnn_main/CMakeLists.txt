cmake_minimum_required(VERSION 3.16)
project(fnn-torch LANGUAGES Fortran)

add_library(fnn fnn.f90)

add_executable(run-fnn.x main.f90)
target_link_libraries(run-fnn.x PRIVATE fnn)
