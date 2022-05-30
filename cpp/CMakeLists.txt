cmake_minimum_required(VERSION 3.10)

# set the project name
project(test)

set (CMAKE_CXX_STANDARD 11)


add_executable (test 
    test.cpp
    inference.cpp
)

target_link_libraries(test m)
