cmake_minimum_required(VERSION 3.10)

# set the project name
project(test)

# set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-DUSE_KISS_FFT")

# add_subdirectory(compute-mfcc)
add_subdirectory(Gist)
include_directories (${PROJECT_SOURCE_DIR}/compute-mfcc)
include_directories (${PROJECT_SOURCE_DIR}/Gist/src)
include_directories (${PROJECT_SOURCE_DIR}/Gist/libs/kiss_fft130)


add_executable (test 
    test.cpp
    inference.cpp
    # ${PROJECT_SOURCE_DIR}/compute-mfcc/mfcc.cc
    ${PROJECT_SOURCE_DIR}/Gist/libs/kiss_fft130/kiss_fft.c
)
target_link_libraries(test gslcblas)
target_link_libraries (test Gist)
target_link_libraries(test sndfile)
# target_compile_features (test PRIVATE cxx_std_17)
target_link_libraries(test gsl)
target_link_libraries(test m)
