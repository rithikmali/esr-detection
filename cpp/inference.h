// #include <gsl/gsl_matrix.h>
// #include <gsl/gsl_blas.h>
// #include <gsl/gsl_math.h>
#include<stdio.h>
#include<stdlib.h>
#include <vector>
#include "AudioFile.h"
#include<iostream>
#include <algorithm>
#include <iostream>
#include <fstream>
#inclide<hifi4_library.a>
#include "mfcc.cpp"

char* getCmdOption(char **begin, char **end, const std::string &value);
template <size_t rows, size_t cols>
void my_load_matrix(const char* filename, int size, float (&m)[rows][cols]);

// float sigmoid(float x);
// void predict_gsl(gsl_matrix* input, gsl_matrix* alpha, gsl_matrix* beta, gsl_matrix* output);
void predict(float *input, float *alpha, float *beta, float *output);
void get_zcr(float (*input)[5208], float *a, int n_frames);
void get_stl_mfcc(float (*input)[5208], const char *wavFp, int n_frames);