#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include<stdio.h>
#include<stdlib.h>
#include <vector>
#include <sndfile.h>
#include <Gist.h>
#include<iostream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "mfcc.cpp"

char* getCmdOption(char **begin, char **end, const std::string &value);
void my_load_matrix(const char* filename, int size, gsl_matrix * m);
double sigmoid(double x);
void predict(gsl_matrix* input, gsl_matrix* alpha, gsl_matrix* beta, gsl_matrix* output);
void get_zcr(gsl_matrix* input, double* a, int n_frames);
void get_stl_mfcc(gsl_matrix * input, const char *wavFp, int n_frames);