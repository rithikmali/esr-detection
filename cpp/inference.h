#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include<stdio.h>
#include<stdlib.h>
#include <vector>
#include <sndfile.h>
#include <Gist.h>
#include<iostream>
void my_load_matrix(const char* filename, int size, gsl_matrix * m);
double sigmoid(double x);
void predict(gsl_matrix* input, gsl_matrix* alpha, gsl_matrix* beta, gsl_matrix* output);