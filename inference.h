#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include<stdio.h>
#include<stdlib.h>

void load_matrix(int size; const char* filename, int size, gsl_matrix * m);
double sigmoid(double x);
void predict(gsl_matrix* input, gsl_matrix* alpha, gsl_matrix* beta, gsl_matrix* output);