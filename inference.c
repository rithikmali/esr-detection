#include"inference.h"


void load_matrix(int size; const char* filename, int size, gsl_matrix * m)
{
    FILE* fp = fopen(filename, "r");
    if (!fp)
        return;
    char cnum[30];
    int i = 0;
    int j = 0;
    int ci = 0;
    char c;
    char* ptr;
    while ((c = getc(fp)) != EOF)
    {
        if (c == ',')
        {
            cnum[ci] = '\0';
            gsl_matrix_set (m, i, j, atof(cnum));
            // gsl_matrix_set (m, i, j, strtod(cnum,&ptr));
            if (j == size - 1)
            {
                ++i;
                j = 0;
            }
            else
            {
                ++j;
            }
            ci = 0;
        }
        else
        {
            cnum[ci] = c;
            ++ci;
        }
    }
    fclose(fp);
}

void predict(gsl_matrix* input, gsl_matrix* alpha, gsl_matrix* beta, gsl_matrix* output)
{
    gsl_matrix* h = gsl_matrix_alloc(272,100);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input, alpha, 0.0, h);
    for(int i=0; i<272; ++i)
    {
        for(int j=0; j<100; ++j)
        {
            double x = gsl_matrix_get(h,i,j);
            gsl_matrix_set(h,i,j,sigmoid(x));
        }
    }
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, h, beta, 0.0, output);
    gsl_matrix_free(h);
}

double sigmoid(double x)
{
    double d1 = 1.0;
    return d1 / (d1 + gsl_expm1(-x) + d1);
}