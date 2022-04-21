#include"inference.h"

int main()
{
    //init the model weights
    gsl_matrix * alpha = gsl_matrix_alloc(7896, 100);
    gsl_matrix * beta = gsl_matrix_alloc(100,2);
    gsl_matrix * output = gsl_matrix_alloc(272,2);
    gsl_matrix * input = gsl_matrix_alloc(272,7896);

    //load the model weights from file
    load_matrix("alpha.txt", 100, alpha);
    load_matrix("beta.txt", 2, beta);
    load_matrix("x_test.txt",7896,input);

    //predict the output
    predict(input, alpha, beta, output);

    // Print the results
    printf("00 %0.17g\n",gsl_matrix_get (alpha, 0, 0));
    printf("pre00 %0.17g\n",gsl_matrix_get (output, 0, 0));
    printf("pre01 %0.17g\n",gsl_matrix_get (output, 0, 1));

    //free memory
    gsl_matrix_free(alpha);
    gsl_matrix_free(beta);
    gsl_matrix_free(output);
    gsl_matrix_free(input);

    return 0;
}