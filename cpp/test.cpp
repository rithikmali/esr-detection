#include"inference.h"
using namespace std;
int main()
{
    const char* inp = "ex2.wav";

    //init the model weights
    gsl_matrix * alpha = gsl_matrix_alloc(5208, 100);
    gsl_matrix * beta = gsl_matrix_alloc(100,2);
    gsl_matrix * output = gsl_matrix_alloc(1,2);
    gsl_matrix * input = gsl_matrix_alloc(1,5208);

    //read wav file to double array
    SF_INFO inFileInfo;
    SNDFILE *file = sf_open(inp, SFM_READ, &inFileInfo);
    int n_frames = inFileInfo.frames;
    double a[n_frames];
    sf_read_double (file, a, n_frames);
    sf_close(file);

    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    int nt = ((n_frames-frameSize)/hop)+1;

    get_zcr(input, a, n_frames);
    get_stl_mfcc(input, inp, n_frames);

    //load the model weights from file
    my_load_matrix("alpha.txt", 100, alpha);
    my_load_matrix("beta.txt", 2, beta);

    //predict the output
    predict(input, alpha, beta, output);


    // Print the results
    printf("a00 %0.17g\n",a[0]);
    printf("in00 %0.17g\n",gsl_matrix_get (input, 0, 2));
    printf("in13,0 %0.17g\n\n",gsl_matrix_get (input, 0, nt*14-1));
    printf("pre00 %0.17g\n",gsl_matrix_get (output, 0, 0));
    printf("pre01 %0.17g\n",gsl_matrix_get (output, 0, 1));

    if (output->data[0] < output->data[1])
        cout << "\nsiren" << endl;
    else
        cout << "\nnot siren" << endl;

    //free memory
    gsl_matrix_free(alpha);
    gsl_matrix_free(beta);
    gsl_matrix_free(output);
    gsl_matrix_free(input);

    return 0;
}