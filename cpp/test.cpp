#include"inference.h"
using namespace std;
int main()
{
    //init the model weights
    gsl_matrix * alpha = gsl_matrix_alloc(5264, 100);
    gsl_matrix * beta = gsl_matrix_alloc(100,2);
    gsl_matrix * output = gsl_matrix_alloc(1,2);
    gsl_matrix * input = gsl_matrix_alloc(1,5264);

    //read wav file to double array
    SF_INFO inFileInfo;
    SNDFILE *file = sf_open("ex.wav", SFM_READ, &inFileInfo);
    int n_frames = inFileInfo.frames;
    double a[n_frames];
    sf_read_double (file, a, n_frames);
    sf_close(file);

    int pn_frames = n_frames+2048;
    double pa[pn_frames];
    for (int i = 0; i < pn_frames; ++i)
    {
        if (i<1024)
        {
            pa[i] = a[0];
        }
        else if (i>n_frames+1024)
        {
            pa[i] = a[n_frames-1];
        }
        else
        {
            pa[i] = a[i-1024];
        }
    }

    //gist setup
    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    Gist<double> gist (frameSize, sampleRate);
    int nt = ((pn_frames-frameSize)/hop)+1;
    cout<<"nt: "<<nt<<'\n';

    double audioFrame[frameSize];
    for (int i = 0; i < pn_frames-frameSize; i+=hop)
    {
        for(int j=0; j<frameSize; ++j)
        {
            audioFrame[j] = pa[i+j];
        }
        gist.processAudioFrame (audioFrame, frameSize);
        double zcr = gist.zeroCrossingRate();
        // printf ("%f\n", zcr);
        const std::vector<double>& mfcc = gist.getMelFrequencyCepstralCoefficients();
        for (int j=0; j<mfcc.size(); ++j)
        {
            gsl_matrix_set(input, 0, (j*nt)+(i/hop), mfcc[j]);
        }
        gsl_matrix_set(input, 0, (nt*13) + i/hop, zcr/frameSize);
        
        // if (i==3*hop)
        //     return 0;
    }


    //load the model weights from file
    my_load_matrix("alpha.txt", 100, alpha);
    my_load_matrix("beta.txt", 2, beta);
    //load_matrix("x_test.txt",5264,input);

    //predict the output
    predict(input, alpha, beta, output);

    // Print the results
    printf("00 %0.17g\n",a[0]);
    printf("in00 %0.17g\n",gsl_matrix_get (input, 0, 0));
    printf("in13,0 %0.17g\n",gsl_matrix_get (input, 0, nt*13));
    printf("pre00 %0.17g\n",gsl_matrix_get (output, 0, 0));
    printf("pre01 %0.17g\n",gsl_matrix_get (output, 0, 1));

    //free memory
    gsl_matrix_free(alpha);
    gsl_matrix_free(beta);
    gsl_matrix_free(output);
    gsl_matrix_free(input);

    return 0;
}