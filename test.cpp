#include"inference.hpp"

int main()
{
    //init the model weights
    gsl_matrix * alpha = gsl_matrix_alloc(5264, 100);
    gsl_matrix * beta = gsl_matrix_alloc(100,2);
    gsl_matrix * output = gsl_matrix_alloc(272,2);
    gsl_matrix * input = gsl_matrix_alloc(1,5264);

    //read wav file to double array
    SF_INFO inFileInfo;
    SNDFILE *file = sf_open("new.wav", SFM_READ, &inFileInfo);
    int n_frames = inFileInfo.frames;
    double a[n_frames];
    sf_read_double (file, a, n_frames);
    sf_close(file);

    //gist setup
    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    Gist<double> gist (frameSize, sampleRate);
    int nt = (n_frames-frameSize)/hop;

    double audioFrame[frameSize];
    for (int i = 0; i < n_frames-frameSize; i+=hop)
    {
        for(int j=0; j<frameSize; ++j)
        {
            audioFrame[j] = a[i+j];
        }
        gist.processAudioFrame (audioFrame, frameSize);
        double zcr = gist.zeroCrossingRate();
        const std::vector<double>& mfcc = gist.getMelFrequencyCepstralCoefficients();
        for (int j=0; j<mfcc.size(); ++j)
        {
            gsl_matrix_set(input, 1, (i/hop)+j, mfcc[j]);
        }
        gsl_matrix_set(input, 1, (nt*13) + i/hop, zcr);
    }


    //load the model weights from file
    my_load_matrix("alpha.txt", 100, alpha);
    my_load_matrix("beta.txt", 2, beta);
    //load_matrix("x_test.txt",5264,input);

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