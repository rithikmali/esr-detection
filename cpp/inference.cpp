#include"inference.h"


void my_load_matrix( const char* filename, int size, gsl_matrix * m)
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
    gsl_matrix* h = gsl_matrix_alloc(1,100);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input, alpha, 0.0, h);
    for(int i=0; i<1; ++i)
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

void get_zcr(gsl_matrix* input, double* a, int n_frames)
{

    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    Gist<double> gist(frameSize, sampleRate);
    int nt = ((n_frames-frameSize)/hop)+1;

    // double pa[pn_frames];
    // for (int i = 0; i < pn_frames; ++i)
    // {
    //     if (i<1024)
    //     {
    //         pa[i] = a[0];
    //     }
    //     else if (i>n_frames+1024)
    //     {
    //         pa[i] = a[n_frames-1];
    //     }
    //     else
    //     {
    //         pa[i] = a[i-1024];
    //     }
    // }

    double audioFrame[frameSize];
    for (int i = 0; i < n_frames-frameSize; i+=hop)
    {
        for(int j=0; j<frameSize; ++j)
        {
            audioFrame[j] = a[i+j];
        }
        gist.processAudioFrame (audioFrame, frameSize);
        double zcr = gist.zeroCrossingRate();
        gsl_matrix_set(input, 0, (nt*13) + i/hop, zcr/frameSize);
    }
}

void get_mfcc(gsl_matrix * input, double* a, int n_frames)
{

    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    Gist<double> gist (frameSize, sampleRate);
    int nt = ((n_frames-frameSize)/hop)+1;

    double audioFrame[frameSize];
    for (int i = 0; i < n_frames-frameSize; i+=hop)
    {
        for(int j=0; j<frameSize; ++j)
        {
            audioFrame[j] = a[i+j];
        }
        gist.processAudioFrame (audioFrame, frameSize);

        const std::vector<double>& mfcc = gist.getMelFrequencyCepstralCoefficients();
        for (int j=0; j<mfcc.size(); ++j)
        {
            gsl_matrix_set(input, 0, (j*nt)+(i/hop), mfcc[j]);
        }
    }
}

void get_mfcc1(gsl_matrix * input, double* a, int n_frames)
{

    int pn_frames = n_frames+2048;

    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    Gist<double> gist(frameSize, sampleRate);
    int nt = ((pn_frames-frameSize)/hop)+1;

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

    double audioFrame[frameSize];
    for (int i = 0; i < pn_frames-frameSize; i+=hop)
    {
        for(int j=0; j<frameSize; ++j)
        {
            audioFrame[j] = pa[i+j];
        }
        gist.processAudioFrame (audioFrame, frameSize);

        const std::vector<double>& mfcc = gist.getMelFrequencyCepstralCoefficients();
        if(i==0)
            printf("mfcc size: %ld\n",mfcc.size());
        for (int j=0; j<mfcc.size(); ++j)
        {
            gsl_matrix_set(input, 0, (j*nt)+(i/hop), mfcc[j]);
        }
    }
}

void get_stl_mfcc(gsl_matrix * input, const char *wavPath, int n_frames)
{
    int numCepstra = 12;
    int numFilters = 40;
    int samplingRate = 48000;
    int winLength = 2048;
    int frameShift = 512;
    int lowFreq = 50;
    int highFreq = samplingRate/2;
    std::ifstream wavFp;

    wavFp.open(wavPath);
    myMFCC mfccComputer (samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);
    
    mfccComputer.process (wavFp, input, n_frames);
    wavFp.close();

}
