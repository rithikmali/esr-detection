#include"inference.h"

// A simple option parser
char* getCmdOption(char **begin, char **end, const std::string &value) {
    char **iter = std::find(begin, end, value);
    if (iter != end && ++iter != end)
        return *iter;
    return nullptr;
}

//load matrix from text file
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

//predict the output from given input
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

//populate input matrix with zcr
void get_zcr(gsl_matrix* input, double* a, int n_frames)
{

    int frameSize = 2048;
    int sampleRate = 48000;
    int hop = 512;
    // Gist<double> gist(frameSize, sampleRate);
    int nt = ((n_frames-frameSize)/hop)+1;

    

    // double audioFrame[frameSize];
    for (int i = 0; i < n_frames-frameSize; i+=hop)
    {
        double zcr = 0;
        
        // for each audio sample, starting from the second one
        for (int j = i+1; j < i+frameSize; j++)
        {
            // initialise two booleans indicating whether or not
            // the current and previous sample are positive
            bool current = (a[j] > 0);
            bool previous = (a[j - 1] > 0);

            // if the sign is different
            if (current != previous)
            {
                // add one to the zero crossing rate
                zcr = zcr + 1.0;
            }
        }
        
        // gist.processAudioFrame (audioFrame, frameSize);
        // double zcr = gist.zeroCrossingRate();
        gsl_matrix_set(input, 0, (nt*13) + i/hop, zcr/frameSize);
    }
}

//populate input matrix with mfcc
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
