#include"inference.h"
using namespace std;
int main(int argc, char* argv[])
{
    std::string USAGE = "./build/test : ESR detection\n";
    USAGE += "OPTIONS\n";
    USAGE += "--input           : Input 16 bit PCM Wave file\n";
    USAGE += "USAGE EXAMPLES\n";
    USAGE += "./build/test --input input.wav\n";

    char *inp = getCmdOption(argv, argv+argc, "--input");

    // Check arguments
    if (!inp) {
        std::cout << USAGE;
        return 1;
    }

    //check if file exists
    std::ifstream wavFp;
    wavFp.open(inp);
    if (!wavFp.is_open()) {
        std::cerr << "Unable to open input file: " << inp << std::endl;
        return 1;
    }
    wavFp.close();

    //init the model weights
    // gsl_matrix * alpha = gsl_matrix_alloc(5208, 100);
    // gsl_matrix * beta = gsl_matrix_alloc(100,2);
    // gsl_matrix * output = gsl_matrix_alloc(1,2);
    // gsl_matrix * input = gsl_matrix_alloc(1,5208);
    float alpha[5208][100];
    float beta[100][2];
    float output[1][2];
    float input[1][5208];

    //read wav file to double array
    // SF_INFO inFileInfo;
    // SNDFILE *file = sf_open(inp, SFM_READ, &inFileInfo);
    // int n_frames = inFileInfo.frames;
    // double a[n_frames];
    // sf_read_double (file, a, n_frames);
    // sf_close(file);

    AudioFile<float> audioFile;
    audioFile.load (inp);
    int n_frames = audioFile.getNumSamplesPerChannel();
    int sampleRate = audioFile.getSampleRate();
    float* a = &audioFile.samples[0][0];
    // double* a = &v[0];


    //set the samplerate, framesize and hop size
    int frameSize = 2048;
    int hop = 512;
    int nt = ((n_frames-frameSize)/hop)+1;

    // populate the input matriz with mfcc and zcr
    get_zcr(input, a, n_frames);
    get_stl_mfcc(input, inp, n_frames);

    //load the model weights from file
    my_load_matrix<5208, 100>("model/alpha.txt", 100, alpha);
    my_load_matrix<100, 2>("model/beta.txt", 2, beta);

    //predict the output
    float *input1D = (float *) input;
    float *output1D = (float *) output;
    float *alpha1D = (float *) alpha;
    float *beta1D = (float *) beta;
    predict(input1D, alpha1D, beta1D, output1D);
    
    // if (output->data[0] < output->data[1])
    //     cout << "\nsiren\n" << endl;
    // else
    //     cout << "\nnot siren\n" << endl;
    if (output[0][0] < output[0][1])
        cout << "\nsiren\n" << endl;
    else
        cout << "\nnot siren\n" << endl;

    //free memory
    // gsl_matrix_free(alpha);
    // gsl_matrix_free(beta);
    // gsl_matrix_free(output);
    // gsl_matrix_free(input);

    return 0;
}