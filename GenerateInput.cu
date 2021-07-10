#include<stdio.h>
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include<time.h>
#define BLOCK_SIZE 1024

__global__ void init_stuff(curandState *state, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );  
}

__global__ void generate(unsigned char * d_randstring, char * d_charset, curandState *state, int size, int length)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x ;
    
    if(length && id < length){
        curandState localState = state[id];
        float RANDOM = curand_uniform( &localState )*100000;
        int key = (int)ceil(RANDOM) % (size-1);
        d_randstring[id] = d_charset[key];
    }
}

int main(int argc, char ** argv)
{
    if(argc != 3){
        printf("Arguments should be input file and number of characters to be inserted. ");
        return -1;
    }
    char * filename = argv[1];
    FILE * inputfile = fopen(filename , "wb");
    int length = atoi(argv[2]);
    
    char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n,.-#'?! %$&()*+/:;<>=@[]^_{}|~";
    int size = strlen(charset);

    unsigned char * randstring ;
    randstring = (unsigned char *)malloc(sizeof(unsigned char)*(length+1));
    
    char * d_charset ;
    cudaMalloc(& d_charset, sizeof(char)*(size));
    cudaMemcpy(d_charset, charset, sizeof(char)*size, cudaMemcpyHostToDevice);
    
    unsigned char * d_randstring ;
    cudaMalloc(& d_randstring, sizeof(unsigned char)*(length+1));
    
    int nblocks;
    int nthreads;
    if(length <= 1024){
        nthreads = length;
        nblocks = 1;
    }
    else{
        nthreads = BLOCK_SIZE;
        nblocks = ceil( float(length) / nthreads);
    }
    
    curandState *d_state;
    cudaMalloc(&d_state , nthreads * nblocks);

    init_stuff<<<nblocks, nthreads >>>(d_state , time(NULL) );

    generate<<< nblocks, nthreads >>>(d_randstring, d_charset, d_state, size, length);

    cudaMemcpy(randstring, d_randstring , sizeof(char)*(length+1) , cudaMemcpyDeviceToHost);

    fwrite(randstring, sizeof(unsigned char), length, inputfile);
    fclose(inputfile);

    cudaFree(d_randstring);
    return 0;
}