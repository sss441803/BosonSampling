// How to compile: nvcc -o test test.cu -arch sm_50 -I ~/YHs_Sample/cnpy -I ./ -L/usr/local/lib -lcnpy -lz

#include"cnpy.h"

#include <TwoDInit.h>
#include <sort.h>

int main() {
    unsigned int m = 17;
    unsigned int n = 21;
    unsigned int d = 3;

    int *h_CL, *h_CR;
    cudaMallocHost(&h_CL, m * sizeof(int));
    cudaMallocHost(&h_CR, n * sizeof(int));

    random_init(h_CL, m, 0, d-1);
    random_init(h_CR, n, 0, d-1);

    int *d_CL, *d_CR;
    cudaMalloc(&d_CL, m * sizeof(int));
    cudaMalloc(&d_CR, n * sizeof(int));

    cudaMemcpy(d_CL, h_CL, m * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(d_CR, h_CR, n * sizeof(int), cudaMemcpyDefault);

    SortedInfo sortedL = sort(d_CL, m);
    SortedInfo sortedR = sort(d_CR, m);
    int* d_incL = sortedL.inc;
    int* d_incR = sortedR.inc;
    int *incL, *incR;
    cudaMallocHost(&incL, d * sizeof(int));
    cudaMallocHost(&incR, d * sizeof(int));
    cudaMemcpy(incL, d_incL, d * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(incR, d_incR, d * sizeof(int), cudaMemcpyDefault);

    NewData new_data = random_init_2D(m, n, d, incL, incR);
    float* data = new_data.data;
    unsigned int mNew =  new_data.m;
    unsigned int nNew = new_data.n;

    cudaFree(d_CL);
    cudaFree(d_CR);
    cudaFree(d_incR);
    cudaFree(d_incL);

    //save results to file
    cnpy::npy_save("data.npy", &data[0], {mNew, nNew}, "w");

    cudaFreeHost(h_CL);
    cudaFreeHost(h_CR);
    cudaFreeHost(incR);
    cudaFreeHost(incL);
}