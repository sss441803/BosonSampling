// How to compile: nvcc -o test test.cu -arch sm_50 -I ~/YHs_Sample/cnpy -I ./ -L/usr/local/lib -lcnpy -lz
#include <../include/DataInit.cuh>
#include <../include/sort.cuh>
#include <../include/save.cuh>

int main() {
    
    unsigned int m = 1000;
    unsigned int n = 100;
    unsigned int k = 100;
    unsigned int d = 10;

    int *h_CL, *h_CR;
    cudaMallocHost(&h_CL, m * sizeof(int));
    cudaMallocHost(&h_CR, n * sizeof(int));

    // Initialize random charge values. To see if the edge case of no charge value is 0 works, change the 0 of the lower bound to an integer value larger than 1.
    random_init(h_CL, m, 2, d-1);
    random_init(h_CR, n, 2, d-1);

    for (int i = 0; i < m; ++i) {
        if (h_CR[i] == 5) {
            h_CR[i] = 6;
        }
    }

    int *d_CL, *d_CR;
    cudaMalloc(&d_CL, m * sizeof(int));
    cudaMalloc(&d_CR, n * sizeof(int));

    cudaMemcpy(d_CL, h_CL, m * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(d_CR, h_CR, n * sizeof(int), cudaMemcpyDefault);

    // Sorting and obtaining indices of charge increments
    SortedInfo sortedL = sort(d_CL, m);
    SortedInfo sortedR = sort(d_CR, m);
    int* d_incL = sortedL.inc;
    int* d_incR = sortedR.inc;
    int *incL, *incR;
    cudaMallocHost(&incL, d * sizeof(int));
    cudaMallocHost(&incR, d * sizeof(int));
    cudaMemcpy(incL, d_incL, d * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(incR, d_incR, d * sizeof(int), cudaMemcpyDefault);

    // Generating data
    NewData LL_data = left_align_init_1d(m, d, incL);
    float* LL = LL_data.data;
    unsigned int mNew = LL_data.m;

    NewData Glc_data = left_align_init(m, k, d, incL);
    float* Glc = Glc_data.data;

    NewData LR_data = right_align_init_1d(n, d, incR);
    float* LR = LR_data.data;
    unsigned int nNew = LR_data.n;

    NewData Gcr_data = right_align_init(k, n, d, incR);
    float* Gcr = Gcr_data.data;

    // Saving results to files
    save((std::string)"../out/CL.npy", h_CL, m);
    save((std::string)"../out/CR.npy", h_CR, n);
    save((std::string)"../out/incL.npy", incL, d);
    save((std::string)"../out/incR.npy", incR, d);
    save((std::string)"../out/LL.npy", LL, mNew);
    save((std::string)"../out/LR.npy", LR, nNew);
    save((std::string)"../out/Glc.npy", Glc, mNew, k);
    save((std::string)"../out/Gcr.npy", Gcr, k, nNew);

    // Freeing memory
    cudaFree(d_CL);
    cudaFree(d_CR);
    cudaFree(d_incR);
    cudaFree(d_incL);
    cudaFreeHost(h_CL);
    cudaFreeHost(h_CR);
    cudaFreeHost(incR);
    cudaFreeHost(incL);
}