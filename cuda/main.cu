// How to compile: nvcc -o test test.cu -arch sm_50 -I ~/YHs_Sample/cnpy -I ./ -L/usr/local/lib -lcnpy -lz
#include <assert.h>

#include <kernel.cuh> // Kernel
#include <sort.cuh> // Sorting and index arrangement functions
#include <DataInit.cuh> // Initialization of data arrays
#include <check.cuh> // Checking the answer
#include <save.cuh> // Saving data as numpy arrays

int main() {
    unsigned int m;
    unsigned int n;
    unsigned int k;
    unsigned int d = 10;
    unsigned int tau = 5;
    unsigned int n_iter = 1;
    bool chk = true;

    for (int i = 0; i < 100 && chk; i += 100){
        m = 1000 + i;
        n = 1000 + i;
        k = 1000 + i;

        //////////////////////
        // Data preparation //
        //////////////////////
        float *h_U, *h_T;
        int *h_CL, *h_CC, *h_CR;
        cudaMallocHost(&h_U, d*d*d * sizeof(float));
        cudaMallocHost(&h_CL, m * sizeof(int));
        cudaMallocHost(&h_CC, k * sizeof(int));
        cudaMallocHost(&h_CR, n * sizeof(int));
        cudaMallocHost(&h_T, m * n * sizeof(float));
        random_init(h_U, d*d*d);
        random_init(h_CL, m, 0, d);
        random_init(h_CC, k, 0, d);
        random_init(h_CR, n, 0, d);

        float *d_U, *d_T;
        int *d_CL, *d_CC, *d_CR;
        cudaMalloc(&d_U, d*d*d * sizeof(float));
        cudaMalloc(&d_CL, m * sizeof(int));
        cudaMalloc(&d_CC, k * sizeof(int));
        cudaMalloc(&d_CR, n * sizeof(int));
        cudaMalloc(&d_T, m * n * sizeof(float));

        cudaMemcpy(d_U, h_U, d*d*d * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_CL, h_CL, m * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_CC, h_CC, k * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_CR, h_CR, n * sizeof(int), cudaMemcpyDefault);
        /////////////////////////////
        // End of data preparation //
        /////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////
        // Index reorganization (sorting charges and aligning charge changes to multiples of 8) //
        //////////////////////////////////////////////////////////////////////////////////////////
        // Left side
        // Sorting
        SortedInfo sortedL = sort(d_CL, m);
        int* d_incL = sortedL.inc;
        int* d_idL = sortedL.id;
        int *h_idL, *h_incL;
        cudaMallocHost(&h_idL, m * sizeof(int));
        cudaMallocHost(&h_incL, m * sizeof(int));
        cudaMemcpy(h_idL, d_idL, m * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(h_incL, d_incL, m * sizeof(int), cudaMemcpyDefault);
        // Reindexing
        RemapInfo remapL = index_remapping(m, d, h_idL, h_incL);
        unsigned int sizeNewL = remapL.size;
        int* h_indexNewL = remapL.index;
        int* h_incNewL = remapL.inc;
        int* h_cNewL = remapL.c;
        // Transfering to device
        int *d_incNewL, *d_cNewL;
        cudaMalloc(&d_incNewL, d * sizeof(int));
        cudaMalloc(&d_cNewL, sizeNewL * sizeof(int));
        cudaMemcpy(d_incNewL, h_incNewL, d * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_cNewL, h_cNewL, sizeNewL * sizeof(int), cudaMemcpyDefault);

        // Center (doesn't need to be aligned)
        // Sorting
        SortedInfo sortedC = sort(d_CC, n);
        int* d_incC = sortedC.inc;
        int* d_idC = sortedC.id;

        // Right side
        // Sorting
        SortedInfo sortedR = sort(d_CR, n);
        int* d_incR = sortedR.inc;
        int* d_idR = sortedR.id;
        int *h_idR, *h_incR;
        cudaMallocHost(&h_idR, n * sizeof(int));
        cudaMallocHost(&h_incR, n * sizeof(int));
        cudaMemcpy(h_idR, d_idR, n * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(h_incR, d_incR, n * sizeof(int), cudaMemcpyDefault);
        // Reindexing
        RemapInfo remapR = index_remapping(m, d, h_idR, h_incR);
        unsigned int sizeNewR = remapR.size;
        int* h_indexNewR = remapR.index;
        int* h_incNewR = remapR.inc;
        int* h_cNewR = remapR.c;
        // Transfering to device
        int *d_incNewR, *d_cNewR;
        cudaMalloc(&d_incNewR, d * sizeof(int));
        cudaMalloc(&d_cNewR, sizeNewR * sizeof(int));
        cudaMemcpy(d_incNewR, h_incNewR, d * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_cNewR, h_cNewR, sizeNewR * sizeof(int), cudaMemcpyDefault);
        ///////////////////////
        // End of reindexing //
        ///////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Initializing Lambdas and Gammas according to new charge alignment //
        ///////////////////////////////////////////////////////////////////////
        // Obtaining lambdas and gammas on host
        // Lambda center doesn't need alignment
        float* h_LC;
        cudaMallocHost(&h_LC, k * sizeof(float));
        random_init(h_LC, k);
        // Lambda left, right and gammas need alignment
        NewData LL_data = left_align_init_1d(m, d, h_incL);
        NewData LR_data = right_align_init_1d(n, d, h_incR);
        NewData Glc_data = left_align_init(m, k, d, h_incL);
        NewData Gcr_data = right_align_init(k, n, d, h_incR);
        float* h_LL = LL_data.data;
        float* h_LR = LR_data.data;
        float* h_Glc = Glc_data.data;
        float* h_Gcr = Gcr_data.data;
        //assert (sizeNewL == Glc_data.m);
        //assert (sizeNewR == Gcr_data.n);
        // Moving lambdas and gammas to device
        float *d_LL, *d_LC, *d_LR, *d_Glc, *d_Gcr;
        cudaMalloc(&d_LL, sizeNewL * sizeof(float));
        cudaMalloc(&d_LC, k * sizeof(float));
        cudaMalloc(&d_LR, sizeNewR * sizeof(float));
        cudaMalloc(&d_Glc, sizeNewL * k * sizeof(float));
        cudaMalloc(&d_Gcr, k * sizeNewR * sizeof(float));
        cudaMemcpy(d_LL, h_LL, sizeNewL * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_LC, h_LC, k * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_LR, h_LR, sizeNewR * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_Glc, h_Glc, sizeNewL * k * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_Gcr, h_Gcr, k * sizeNewR * sizeof(float), cudaMemcpyDefault);
        //cnpy::npy_save("Glc.npy", &h_Glc[0], {sizeNewL, k}, "w");
        //cnpy::npy_save("Gcr.npy", &h_Gcr[0], {k, sizeNewR}, "w");
        ////////////////////////////////////////
        // End of Lambda Gamma initialization //
        ////////////////////////////////////////

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        dim3 grid((sizeNewR + 63) / 64, (sizeNewL + 127) / 128);

        // warmup
        /*
        d: Maximum number of photons. Local (mode) dimensionality of the Hilbert space
        tau: the center charge.
        U: Unitary
        Glc: Gamma for left and center; Gcr: Gamma for center and right
        LL: Lambda for left; LC: lambda for center; LR: lambda for right
        cNewL: aligned charge for left; CC: sorted but unaligned charge for center; cNewR: aligned charge for right
        incC: charge increment index for center
        T: result theta matrix
        sizeNewL: m dimension after alignment; sizeNewR: n dimension after alignment
        */
        kernel<<<grid, 256>>>(
            d, tau, d_U, d_Glc, d_Gcr, d_LL, d_LC, d_LR, d_cNewL, d_CC, d_cNewR, d_incC, d_T, sizeNewL, sizeNewR, k, k * sizeof(float), sizeNewR * sizeof(float) * 8);

        cudaEventRecord(start);
        for (int i = 0; i < n_iter; ++i) {
            kernel<<<grid, 256>>>(
                d, tau, d_U, d_Glc, d_Gcr, d_LL, d_LC, d_LR, d_cNewL, d_CC, d_cNewR, d_incC, d_T, sizeNewL, sizeNewR, k, k * sizeof(float), sizeNewR * sizeof(float) * 8);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        long workload = n_iter * long(m) * n * k * 2;
        double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
        printf("Performance: %fGFLOPS; total time %fms\n", gflops, ms/n_iter);

        cudaMemcpy(h_T, d_T, m * n * sizeof(float), cudaMemcpyDefault);

        cudaFree(d_U);
        cudaFree(d_Glc);
        cudaFree(d_Gcr);
        cudaFree(d_LL);
        cudaFree(d_LC);
        cudaFree(d_LR);
        cudaFree(d_T);

        chk = check(h_U, h_Glc, h_Gcr, h_LL, h_LC, h_LR, h_T, m, n, k); 

        //save results to file
        save((std::string)"./out/U.npy", h_U, d, d, d);
        save((std::string)"./out/A.npy", h_Glc, m, k);
        save((std::string)"./out/B.npy", h_Gcr, k, n);
        save((std::string)"./out/LL.npy", h_LL, m);
        save((std::string)"./out/LC.npy", h_LC, k);
        save((std::string)"./out/LR.npy", h_LR, n);
        save((std::string)"./out/C.npy", h_T, m, n);

        cudaFreeHost(h_U);
        cudaFreeHost(h_Glc);
        cudaFreeHost(h_Gcr);
        cudaFreeHost(h_LL);
        cudaFreeHost(h_LC);
        cudaFreeHost(h_LR);
        cudaFreeHost(h_T);
    }
}