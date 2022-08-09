// How to compile: nvcc -o main main.cu ./src/kernel.cu ./src/DataInit.cu ./src/check.cu ./src/sort.cu -arch sm_50 -I ~/YHs_Sample/cnpy -I ./include -L/usr/local/lib -lcnpy
#include <assert.h>

#include <cnpy.h>

#include <kernel.cuh>
#include <check.cuh>
#include <sort.cuh>
#include <DataInit.cuh>

int main() {
    unsigned int m;
    unsigned int n;
    unsigned int k;
    unsigned int d = 20;
    unsigned int tau = 10;
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
        int* h_idL, *h_incL;
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

        // Center
        // Sorting
        SortedInfo sortedC = sort(d_CC, n);
        int* d_incC = sortedC.inc;
        int* d_idC = sortedC.id;
        int *h_idC, *h_incC;
        cudaMallocHost(&h_idC, k * sizeof(int));
        cudaMallocHost(&h_incC, k * sizeof(int));
        cudaMemcpy(h_idC, d_idC, k * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(h_incC, d_incC, k * sizeof(int), cudaMemcpyDefault);
        // Reindexing
        RemapInfo remapC = index_remapping(k, d, h_idC, h_incC);
        unsigned int sizeNewC = remapC.size;
        int* h_indexNewC = remapC.index;
        int* h_incNewC = remapC.inc;
        int* h_cNewC = remapC.c;
        // Transfering to device
        int *d_incNewC, *d_cNewC;
        cudaMalloc(&d_incNewC, d * sizeof(int));
        cudaMalloc(&d_cNewC, sizeNewC * sizeof(int));
        cudaMemcpy(d_incNewC, h_incNewC, d * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_cNewC, h_cNewC, sizeNewC * sizeof(int), cudaMemcpyDefault);

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
        kernel<<<grid, 256>>>(
            d, tau, d_U, d_Glc, d_Gcr, d_LL, d_LC, d_LR, d_cNewL, d_cNewC, d_cNewR, d_incC, d_T, sizeNewL, sizeNewR, k, k * sizeof(float), sizeNewR * sizeof(float) * 8);

        cudaEventRecord(start);
        for (int i = 0; i < n_iter; ++i) {
            kernel<<<grid, 256>>>(
                d, tau, d_U, d_Glc, d_Gcr, d_LL, d_LC, d_LR, d_cNewL, d_cNewC, d_cNewR, d_incC, d_T, sizeNewL, sizeNewR, k, k * sizeof(float), sizeNewR * sizeof(float) * 8);
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
        cnpy::npy_save("U.npy", &h_U[0], {d, d, d }, "w");
        cnpy::npy_save("A.npy", &h_Glc[0], {m, k}, "w");
        cnpy::npy_save("B.npy", &h_Gcr[0], {k, n}, "w");
        cnpy::npy_save("LL.npy", &h_LL[0], {m}, "w");
        cnpy::npy_save("LC.npy", &h_LC[0], {k}, "w");
        cnpy::npy_save("LR.npy", &h_LR[0], {n}, "w");
        cnpy::npy_save("C.npy", &h_T[0], {m, n}, "w");

        cudaFreeHost(h_U);
        cudaFreeHost(h_Glc);
        cudaFreeHost(h_Gcr);
        cudaFreeHost(h_LL);
        cudaFreeHost(h_LC);
        cudaFreeHost(h_LR);
        cudaFreeHost(h_T);
    }
}