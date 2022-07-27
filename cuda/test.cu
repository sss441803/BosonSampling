// How to compile: nvcc -o test test.cu -arch sm_50 -I ~/YHs_Sample/cnpy -I ./ -L/usr/local/lib -lcnpy -lz

#include"cnpy.h"

#include <kernel.h>
#include <helper.h>
#include <sort.h>

int main() {
    unsigned int m;
    unsigned int n;
    unsigned int k;
    unsigned int d = 20;
    unsigned int tau = 10;
    unsigned int n_iter = 1;
    bool chk = true;

    for (int i = 0; i < 100 && chk; i += 100){
        m = 10000 + i;
        n = 5000 + i;
        k = 5000 + i;

        float *h_U, *h_Glc, *h_Gcr, *h_LL, *h_LC, *h_LR, *h_T;
        int *h_CL, *h_CC, *h_CR;
        cudaMallocHost(&h_U, d*d*d * sizeof(float));
        cudaMallocHost(&h_Glc, m * k * sizeof(float));
        cudaMallocHost(&h_Gcr, k * n * sizeof(float));
        cudaMallocHost(&h_LL, m * sizeof(float));
        cudaMallocHost(&h_LC, k * sizeof(float));
        cudaMallocHost(&h_LR, n * sizeof(float));
        cudaMallocHost(&h_CL, m * sizeof(int));
        cudaMallocHost(&h_CC, k * sizeof(int));
        cudaMallocHost(&h_CR, n * sizeof(int));
        cudaMallocHost(&h_T, m * n * sizeof(float));
        random_init(h_U, d*d*d);
        random_init(h_Glc, m * k);
        random_init(h_Gcr, k * n);
        random_init(h_LL, m);
        random_init(h_LC, k);
        random_init(h_LR, n);
        random_init(h_CL, m, 0, d);
        random_init(h_CC, k, 0, d);
        random_init(h_CR, n, 0, d);

        float *d_U, *d_Glc, *d_Gcr, *d_LL, *d_LC, *d_LR, *d_T;
        int *d_CL, *d_CC, *d_CR;
        cudaMalloc(&d_U, d*d*d * sizeof(float));
        cudaMalloc(&d_Glc, m * k * sizeof(float));
        cudaMalloc(&d_Gcr, k * n * sizeof(float));
        cudaMalloc(&d_LL, m * sizeof(float));
        cudaMalloc(&d_LC, k * sizeof(float));
        cudaMalloc(&d_LR, n * sizeof(float));
        cudaMalloc(&d_CL, m * sizeof(int));
        cudaMalloc(&d_CC, k * sizeof(int));
        cudaMalloc(&d_CR, n * sizeof(int));
        cudaMalloc(&d_T, m * n * sizeof(float));

        cudaMemcpy(d_U, h_U, d*d*d * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_Glc, h_Glc, m * k * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_Gcr, h_Gcr, k * n * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_LL, h_LL, m * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_LC, h_LC, k * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_LR, h_LR, n * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(d_CL, h_CL, m * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_CC, h_CC, k * sizeof(int), cudaMemcpyDefault);
        cudaMemcpy(d_CR, h_CR, n * sizeof(int), cudaMemcpyDefault);

        SortedInfo sorted = sort(d_CL, d_CC, d_CR, m, k, n);
        int* incC = sorted.incC;

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        dim3 grid((n + 63) / 64, (m + 127) / 128);

        // warmup
        kernel<<<grid, 256>>>(
            d, tau, d_U, d_Glc, d_Gcr, d_LL, d_LC, d_LR, d_CL, d_CC, d_CR, incC, d_T, m, n, k, k * sizeof(float), n * sizeof(float) * 8);

        cudaEventRecord(start);
        for (int i = 0; i < n_iter; ++i) {
            kernel<<<grid, 256>>>(
                d, tau, d_U, d_Glc, d_Gcr, d_LL, d_LC, d_LR, d_CL, d_CC, d_CR, incC, d_T, m, n, k, k * sizeof(float), n * sizeof(float) * 8);
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