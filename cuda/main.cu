// How to compile: nvcc -o test test.cu -arch sm_50 -I ~/YHs_Sample/cnpy -I ./ -L/usr/local/lib -lcnpy -lz
#include <assert.h>
#include <cstdio>

#include <kernel.cuh> // Kernel
#include <sort.cuh> // Sorting and index arrangement functions
#include <DataInit.cuh> // Initialization of data arrays
#include <check.cuh> // Checking the answer
#include <save.cuh> // Saving data as numpy arrays

int main() {
    unsigned int m;
    unsigned int n;
    unsigned int k;
    unsigned int d = 5;
    unsigned int tau = d/2;
    assert (d >= tau);
    unsigned int n_iter = 1;
    bool chk = true;

    for (int i = 0; i < 300 && chk; i += 3) {
        m = 5000 + i;

        for (int j = 0; j < 300; j += 3) {
            n = 5000 + j;
            d = min(m, n)/100 + 1;
            tau = d/2;
            printf("\nm: %i, n: %i, d: %i, tau: %i\n", m, n, d, tau);

            for (int l = 0; l < 3; l += 3) {
                k = 5000 + l;

                //////////////////////
                // Data preparation //
                //////////////////////
                // Initialize U
                float *h_U_r, *h_U_i;
                cudaMallocHost(&h_U_r, d*d*d * sizeof(float));
                cudaMallocHost(&h_U_i, d*d*d * sizeof(float));
                random_init(h_U_r, d*d*d);
                random_init(h_U_i, d*d*d);
                // Fill charges with random integers. A d dimensional Hilbert space has from 0 to d-1 charges possible.
                int *h_CL, *h_CC, *h_CR;
                cudaMallocHost(&h_CL, m * sizeof(int));
                cudaMallocHost(&h_CC, k * sizeof(int));
                cudaMallocHost(&h_CR, n * sizeof(int));
                random_init(h_CL, m, tau, d-1);
                random_init(h_CC, k, 0, d-1);
                random_init(h_CR, n, 0, tau);
                // Transfer data from cpu to gpu
                float *d_U_r, *d_U_i;
                cudaMalloc(&d_U_r, d*d*d * sizeof(float));
                cudaMalloc(&d_U_i, d*d*d * sizeof(float));
                cudaMemcpy(d_U_r, h_U_r, d*d*d * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_U_i, h_U_i, d*d*d * sizeof(float), cudaMemcpyDefault);
                int *d_CL, *d_CC, *d_CR;
                cudaMalloc(&d_CL, m * sizeof(int));
                cudaMalloc(&d_CC, k * sizeof(int));
                cudaMalloc(&d_CR, n * sizeof(int));
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
                SortedInfo sortedL = sort(d, m, d_CL);
                int* d_incL = sortedL.inc;
                int* d_idL = sortedL.id;
                int *h_idL, *h_incL;
                cudaMallocHost(&h_idL, m * sizeof(int));
                cudaMallocHost(&h_incL, m * sizeof(int));
                cudaMemcpy(h_idL, d_idL, m * sizeof(int), cudaMemcpyDefault);
                cudaMemcpy(h_incL, d_incL, d * sizeof(int), cudaMemcpyDefault);
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
                SortedInfo sortedC = sort(d, k, d_CC);
                cudaMemcpy(h_CC, d_CC, k * sizeof(int), cudaMemcpyDefault);
                int* d_incC = sortedC.inc;
                int* d_idC = sortedC.id;
                int* h_incC;
                cudaMallocHost(&h_incC, d * sizeof(int));
                cudaMemcpy(h_incC, d_incC, d * sizeof(int), cudaMemcpyDefault);

                // Right side
                // Sorting
                SortedInfo sortedR = sort(d, n, d_CR);
                int* d_incR = sortedR.inc;
                int* d_idR = sortedR.id;
                int *h_idR, *h_incR;
                cudaMallocHost(&h_idR, n * sizeof(int));
                cudaMallocHost(&h_incR, n * sizeof(int));
                cudaMemcpy(h_idR, d_idR, n * sizeof(int), cudaMemcpyDefault);
                cudaMemcpy(h_incR, d_incR, d * sizeof(int), cudaMemcpyDefault);
                // Reindexing
                RemapInfo remapR = index_remapping(n, d, h_idR, h_incR);
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
                float *h_LC;
                cudaMallocHost(&h_LC, k * sizeof(float));
                random_init(h_LC, k);
                // Lambda left, right and gammas need alignment
                NewData LL_data = left_align_init_1d(m, d, h_incL);
                NewData LR_data = right_align_init_1d(n, d, h_incR);
                NewData Glc_r_data = left_align_init(m, k, d, h_incL);
                NewData Glc_i_data = left_align_init(m, k, d, h_incL);
                NewData Gcr_r_data = right_align_init(k, n, d, h_incR);
                NewData Gcr_i_data = right_align_init(k, n, d, h_incR);
                float* h_LL = LL_data.data;
                float* h_LR = LR_data.data;
                float* h_Glc_r = Glc_r_data.data;
                float* h_Glc_i = Glc_i_data.data;
                float* h_Gcr_r = Gcr_r_data.data;
                float* h_Gcr_i = Gcr_i_data.data;
                // Making sure remapping gives the same sizes as DataInit
                assert (sizeNewL == Glc_r_data.m);
                assert (sizeNewR == Gcr_r_data.n);
                // Moving lambdas and gammas to device
                float *d_LL, *d_LC, *d_LR, *d_Glc_r, *d_Gcr_r,*d_Glc_i, *d_Gcr_i;
                cudaMalloc(&d_LL, sizeNewL * sizeof(float));
                cudaMalloc(&d_LC, k * sizeof(float));
                cudaMalloc(&d_LR, sizeNewR * sizeof(float));
                cudaMalloc(&d_Glc_r, sizeNewL * k * sizeof(float));
                cudaMalloc(&d_Glc_i, sizeNewL * k * sizeof(float));
                cudaMalloc(&d_Gcr_r, k * sizeNewR * sizeof(float));
                cudaMalloc(&d_Gcr_i, k * sizeNewR * sizeof(float));
                cudaMemcpy(d_LL, h_LL, sizeNewL * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_LC, h_LC, k * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_LR, h_LR, sizeNewR * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_Glc_r, h_Glc_r, sizeNewL * k * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_Glc_i, h_Glc_i, sizeNewL * k * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_Gcr_r, h_Gcr_r, k * sizeNewR * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(d_Gcr_i, h_Gcr_i, k * sizeNewR * sizeof(float), cudaMemcpyDefault);
                ////////////////////////////////////////
                // End of Lambda Gamma initialization //
                ////////////////////////////////////////

                // Initializing results array
                float *h_T_r, *h_T_i, *d_T_r, *d_T_i;
                cudaMallocHost(&h_T_r, sizeNewL * sizeNewR * sizeof(float));
                cudaMallocHost(&h_T_i, sizeNewL * sizeNewR * sizeof(float));
                cudaMalloc(&d_T_r, sizeNewL * sizeNewR * sizeof(float));
                cudaMalloc(&d_T_i, sizeNewL * sizeNewR * sizeof(float));

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
                    d, tau, d_U_r, d_U_i, d_Glc_r, d_Glc_i, d_Gcr_r, d_Gcr_i, d_LL, d_LC, d_LR, d_cNewL, d_CC, d_cNewR, d_incC, d_T_r, d_T_i, sizeNewL, sizeNewR, k, k * sizeof(float), sizeNewR * sizeof(float) * 8);

                cudaEventRecord(start);
                for (int i = 0; i < n_iter; ++i) {
                    kernel<<<grid, 256>>>(
                        d, tau, d_U_r, d_U_i, d_Glc_r, d_Glc_i, d_Gcr_r, d_Gcr_i, d_LL, d_LC, d_LR, d_cNewL, d_CC, d_cNewR, d_incC, d_T_r, d_T_i, sizeNewL, sizeNewR, k, k * sizeof(float), sizeNewR * sizeof(float) * 8);
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

                cudaMemcpy(h_T_r, d_T_r, sizeNewL * sizeNewR * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(h_T_i, d_T_i, sizeNewL * sizeNewR * sizeof(float), cudaMemcpyDefault);

                cudaFree(d_CL);
                cudaFree(d_CC);
                cudaFree(d_CR);
                cudaFree(d_idL);
                cudaFree(d_idC);
                cudaFree(d_idR);
                cudaFree(d_incL);
                cudaFree(d_incR);
                cudaFree(d_incNewL);
                cudaFree(d_incNewR);

                cudaFree(d_U_r);
                cudaFree(d_U_i);
                cudaFree(d_Glc_r);
                cudaFree(d_Glc_i);
                cudaFree(d_Gcr_r);
                cudaFree(d_Gcr_i);
                cudaFree(d_LL);
                cudaFree(d_LC);
                cudaFree(d_LR);
                cudaFree(d_cNewL);
                cudaFree(d_CC);
                cudaFree(d_cNewR);
                cudaFree(d_incC);
                cudaFree(d_T_r);
                cudaFree(d_T_i);

                //chk = check(h_U, h_Glc, h_Gcr, h_LL, h_LC, h_LR, h_T, sizeNewL, sizeNewR, k);
                chk = check(h_T_r, h_T_i, d, tau, h_U_r, h_U_i, h_Glc_r, h_Glc_i, h_Gcr_r, h_Gcr_i, h_LL, h_LC, h_LR, h_cNewL, h_CC, h_cNewR, sizeNewL, sizeNewR, k); 
                if (!chk) { printf("Failed at m %i, n %i, k %i", m, n, k); }
                //save results to file
                if (chk) {
                    save((std::string)"./out/U_r.npy", h_U_r, d, d, d);
                    save((std::string)"./out/U_i.npy", h_U_i, d, d, d);
                    save((std::string)"./out/Glc_r.npy", h_Glc_r, sizeNewL, k);
                    save((std::string)"./out/Glc_i.npy", h_Glc_i, sizeNewL, k);
                    save((std::string)"./out/Gcr_r.npy", h_Gcr_r, k, sizeNewR);
                    save((std::string)"./out/Gcr_i.npy", h_Gcr_i, k, sizeNewR);
                    save((std::string)"./out/LL.npy", h_LL, sizeNewL);
                    save((std::string)"./out/LC.npy", h_LC, k);
                    save((std::string)"./out/LR.npy", h_LR, sizeNewR);
                    save((std::string)"./out/CL.npy", h_cNewL, sizeNewL);
                    save((std::string)"./out/CC.npy", h_CC, k);
                    save((std::string)"./out/CR.npy", h_cNewR, sizeNewR);
                    save((std::string)"./out/incC.npy", h_incC, d);
                    save((std::string)"./out/T_r.npy", h_T_r, sizeNewL, sizeNewR);
                    save((std::string)"./out/T_i.npy", h_T_i, sizeNewL, sizeNewR);
                }

                cudaFreeHost(h_CL);
                cudaFreeHost(h_CC);
                cudaFreeHost(h_CR);
                cudaFreeHost(h_idL);
                cudaFreeHost(h_idR);
                cudaFreeHost(h_incL);
                cudaFreeHost(h_incC);
                cudaFreeHost(h_incR);
                cudaFreeHost(h_incNewL);
                cudaFreeHost(h_incNewR);

                cudaFreeHost(h_U_r);
                cudaFreeHost(h_U_i);
                cudaFreeHost(h_Glc_r);
                cudaFreeHost(h_Glc_i);
                cudaFreeHost(h_Gcr_r);
                cudaFreeHost(h_Gcr_i);
                cudaFreeHost(h_LL);
                cudaFreeHost(h_LC);
                cudaFreeHost(h_LR);
                cudaFreeHost(h_cNewL);
                cudaFreeHost(h_CC);
                cudaFreeHost(h_cNewR);
                cudaFreeHost(h_T_r);
                cudaFreeHost(h_T_i);
            }
        }
    }
}