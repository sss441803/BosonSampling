#include <cstdint>

// CUDA kernel
__global__ __launch_bounds__(256, 2)
void kernel(const int d,
            const int tau,
            const float *U_r,
            const float *U_i,
            const float *Glc_r,
            const float *Glc_i,
            const float *Gcr_r,
            const float *Gcr_i,
            const float *LL,
            const float *LC,
            const float *LR,
            const int *CL,
            const int *CC,
            const int *CR,
            const int *incC,
            float *T_r,
            float *T_i,
            uint32_t m,
            uint32_t n,
            uint32_t k,
            uint32_t Glc_ldg_step,    // k * sizeof(float)
            uint32_t Gcr_ldg_step);