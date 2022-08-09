#include <cstdint>

// CUDA kernel
__global__ __launch_bounds__(256, 2)
void kernel(const int d,
            const int tau,
            const float *U,
            const float *Glc,
            const float *Gcr,
            const float *LL,
            const float *LC,
            const float *LR,
            const int *CL,
            const int *CC,
            const int *CR,
            const int *incC,
            float *T,
            uint32_t m,
            uint32_t n,
            uint32_t k,
            uint32_t Glc_ldg_step,    // k * sizeof(float)
            uint32_t Gcr_ldg_step);