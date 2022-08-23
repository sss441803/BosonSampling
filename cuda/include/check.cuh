#include <cstdint>

// Checking if GPU computed results agree with the CPU results
bool check( const float *T_r,
            const float *T_i,
            const int d,
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
            int m, int n, int k);