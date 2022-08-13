#include <cstdint>

// Checking if GPU computed results agree with the CPU results
bool check( const float *T,
            const int d,
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
            int m, int n, int k);