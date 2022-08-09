#include <cstdint>

// Checking if GPU computed results agree with the CPU results
bool check( const float *U,
            const float *A, const float *B,
            const float *LL, const float *LC, const float *LR,
            const float *C,
            int m, int n, int k);