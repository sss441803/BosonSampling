#include <cstdio>
#include <cstdint>

// Checking if GPU computed results agree with the CPU results
bool check( const float *U,
            const float *A,
            const float *B,
            const float *LL,
            const float *LC,
            const float *LR,
            const float *C,
            int m, int n, int k) {

    for (int i = m-1; i >= max(m-100,0); --i) {
        for (int j = n-1; j >= max(n-100,0); --j) {
            float result = 0.f;
            for (int p = 0; p < k; ++p) {
                result += A[i * k + p] * B[j + p * n] * LC[p];
            }
            result *= LL[i] * LR[j];

            if (std::fabs(result - C[i * n + j]) / std::fabs(result) > 1e-5f) {
                printf("Size %i %i C[%d][%d] not match, %f vs %f\n", m, n, i, j, result, C[i * n + j]);
                return false;
            }
        }
    }
    printf("Size %i %i match\n", m, n);
    return true;
}