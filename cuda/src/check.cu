#include <cstdio>
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
            int m, int n, int k) {

    float ll, lr, u, glc, gcr, lc;
    int cl, cc, cr;
    for (int i = m-1; i >= max(m-1000,0); --i) {
        cl = CL[i];
        ll = LL[i];

        for (int j = n-1; j >= max(n-1000,0); --j) {
            cr = CR[j];
            lr = LR[j];
            float result = 0.f;

            for (int p = 0; p < k; ++p) {
                cc = CC[p];
                u = U[(cl - tau) * d * d + (tau - cr) * d + cl - cc];
                if (cl >= cc && cl >= tau && tau >= cr) { u = -1; }
                else { u = 0; }
                glc = Glc[i * k + p];
                gcr = Gcr[j + p * n];
                lc = LC[p];
                result = u;// * glc * gcr * lc;
            }

            //result *= ll * lr;

            // if (std::fabs(result - T[i * n + j]) / std::fabs(result) > 1e-5f) {
            //     printf("Size %i %i C[%d][%d] not match, %f vs %f\n", m, n, i, j, result, T[i * n + j]);
            //     //return false;
            // }
            printf("Size %i %i C[%d][%d] match, %f vs %f\n", m, n, i, j, result, T[i * n + j]);
        }
    }
    printf("Size %i %i match\n", m, n);
    return true;
}