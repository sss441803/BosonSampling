#include <cstdio>
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
            int m, int n, int k) {

    float ll, lc, lr, u_r, u_i, glc_r, glc_i, gcr_r, gcr_i;
    int cl, cc, cr;
    for (int i = m-1; i >= max(m-1000,0); i -= (m+9999)/1000) {
        cl = CL[i];
        ll = LL[i];

        for (int j = n-1; j >= max(n-1000,0); j -= (n+999)/1000) {
            cr = CR[j];
            lr = LR[j];
            float result_r = 0.f;
            float result_i = 0.f;
            float temp_r = 0.f;
            float temp_i = 0.f;
            float temp_r_2 = 0.f;
            float temp_i_2 = 0.f;

            for (int p = 0; p < k; p += 1) {
                cc = CC[p];
                u_r = U_r[(cl - tau) * d * d + (tau - cr) * d + cl - cc];
                u_i = U_i[(cl - tau) * d * d + (tau - cr) * d + cl - cc];
                if (cl < cc || cc < cr) { u_r = u_i = 0; }
                glc_r = Glc_r[i * k + p];
                glc_i = Glc_i[i * k + p];
                gcr_r = Gcr_r[j + p * n];
                gcr_i = Gcr_i[j + p * n];
                lc = LC[p];
                
                temp_r = glc_r * gcr_r - glc_i * gcr_i;
                temp_i = glc_r * gcr_i + glc_i * gcr_r;
                temp_r_2 = u_r * temp_r - u_i * temp_i;
                temp_i_2 = u_r * temp_i + u_i * temp_r;
                result_r += temp_r_2 * lc;//lc * (u_r * (glc_r * gcr_r - glc_i * gcr_i) - u_i * (glc_r * gcr_i + glc_i * gcr_r));
                result_i += temp_i_2 * lc;
            }

            result_r *= ll * lr;
            result_i *= ll * lr;

            if ( (std::fabs(result_r - T_r[i * n + j]) / std::fabs(result_r) > 1e-3f) || (std::fabs(result_i - T_i[i * n + j]) / std::fabs(result_i) > 1e-3f) ) {
                printf("C[%d][%d] does NOT match, %f + %f j vs %f + %f j\n", i, j, result_r, result_i, T_r[i * n + j], T_i[i * n + j]);
                return false;
            }
            //printf("Size %i %i C[%d][%d] match, %f vs %f\n", m, n, i, j, result, T[i * n + j]);
        }
    }
    //printf("Size %i %i match\n", m, n);
    return true;
}