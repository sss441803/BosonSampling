#define pi          3.14159265358979323846

__device__
float factorial(const int n)
{
    float f = 1.0;
    for (int i=1; i<=n; ++i)
        f *= i;
    return f;
}

__device__
int comb(const int n, const int r)
{   
    if (n>= r)
    {
        return factorial(n) / (factorial(r) * factorial(n-r));
    }
    else {return 0;}
}

__device__
double logfactorial(const int n){return lgamma(n+1.0);}


__device__
double logcomb(const int n, const int r)
{   
    if (n >= r)
    {
        return logfactorial(n) - logfactorial(r) - logfactorial(n-r);
    }
    else {return 0;}
}


extern "C" __global__ void unitary(const int d, const float t_r, const float t_i, const float r_r, const float r_i, float* U_r, float* U_i)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;

    if (n<d && m<d && l<d)
    {   
        int l_low = max(0, n + m + 1 - d);
        int l_high = min(d, n + m + 1);

        if (l >= l_low && l < l_high)
        {
            int k_low = max(0, l - m);
            int k_high = min(min(l + 1, n + 1), d);

            double coeff_r, coeff_i, t_abs, t_arg, r_abs, r_arg, loganswerabs, loganswerarg;
            for (int k = k_low; k < k_high; k += 1)
            {   
                if (n<k||m<l-k){ float coeff_r = 0; float coeff_i = 0;  }
                else {
                    t_abs = sqrt(t_r * t_r + t_i * t_i);
                    t_arg = atan2(t_i, t_r);
                    r_abs = sqrt(r_r * r_r + r_i * r_i);
                    r_arg = atan2(r_i, r_r);
                    loganswerabs = 0.5*(logfactorial(l) + logfactorial(n + m - l) - logfactorial(n) - logfactorial(m)) + logcomb(n, k) + logcomb(m, l - k) + log(t_abs)*k + log(t_abs)*(m - l + k) + log(r_abs)*(n - k) + log(r_abs)*(l - k);
                    loganswerarg = t_arg*k - t_arg*(m-l+k) + r_arg*(n-k) - (r_arg + pi)*(l-k);
                    coeff_r = exp(loganswerabs)*cos(loganswerarg);
                    coeff_i = exp(loganswerabs)*sin(loganswerarg);
                }
                U_r[n*d*d + m*d + l] += coeff_r;
                U_i[n*d*d + m*d + l] += coeff_i;
            }
        }
    }
}