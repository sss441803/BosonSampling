import cupy as cp
import numpy as np

from math import sqrt, ceil, pi
from cmath import exp


# CUDA kernel for updating unitary
unitary = cp.RawKernel(r'''

#include <cuComplex.h>

// Commonly used complex number operations
typedef double     rtype;
typedef cuDoubleComplex ctype;
#define rpart(x)   (cuCreal(x))
#define ipart(x)   (cuCimag(x))
#define cmplx(x,y) (make_cuDoubleComplex(x,y))
#define conj(z)    (cuConj(z))
#define cmul(x, y) (cuCmul(x,y))
#define cadd(x, y) (cuCadd(x,y))
#define pi          3.14159265358979323846

// Defining pow for complex numbers
__host__ __device__ rtype carg(const ctype& z) {return (rtype)atan2(ipart(z), rpart(z));} // polar angle
__host__ __device__ rtype cabs(const ctype& z) {return (rtype)cuCabs(z);} // magnitude (absolute value)
__host__ __device__ ctype cpow(const ctype& z, const int &n) {return cmplx((pow(cabs(z), n)*cos(n*carg(z))), (pow(cabs(z), n)*sin(n*carg(z))));} // power of complex numbers

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

__device__
double logcabs(ctype x){return log(cabs(x));}

__device__
ctype bs_coeff(int n, int m, int l, int k, ctype t, ctype r)
{
    if (n<k||m<l-k){return 0;}
    ctype ct = conj(t);
    ctype cr = conj(r);
    //ctype answer = cmul(cmplx(sqrt(factorial(l) * factorial(n + m - l) / factorial(n) / factorial(m)) * comb(n, k) * comb(m, l - k), 0), cmul(cmul(cpow(t, k), cpow(ct, m - l + k)), cmul(cpow(r ,n - k), cpow(cmul(cmplx(-1, 0), cr), l - k))));
    double loganswerabs = 0.5*(logfactorial(l) + logfactorial(n + m - l) - logfactorial(n) - logfactorial(m)) + logcomb(n, k) + logcomb(m, l - k) + logcabs(t)*k + logcabs(ct)*(m - l + k) + logcabs(r)*(n - k) + logcabs(cr)*(l - k);
    double loganswerarg = carg(t)*k + carg(ct)*(m-l+k) + carg(r)*(n-k) + (carg(cr)-pi)*(l-k);
    ctype answer = cmplx(exp(loganswerabs)*cos(loganswerarg), exp(loganswerabs)*sin(loganswerarg));
    
    return answer;
}

extern "C" __global__ void unitary(const int d, const ctype t, const ctype r, ctype* BS)
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
            for (int k = k_low; k < k_high; k += 1)
            {
                BS[n*d*d + m*d + l] = cadd(BS[n*d*d + m*d + l], bs_coeff(n, m, l, k, t, r));
            }
        }
    }
}''',
'unitary', backend='nvcc', translate_cucomplex=True)

# Function generating random unitary on CUDA
def Rand_U(d: int, r: float, BS):
    t = sqrt(1 - r ** 2) * exp(1j * np.random.rand() * 2 * pi)
    r = r * exp(1j * np.random.rand() * 2 * pi)
    threadsperblock = (8, 8, 8)
    bpg = ceil(d/8)
    blockspergrid = (bpg, bpg, bpg)
    # Memory holder for the Unitaries
    unitary(blockspergrid, threadsperblock, (d, t, r, BS))


# CUDA kernel for computing the Theta matrix (to be SVDed) for updating the MPS upon application of the unitary
update = cp.RawKernel(r'''
#include <cuComplex.h>

typedef cuDoubleComplex ctype;

#define cmplx(x,y) (make_cuDoubleComplex(x,y))
#define rpart(x)   (cuCreal(x))
#define ipart(x)   (cuCimag(x))
#define cmul(x, y) (cuCmul(x,y))

__device__ void atomicAddComplex(cuDoubleComplex* a, cuDoubleComplex b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAdd(x, cuCreal(b));
  atomicAdd(y, cuCimag(b));
}


extern "C" __global__
void update(const int tau, const int d, const int chi, int* l_bond, int* c_bond, int* r_bond, int len_l, int len_c, int len_r, int* bc_l, int* bc_c, int* bc_r, ctype* C, ctype* BS, double* Lambda_l, ctype* Gamma_lc, double* Lambda_c, ctype* Gamma_cr, double* Lambda_r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x<len_l && y<len_c && z<len_r)
    {   
        // For some reason the array initialized in cupy passed to the CUDA kernel with int type must be indexed with factor of 2
        int id_l = l_bond[x*2]; int id_c = c_bond[y*2]; int id_r = r_bond[z*2];
        int ch_l = bc_l[id_l*2]; int ch_c = bc_c[id_c*2]; int ch_r = bc_r[id_r*2];

        if (ch_c >= ch_r && ch_c <= ch_l)
        {
            ctype coeff = cmul(cmul(BS[(ch_l-tau)*d*d + (tau-ch_r)*d + (ch_l-ch_c)], cmul(Gamma_lc[id_l*chi + id_c], Gamma_cr[id_c*chi + id_r])), cmplx(Lambda_l[id_l]*Lambda_c[id_c]*Lambda_r[id_r], 0));
            atomicAddComplex(&C[x*len_r + z], coeff);
            //atomicAdd(&C[(x*len_r + z)*sizeof(int)], z);
            //C[(x*len_r + z)]=1.5;
        }
    }
}
''',
'update', backend='nvcc', translate_cucomplex=True)

def update_MPS(tau, d, chi, l_bond, c_bond, r_bond, bc_l, bc_c, bc_r, C, BS, Lambda_l, Gamma_lc, Lambda_c, Gamma_cr, Lambda_r):
    
    len_l = l_bond.shape[0]
    len_c = c_bond.shape[0]
    len_r = r_bond.shape[0]
    
    threadsperblock = (8, 8, 8)
    bpgx = ceil(len_l/8)
    bpgy = ceil(len_c/8)
    bpgz = ceil(len_r/8)
    blockspergrid = (bpgx, bpgy, bpgz)
    #print(blockspergrid)
    
    update(blockspergrid, threadsperblock, (tau, d, chi, l_bond, c_bond, r_bond, len_l, len_c, len_r, bc_l, bc_c, bc_r, C, BS, Lambda_l, Gamma_lc, Lambda_c, Gamma_cr, Lambda_r))

    #return C