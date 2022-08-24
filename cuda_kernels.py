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
def Rand_U(d: int, r: float):
    U = cp.zeros(d**3, dtype=complex)
    t = sqrt(1 - r ** 2) * exp(1j * np.random.rand() * 2 * pi)
    r = r * exp(1j * np.random.rand() * 2 * pi)
    threadsperblock = (8, 8, 8)
    bpg = ceil(d/8)
    blockspergrid = (bpg, bpg, bpg)
    # Memory holder for the Unitaries
    unitary(blockspergrid, threadsperblock, (d, t, r, U))
    
    return U.reshape(d, d, d)



# CUDA kernel for computing the Theta matrix (to be SVDed) for updating the MPS upon application of the unitary
# Load kernel file
kernel_file = open('kernel_file.cu')
kernel_string = kernel_file.read()
kernel_file.close()
# Definition of the kernel
update = cp.RawKernel(kernel_string, 'kernel', backend='nvcc')

def update_MPS(d, tau, U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC):

    U_r = cp.array(cp.real(U), np.float32)
    U_i = cp.array(cp.imag(U), np.float32)
    Glc_r = cp.array(cp.real(Glc), np.float32)
    Glc_i = cp.array(cp.imag(Glc), np.float32)
    Gcr_r = cp.array(cp.real(Gcr), np.float32)
    Gcr_i = cp.array(cp.imag(Gcr), np.float32)

    len_l = LL.shape[0]
    len_c = LC.shape[0]
    len_r = LR.shape[0]
    grid = ((len_r + 63) // 64, (len_l + 127) // 128, 1)
    T_r = cp.zeros(len_l * len_r, dtype = np.float32)
    T_i = cp.zeros(len_l * len_r, dtype = np.float32)
    update(grid, (256, 1, 1), (d, tau, U_r, U_i, Glc_r, Glc_i, Gcr_r, Gcr_i, LL, LC, LR, CL, CC, CR, incC, T_r, T_i, len_l, len_r, len_c, int(len_c * 4), int(len_r * 32)))

    return (T_r + 1j*T_i).reshape(len_l, len_r)


data_type = np.float32

if __name__ == '__main__':

    import time

    T = np.load('cuda/out/T.npy')
    incC = np.load('cuda/out/incC.npy')
    U = np.load('cuda/out/U.npy')
    CL = np.load('cuda/out/CL.npy')
    CC = np.load('cuda/out/CC.npy')
    CR = np.load('cuda/out/CR.npy')
    Glc = np.load('cuda/out/Glc.npy')
    Gcr = np.load('cuda/out/Gcr.npy')
    LL = np.load('cuda/out/LL.npy')
    LC = np.load('cuda/out/LC.npy')
    LR = np.load('cuda/out/LR.npy')

    T = cp.array(T, dtype = data_type)
    incC = cp.array(incC, dtype = np.int32)
    U = cp.array(U, dtype = data_type)
    CL = cp.array(CL, dtype = np.int32)
    CC = cp.array(CC, dtype = np.int32)
    CR = cp.array(CR, dtype = np.int32)
    Glc = cp.array(Glc, dtype = data_type)
    Gcr = cp.array(Gcr, dtype = data_type)
    LL = cp.array(LL, dtype = data_type)
    LC = cp.array(LC, dtype = data_type)
    LR = cp.array(LR, dtype = data_type)

    d = incC.shape[0]
    tau = d // 2

    print("d: {}, tau: {}, T shape: {}.".format(d, tau, T.shape))

    cpT = update_MPS(d, tau, U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC)
    print(cpT[0, 0])
    start = time.time()
    cpT = update_MPS(d, tau, U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC)
    print(cpT[0, 0])
    print("Time per kernel: ", time.time() - start)

    print("Results agree? ", cp.allclose(cpT, T))
    














r'''
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
'''