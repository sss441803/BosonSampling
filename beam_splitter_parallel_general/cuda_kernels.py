import cupy as cp
import numpy as np

from math import sqrt, ceil, pi
from cmath import exp
from scipy.special import factorial, comb

from mpo_sort import Aligner

# CUDA kernel for updating unitary
u_kernel_file = open('u_kernel.cu')
u_kernel_string = u_kernel_file.read()
u_kernel_file.close()
unitary = cp.RawKernel(u_kernel_string, 'unitary', backend='nvcc')

# Function generating random unitary on CUDA
def Rand_U(d: int, r: float):

    if r == 0:
        # t = np.sqrt(1 - r ** 2) * np.exp(1j * np.random.rand() * 2 * np.pi);
        t = 1
        r = r * np.exp(1j * np.random.rand() * 2 * np.pi);
        ct = np.conj(t); cr = np.conj(r);
        bs_coeff = lambda n, m, k, l: np.sqrt(factorial(l) * factorial(n + m - l) / factorial(n) / factorial(m)) * comb(n, k) * comb(m, l - k) * (t ** k) * (ct ** (m - l + k)) * (r ** (n - k)) * ((-cr) ** (l - k))
        U = np.zeros([d, d, d], dtype = 'complex64');
        for n in range(d): #photon number from 0 to d-1
            for m in range(d):
                for l in range(max(0, n + m + 1 - d), min(d, n + m + 1)): #photon number on first output mode
                    k = np.arange(max(0, l - m), min(l + 1, n + 1, d))
                    U[n, m, l] = np.sum(bs_coeff(n, m, k, l))
        
        return cp.array(np.real(U)), cp.array(np.imag(U))

    U_r = cp.zeros([d, d, d], dtype=np.float32)
    U_i = cp.zeros([d, d, d], dtype=np.float32)
    t = sqrt(1 - r ** 2) * exp(1j * np.random.rand() * 2 * pi)
    t_r = np.float32(np.real(t))
    t_i = np.float32(np.imag(t))
    r = r * exp(1j * np.random.rand() * 2 * pi)
    #print(t, r)
    r_r = np.float32(np.real(r))
    r_i = np.float32(np.imag(r))
    threadsperblock = (8, 8, 8)
    bpg = ceil(d/8)
    blockspergrid = (bpg, bpg, bpg)
    # Memory holder for the Unitaries
    unitary(blockspergrid, threadsperblock, (d, t_r, t_i, r_r, r_i, U_r, U_i))
        
    return U_r, U_i


# Function generating random unitary on CUDA
def Rand_MPO_U(d: int, r: float):
    
    U_r, U_i = Rand_U(d, r)

    MPO_U_r = cp.kron(U_r, U_r) + cp.kron(U_i, U_i)
    MPO_U_i = cp.kron(U_i, U_r) - cp.kron(U_r, U_i)
    
    return MPO_U_r, MPO_U_i


# CUDA kernel for computing the Theta matrix (to be SVDed) for updating the MPS upon application of the unitary
# Load kernel file
update_kernel_file = open('update_kernel.cu')
update_kernel_string = update_kernel_file.read()
update_kernel_file.close()
# Definition of the kernel
update = cp.RawKernel(update_kernel_string, 'kernel', backend='nvcc')

def update_MPS(d, tau, U_r, U_i, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC):

    # U_r = cp.real(U)
    # U_i = cp.imag(U)
    Glc_r = cp.ascontiguousarray(cp.real(Glc), np.float32)
    Glc_i = cp.ascontiguousarray(cp.imag(Glc), np.float32)
    Gcr_r = cp.ascontiguousarray(cp.real(Gcr), np.float32)
    Gcr_i = cp.ascontiguousarray(cp.imag(Gcr), np.float32)
    ll, lc, lr, cl, cc, cr, incc = map(cp.ascontiguousarray, [LL, LC, LR, CL, CC, CR, incC])

    len_l = LL.shape[0]
    len_c = LC.shape[0]
    len_r = LR.shape[0]
    grid = ((len_r + 63) // 64, (len_l + 127) // 128, 1)
    T_r = cp.zeros([len_l, len_r], dtype = np.float32)
    T_i = cp.zeros([len_l, len_r], dtype = np.float32)
    update(grid, (256, 1, 1), (d, tau, U_r, U_i, Glc_r, Glc_i, Gcr_r, Gcr_i, ll, lc, lr, cl, cc, cr, incc, T_r, T_i, len_l, len_r, len_c, int(len_c * 4), int(len_r * 32)))

    return T_r + 1j*T_i


# CUDA kernel for computing the Theta matrix (to be SVDed) for updating the MPO upon application of the unitary
# Load kernel file
update_MPO_kernel_file = open('update_MPO_kernel.cu')
update_MPO_kernel_string = update_MPO_kernel_file.read()
update_MPO_kernel_file.close()
# Definition of the kernel
update_mpo = cp.RawKernel(update_MPO_kernel_string, 'kernel', backend='nvcc')

def update_MPO(d, charge_c_0, charge_c_1, U_r, U_i, glc_obj, gcr_obj, cl_obj, cr_obj, change_charges_C, change_idx_C):

    changes = change_idx_C.shape[0]

    Glc_r = cp.ascontiguousarray(cp.real(glc_obj.data), np.float32)
    Glc_i = cp.ascontiguousarray(cp.imag(glc_obj.data), np.float32)
    Gcr_r = cp.ascontiguousarray(cp.real(gcr_obj.data), np.float32)
    Gcr_i = cp.ascontiguousarray(cp.imag(gcr_obj.data), np.float32)
    try:
        cl0, cl1, cr0, cr1, chcC0, chcC1, idxcC = map(cp.ascontiguousarray, [cl_obj.data[:, 0], cl_obj.data[:, 1], cr_obj.data[:, 0], cr_obj.data[:, 1], change_charges_C[0], change_charges_C[1], change_idx_C])
    except:
        print(cl_obj.data.shape, cr_obj.data.shape, change_charges_C.shape, change_idx_C.shape)
        quit()

    len_l = Glc_r.shape[0]
    len_c = Glc_r.shape[1]
    len_r = Gcr_r.shape[1]
    grid = ((len_r + 63) // 64, (len_l + 127) // 128, 1)
    C_r = cp.zeros([len_l, len_r], dtype = np.float32)
    C_i = cp.zeros([len_l, len_r], dtype = np.float32)

    update_mpo(grid, (256, 1, 1), (d, charge_c_0, charge_c_1, U_r, U_i, Glc_r, Glc_i, Gcr_r, Gcr_i, cl0, cl1, cr0, cr1, changes, chcC0, chcC1, idxcC, C_r, C_i, len_l, len_r, len_c, int(len_c * 4), int(len_r * 32)))

    C = C_r + 1j*C_i
    idx_select = [glc_obj.idx_select[0], gcr_obj.idx_select[1]]
    C_obj = Aligner.make_data_obj('T', True, C, idx_select)

    return C_obj


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