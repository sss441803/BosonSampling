import numpy as np
import cupy as cp
import time


data_type = np.complex64
float_type = np.float32


def numpy(d, tau, U, CL, CC, CR, LL, Glc, LC, Gcr, LR):

    len_l, len_r = CL.shape[0], CR.shape[0]
    T = np.zeros([len_l, len_r], dtype = data_type)
    idx_L = []; idx_R = []; idx_C = []

    for j in range(d):
        idx_L.append(np.array(np.nonzero(CL == j)[0], dtype = int)) #left hand side charge
        idx_C.append(np.array(np.nonzero(CC == j)[0], dtype = int))
        idx_R.append(np.array(np.nonzero(CR == j)[0], dtype = int))

    l_charge = np.arange(tau, d, dtype=int)
    c_charge = np.arange(d, dtype=int)
    r_charge = np.arange(tau + 1, dtype=int)
    idx_l = [idx_L[i] if i in l_charge else np.array([]) for i in range(d)]
    idx_c = [idx_C[i] if i in c_charge else np.array([]) for i in range(d)]
    idx_r = [idx_R[i] if i in r_charge else np.array([]) for i in range(d)]

    start = time.time()

    # print(idx_l[30])
    count = 0
    temp_l = 0

    for ch_l in range(d):

        if len(idx_l[ch_l]) == 0:
            continue

        temp_l += len(idx_l[ch_l])
        temp_r = 0

        for ch_r in range(d):

            if len(idx_r[ch_r]) == 0:
                continue

            if len(idx_r[ch_r]) == 0:
                continuer_charge = np.arange(tau + 1, dtype=int)

            temp_r += len(idx_r[ch_r])

            for ch_c in range(ch_r, ch_l + 1):

                if len(idx_c[ch_c]) == 0:
                    continue

                added = U[ch_l - tau, tau - ch_r, ch_l - ch_c] * np.multiply(np.matmul(np.multiply(np.multiply(LL[idx_l[ch_l]].reshape(-1, 1), Glc[idx_l[ch_l].reshape(-1, 1), idx_c[ch_c].reshape(1, -1)]), LC[idx_c[ch_c]].reshape(1, -1)), Gcr[idx_c[ch_c].reshape(-1, 1), idx_r[ch_r].reshape(1, -1)]), LR[idx_r[ch_r]].reshape(1, -1))

                T[temp_l - len(idx_l[ch_l]):temp_l, temp_r - len(idx_r[ch_r]):temp_r] += added
    
    stop = time.time()
    print('Numpy time: ', (stop - start) * 1000)

    return T


# Checking if GPU computed results agree with the CPU results
def loop(d, tau, U, CL, CC, CR, LL, Glc, LC, Gcr, LR):

    m, n, k = CL.shape[0], CR.shape[0], CC.shape[0]
    T = np.zeros([m, n], dtype = data_type)

    for i in range(m):
        print(m)
        cl = CL[i]
        ll = LL[i]

        for j in range(n):
            cr = CR[j]
            lr = LR[j]
            result = 0

            for p in range(k):
                cc = CC[p]
                u = U[cl - tau, tau - cr, cl - cc]
                if (cl >= cc) and (cc >= cr):
                    u = u
                else:
                    u = 0
                glc = Glc[i, p]
                gcr = Gcr[p, j]
                lc = LC[p]
                result += u * glc * gcr * lc

            result *= ll * lr
            T[i, j] = result

    return T


# CUDA kernel for computing the Theta matrix (to be SVDed) for updating the MPS upon application of the unitary
# Load kernel file
kernel_file = open('../../kernel_file.cu')
kernel_string = kernel_file.read()
kernel_file.close()
# Definition of the kernel
update = cp.RawKernel(kernel_string, 'kernel', backend='nvcc')

def update_MPS(d, tau, U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC):
    # idx_c = cp.argsort(CC)
    # CC = cp.take(CC, idx_c, axis = 0)
    # Glc = cp.take(Glc, idx_c, axis = 1)
    # LC = cp.take(LC, idx_c, axis = 0)
    # Gcr = cp.take(Gcr, idx_c, axis = 0)
    # U = U.reshape(-1)
    # Glc = Glc.reshape(-1)
    # Gcr = Gcr.reshape(-1)
    len_l = LL.shape[0]
    len_c = LC.shape[0]
    len_r = LR.shape[0]
    grid = ((len_r + 63) // 64, (len_l + 127) // 128, 1)
    T = cp.zeros(len_l * len_r, dtype = np.float32)
    update(grid, (256, 1, 1), (d, tau, U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC, T, len_l, len_r, len_c, int(len_c * 4), int(len_r * 32)))
    return T.reshape(len_l, len_r)


T = np.load('../out/T_r.npy') + 1j*np.load('../out/T_i.npy')
incC = np.load('../out/incC.npy')
U = np.load('../out/U_r.npy') + 1j*np.load('../out/U_i.npy')
CL = np.load('../out/CL.npy')
CC = np.load('../out/CC.npy')
CR = np.load('../out/CR.npy')
Glc = np.load('../out/Glc_r.npy') + 1j*np.load('../out/Glc_i.npy')
Gcr = np.load('../out/Gcr_r.npy') + 1j*np.load('../out/Gcr_i.npy')
LL = np.load('../out/LL.npy')
LC = np.load('../out/LC.npy')
LR = np.load('../out/LR.npy')

T = np.array(T, dtype = data_type)
incC = np.array(incC, dtype = np.int32)
U = np.array(U, dtype = data_type)
CL = np.array(CL, dtype = np.int32)
CC = np.array(CC, dtype = np.int32)
CR = np.array(CR, dtype = np.int32)
Glc = np.array(Glc, dtype = data_type)
Gcr = np.array(Gcr, dtype = data_type)
LL = np.array(LL, dtype = float_type)
LC = np.array(LC, dtype = float_type)
LR = np.array(LR, dtype = float_type)

d = incC.shape[0]
tau = d // 2

start = time.time()
numpy_T = numpy(d, tau, U, CL, CC, CR, LL, Glc, LC, Gcr, LR)
print(numpy_T)
print('numpy: ', time.time() - start)
print('all close? ', np.allclose(numpy_T, T))
# U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC = map(cp.array, [U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC])
# U = cp.random.rand(2, 2, 2)
# d = 2
# tau = 0
# Glc = cp.array([[1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
# Gcr = cp.array([[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
# LL = cp.array([1,1,0,0,0,0,0,0])
# LC = cp.array([1,0])
# LR =  cp.array([1,0,0,0,0,0,0,0])
# CL = cp.array([1,1,1,1,1,1,1,1])
# CC = cp.array([0,0])
# CR = cp.array([0,0,0,0,0,0,0,0])
# incC = cp.array([0,-1])

# cupy_T = update_MPS(d, tau, U, Glc, Gcr, LL, LC, LR, CL, CC, CR, incC)
# U, CL, CC, CR, LL, Glc, LC, Gcr, LR = map(cp.asnumpy, [U, CL, CC, CR, LL, Glc, LC, Gcr, LR])
# numpy_T = numpy(d, tau, U, CL, CC, CR, LL, Glc, LC, Gcr, LR)

# start = time.time()
# loop_T = loop(d, tau, U, CL, CC, CR, LL, Glc, LC, Gcr, LR)
# stop = time.time()
# print('Loop time: ', (stop - start) * 1000)

# start = time.time()
# cpu_T = np.multiply(LL.reshape(-1,1),Glc)
# cpu_T = np.multiply(LC, cpu_T)
# cpu_T = np.matmul(cpu_T, Gcr)
# cpu_T = np.multiply(LR, cpu_T)
# stop = time.time()
# print('CPU time: ', (stop - start) * 1000)

# Glc = cp.array(Glc, dtype = data_type)
# Gcr = cp.array(Gcr, dtype = data_type)
# LL = cp.array(LL, dtype = data_type)
# LC = cp.array(LC, dtype = data_type)
# LR = cp.array(LR, dtype = data_type)

# start = time.time()
# gpu_T = cp.multiply(LL.reshape(-1,1),Glc)
# gpu_T = cp.multiply(LC, gpu_T)
# gpu_T = cp.matmul(gpu_T, Gcr)
# gpu_T = cp.multiply(LR, gpu_T)
# print(gpu_T[0][0].item())
# stop = time.time()
# print('Cupy time: ', (stop - start) * 1000)

# start = time.time()
# gpu_T = cp.einsum('a,ab,b,bc,c->ac', LL, Glc, LC, Gcr, LR)
# print(gpu_T[0][0].item())
# stop = time.time()
# print('Cupy einsum time: ', (stop - start) * 1000)

# print('GPU results: ', cupy_T)
# print('Numpy results: ', numpy_T)
# # print('Loop results: ', loop_T)
# print(np.allclose(T, numpy_T, rtol=1e-05, atol=1e-08, equal_nan=False))
# print(np.allclose(cupy_T, numpy_T, rtol=1e-05, atol=1e-08, equal_nan=False))
# print(np.max(np.abs(cp.asnumpy(cupy_T)- numpy_T)))