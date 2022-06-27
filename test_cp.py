from cuda_kernels import Rand_U, update_MPS

import cupy as cp
import numpy as np
import scipy.special as s

import time
import itertools


def Rand_BS_MPS_np(d, r):
    t = np.sqrt(1 - r ** 2) * np.exp(1j * np.random.rand() * 2 * np.pi);
    r = r * np.exp(1j * np.random.rand() * 2 * np.pi);
    ct = np.conj(t); cr = np.conj(r);
    bs_coeff = lambda n, m, k, l: np.sqrt(s.factorial(l) * s.factorial(n + m - l) / s.factorial(n) / s.factorial(m)) * s.comb(n, k) * s.comb(m, l - k) * (t ** k) * (ct ** (m - l + k)) * (r ** (n - k)) * ((-cr) ** (l - k))
    BS = np.zeros([d, d, d], dtype = complex);
    for n in range(d): #photon number from 0 to d-1
        for m in range(d):
            for l in range(max(0, n + m + 1 - d), min(d, n + m + 1)): #photon number on first output mode
                k = np.arange(max(0, l - m), min(l + 1, n + 1, d))
                BS[n, m, l] = np.sum(bs_coeff(n, m, k, l))
                if (n*d*d+m*d+l==6397):
                    print(bs_coeff(n, m, k, l))
                    print(np.sum(bs_coeff(n,m,k,l)))
                
    Output = BS;
    
    return Output

def update_MPS_np(tau, d, idx_L, idx_C, idx_R, C, BS, Lambda_l, Gamma_lc, Lambda_c, Gamma_cr, Lambda_r):

    count = 0
    temp_l = 0;
    for ch_l in range(d):
        if len(idx_L[ch_l]) == 0:
            continue
        temp_l += len(idx_L[ch_l]);
        temp_r = 0;
        for ch_r in range(d):
            if len(idx_R[ch_r]) == 0:
                continue
            temp_r += len(idx_R[ch_r]);
            for ch_c in range(ch_r, ch_l + 1):
                added = BS[ch_l - tau, tau - ch_r, ch_l - ch_c] * np.multiply(np.matmul(np.multiply(np.multiply(Lambda_l[idx_L[ch_l]].reshape(-1, 1), Gamma_lc[idx_L[ch_l].reshape(-1, 1), idx_C[ch_c].reshape(1, -1)]), Lambda_c[idx_C[ch_c]].reshape(1, -1)), Gamma_cr[idx_C[ch_c].reshape(-1, 1), idx_R[ch_r].reshape(1, -1)]), Lambda_r[idx_R[ch_r]].reshape(1, -1))
                #added = 1
                #if np.isnan(BS[ch_l - tau, tau - ch_r, ch_l - ch_c].reshape(-1)).shape[0] != 0:
                #    print(np.isnan(BS[ch_l - tau, tau - ch_r, ch_l - ch_c].reshape(-1)).shape[0], ch_l, ch_r, ch_c)
                #if np.isnan(added.reshape(-1)).shape[0] != 0:
                #    print(np.isnan(added.reshape(-1)).shape[0], ch_l, ch_r, ch_c)
                C[temp_l - len(idx_L[ch_l]):temp_l, temp_r - len(idx_R[ch_r]):temp_r] += added
                if BS[ch_l - tau, tau - ch_r, ch_l - ch_c] != 0:
                    #print(added.shape, len(idx_L[ch_l]), len(idx_R[ch_r]))
                    #print(C[temp_l - len(idx_L[ch_l]):temp_l, temp_r - len(idx_R[ch_r]):temp_r])
                    #print(np.nonzero(added.reshape(-1))[0], added.reshape(-1),idx_L[ch_l],idx_C[ch_c],idx_R[ch_r])
                    #print(added)
                    count += 1
    
    #print("count is ", count)
    return C

if __name__ == "__main__":

    test_unitary = False
    test_update = True

    if test_unitary:

        d=50
        iterations = 1

        seed = 1#np.random.randint(0, 100000)
        np.random.seed(seed)
        cp_result = cp.zeros(d**3, dtype=complex)
        Rand_U(d, 0.2, cp_result)
        print("GPU nan elements: ", np.argwhere(np.isnan(cp_result.reshape(-1))).shape[0])
        np.random.seed(seed)
        np_result = Rand_BS_MPS_np(d, 0.2)
        print("CPU non elements: ", np.argwhere(np.isnan(np_result.reshape(-1))).shape[0])
        start = np.random.randint(0, d**3-10)
        print("Results agree? ", np.allclose(cp.asnumpy(cp_result).reshape(-1), np_result.reshape(-1), atol=0.0001))
        print("Maximum error: ", np.abs((cp.asnumpy(cp_result) - np_result.reshape(-1))).max())
        start = time.time()
        for i in range(iterations):
            Rand_U(d, 0.2, cp_result)
        print(cp_result[0])
        print("GPU time: ", time.time()-start)
        start = time.time()
        for i in range(iterations):
            a=Rand_BS_MPS_np(d, 0.2)
        print(a[0,0,0])
        print("CPU time: ", time.time()-start)

    if test_update:

        errtol = 0.7

        d = 20
        tau = 10
        chi = 2000

        bc_l = np.random.randint(tau, d, chi)
        bc_c = np.random.randint(0, d, chi)
        bc_r = np.random.randint(0, tau+1, chi)
        Lambda_l = np.random.rand(chi)
        Lambda_c = np.random.rand(chi)
        Lambda_r = np.random.rand(chi)
        
        d_bc_l = cp.array(bc_l, dtype=int)
        d_bc_c = cp.array(bc_c, dtype=int)
        d_bc_r = cp.array(bc_r, dtype=int)
        d_Lambda_l = cp.array(Lambda_l)
        d_Lambda_c = cp.array(Lambda_c)
        d_Lambda_r = cp.array(Lambda_r)

        d_Gamma_lc = cp.random.rand(chi**2) + 1j*cp.random.rand(chi**2)
        d_Gamma_cr = cp.random.rand(chi**2) + 1j*cp.random.rand(chi**2)
        d_BS = cp.zeros(d**3, dtype=complex)
        Rand_U(d, 0.2, d_BS)
        #print(d_BS)
        Gamma_lc = cp.asnumpy(d_Gamma_lc).reshape([chi, chi])
        Gamma_cr = cp.asnumpy(d_Gamma_cr).reshape([chi, chi])
        BS = cp.asnumpy(d_BS).reshape(d, d, d)

        print(np.argwhere(np.isnan(BS.reshape(-1))).shape[0])

        idx_L = []; idx_R = []; idx_C = [];

        #print(bc_l)
        for j in range(d):
            idx_L.append(np.array(list(set.intersection(set(np.nonzero(bc_l == j)[0]), set(np.nonzero(Lambda_l > errtol)[0]))), dtype = 'int32')) #left hand side charge
            idx_C.append(np.array(list(set.intersection(set(np.nonzero(bc_c == j)[0]), set(np.nonzero(Lambda_c > errtol)[0]))), dtype = 'int32'))
            idx_R.append(np.array(list(set.intersection(set(np.nonzero(bc_r == j)[0]), set(np.nonzero(Lambda_r > errtol)[0]))), dtype = 'int32'))
        #print(idx_L)

        l_bond = np.array(list(itertools.chain(*[idx_L[i] for i in range(d)])))
        c_bond = np.array(list(itertools.chain(*[idx_C[i] for i in range(d)])))
        r_bond = np.array(list(itertools.chain(*[idx_R[i] for i in range(d)])))
        d_l_bond = cp.array(l_bond, dtype=int)
        d_c_bond = cp.array(c_bond, dtype=int)
        d_r_bond = cp.array(r_bond, dtype=int)

        #print(bc_l, Lambda_l, idx_L, l_bond)
        #print(bc_c, Lambda_c, idx_C, c_bond)
        #print(bc_r, Lambda_r, idx_R, r_bond)

        len_l = l_bond.shape[0]
        len_r = r_bond.shape[0]
        print(len_l, len_r)

        C = cp.zeros(len_l*len_r, dtype=complex)
        update_MPS(tau, d, chi, d_l_bond, d_c_bond, d_r_bond, d_bc_l, d_bc_c, d_bc_r, C, d_BS, d_Lambda_l, d_Gamma_lc, d_Lambda_c, d_Gamma_cr, d_Lambda_r)
        C_np = np.zeros([len_l,len_r], dtype=complex)
        update_MPS_np(tau, d, idx_L, idx_C, idx_R, C_np, BS, Lambda_l, Gamma_lc, Lambda_c, Gamma_cr, Lambda_r)
        print("Results agree? ", np.allclose(cp.asnumpy(C).reshape(-1), C_np.reshape(-1)))

        start = time.time()
        for i in range(1):
            update_MPS(tau, d, chi, d_l_bond, d_c_bond, d_r_bond, d_bc_l, d_bc_c, d_bc_r, C, d_BS, d_Lambda_l, d_Gamma_lc, d_Lambda_c, d_Gamma_cr, d_Lambda_r)
            print(cp.asnumpy(C[-1]), np.argwhere(np.isnan(cp.asnumpy(C.reshape(-1)))).shape[0])
        print("GPU time: ", time.time()-start)
        
        start = time.time()
        for i in range(1):
            update_MPS_np(tau, d, idx_L, idx_C, idx_R, C_np, BS, Lambda_l, Gamma_lc, Lambda_c, Gamma_cr, Lambda_r)
            print(C_np[-1, -1], np.argwhere(np.isnan(C_np.reshape(-1))).shape[0])
        print("CPU time: ", time.time()-start)

        #print(cp.asnumpy(C).reshape(-1))
        #print(np.nonzero(np.abs(cp.asnumpy(C)).reshape(-1))[0].shape[0])
        #print('Numpy result:')
        #print(np.nonzero(np.abs(C_np).reshape(-1))[0].shape[0])
        #print(C_np)
        #print(np.allclose(cp.asnumpy(C).reshape(-1), C_np.reshape(-1)))