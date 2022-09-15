'''Full simulation code containing the Device method (cupy, unified update)'''
import numpy as np
import cupy as cp

from scipy.stats import rv_continuous
from scipy.special import factorial, comb

import time
from itertools import combinations
from multiprocessing import Pool

from mpo_sort import Aligner

np.random.seed(1)
np.set_printoptions(precision=3)

data_type = np.complex64
float_type = np.float32
int_type = np.int32

def update_MPO(d, charge_c_0, charge_c_1, U, glc_obj, gcr_obj, cl_obj, cc_obj, cr_obj, change_charges_C, change_idx_C):

    charge_id_max = change_idx_C.shape[0]

    charge_c = charge_c_0 * d + charge_c_1

    glc, gcr, cl, cc, cr = glc_obj.data, gcr_obj.data, cl_obj.data, cc_obj.data, cr_obj.data

    M, K, N = glc.shape[0], glc.shape[1], gcr.shape[1]

    T = np.zeros([M, N], dtype="complex64")

    for m in range(M):
        charge_l_0, charge_l_1 = cl[m]
        charge_l = charge_l_0 * d + charge_l_1
        for n in range(N):
            charge_r_0, charge_r_1 = cr[n]
            charge_r = charge_r_0 * d + charge_r_1
            charge_id = 0
            next_k = 0
            for k in range(K):
                if k == next_k:
                    charge_local_0 = change_charges_C[0][charge_id]
                    charge_local_1 = change_charges_C[1][charge_id]
                    charge_local = charge_local_0 * d + charge_local_1
                    charge_id += 1
                    if charge_id < charge_id_max:
                        next_k = change_idx_C[charge_id]
                # charge_local_0, charge_local_1 = cc[k]
                if charge_l_0 >= charge_c_0 and charge_l_1 >= charge_c_1 and charge_r_0 <= charge_c_0 and charge_r_1 <= charge_c_1 and charge_l_0 >= charge_local_0 and charge_l_1 >= charge_local_1 and charge_r_0 <= charge_local_0 and charge_r_1 <= charge_local_1:
                    u = U[charge_l - charge_c, charge_c - charge_r, charge_l - charge_local, charge_local - charge_r]
                    add = u * glc[m, k] * gcr[k, n] #  * lc[k]
                    T[m, n] += add
            # T[m, n] *= ll[m] * lr[n]

    idx_select = [glc_obj.idx_select[0], gcr_obj.idx_select[1]]
    T_obj = Aligner.make_data_obj('T', True, T, idx_select)
    
    return T_obj

def Rand_U(d, r):
    #t = 1 / np.sqrt(2); r = 1 / np.sqrt(2);
    t = np.sqrt(1 - r ** 2) * np.exp(1j * np.random.rand() * 2 * np.pi)
    r = r * np.exp(1j * np.random.rand() * 2 * np.pi)
    ct = np.conj(t); cr = np.conj(r)
    bs_coeff = lambda n, m, k, l: np.sqrt(factorial(l) * factorial(n + m - l) / factorial(n) / factorial(m)) * comb(n, k) * comb(m, l - k) * (t ** k) * (ct ** (m - l + k)) * (r ** (n - k)) * ((-cr) ** (l - k))
    U = np.zeros([d, d, d, d], dtype = 'complex64')
    for n in range(d): #photon number from 0 to d-1
        for m in range(d):
            for l in range(max(0, n + m + 1 - d), min(d, n + m + 1)): #photon number on first output mode
                k = np.arange(max(0, l - m), min(l + 1, n + 1, d))
                U[n, m, l, n + m - l] = np.sum(bs_coeff(n, m, k, l))
    
    return np.kron(U, np.conj(U))

def f(m, k, *arg):
    temp = np.zeros([m], dtype = 'int')
    temp[::k] = 1
    temp[np.array(*arg)] = 0
    return tuple(temp)

def random_photon_state(n_ph, lost_ph, loss):
    c = np.zeros([2] * n_ph)
    if lost_ph == 0:
        c[tuple([1] * n_ph)] = (1 - loss) ** n_ph
        return c
    
    if lost_ph == n_ph:
        c[tuple([0] * n_ph)] = loss ** n_ph
        return c
    
    photon_array = list(combinations(np.arange(n_ph), lost_ph))

    for photon in photon_array:
        c[f(n_ph, 1, np.array(photon))] = loss ** lost_ph * (1 - loss) ** (n_ph - lost_ph)
    return c

def canonicalize(chi, n_ph, loss):
    #c = np.copy(state)
    A = np.zeros([n_ph, chi, chi], dtype = 'complex64')
    Lambda = np.zeros([n_ph - 1, chi], dtype = 'float32')
    charge = np.zeros([n_ph + 1, chi], dtype = 'int32')
    charge_d = n_ph + 1
    
    if n_ph == 1:
        c = np.zeros((2, 2))
        for lost_ph in range(n_ph + 1):
            c[lost_ph, :] = random_photon_state(n_ph, lost_ph, loss)
            charge[0, lost_ph] = n_ph - lost_ph
        A[n_ph - 1, 0, 0] = c[0, 1]
        A[n_ph - 1, 1, 0] = c[1, 0]
        return A, Lambda, charge
    
    size_ = np.array([n_ph + 1], dtype = 'int32')
    size_ = np.append(size_, [2] * n_ph)
    c = np.zeros(size_)
    for lost_ph in range(n_ph + 1):
        c[lost_ph, :, :] = random_photon_state(n_ph, lost_ph, loss)
        charge[0, lost_ph] = n_ph - lost_ph
        
    c = c.reshape(n_ph + 1, 2, -1)

    pre_tot = n_ph + 1
    for l in range(n_ph - 1):
        tot = 0; check = 0
        for tau in range(n_ph + 1):
            l_bond_0 = np.nonzero(charge[l, :pre_tot] - tau == 0)[0][:min(chi, 2 ** (n_ph - l + 1))]
            l_bond_1 = np.nonzero(charge[l, :pre_tot] - tau == 1)[0][:min(chi, 2 ** (n_ph - l + 1))]
            l_bond = np.union1d(l_bond_0, l_bond_1)[:min(chi, 2 ** (n_ph - l))]
            if len(l_bond) == 0:
                continue
            c_temp = np.vstack((c[l_bond_0, 0, :], c[l_bond_1, 1, :]))
            u, v, w = np.linalg.svd(c_temp, full_matrices = False)
            
            len_ = int(np.sum(v > 10 ** (-10)))
            if len_ == 0:
                continue
            tot += len_
            
            charge[l + 1, tot - len_:tot] = tau
            if check == 0:
                temp_w = w[:len_, :]
                check = 1
            else:
                temp_w = np.vstack((temp_w, w[:len_, :]))
            
            Lambda[l, tot - len_:tot] = v[:len_]

            if len(l_bond_0) > 0:
                u0 = u[:len(l_bond_0), :len_]
                A[l, l_bond_0, tot - len_:tot] = u0
            if len(l_bond_1) > 0:
                u1 = u[len(l_bond_0):len(l_bond_0) + len(l_bond_1), :len_]
                A[l, l_bond_1, tot - len_:tot] = u1
        if l == n_ph - 2 :
            continue
        c = np.matmul(np.diag(Lambda[l, :tot]), temp_w).reshape(tot, 2, -1)
        pre_tot = tot
    
    c = np.matmul(np.diag(Lambda[l, :tot]), temp_w).reshape(tot, 2)
    if tot == 1:
        print("here")
        A[n_ph - 1, 0, 0] = np.sum(c)
    elif charge[0, n_ph] == 0:
        A[n_ph - 1, 0, 0] = c[0, 0]
        A[n_ph - 1, 1, 0] = c[1, 1]
    else:
        print("warning")
        A[n_ph - 1, 1, 0] = c[0, 0]
        A[n_ph - 1, 0, 0] = c[1, 1]
        
    return A, Lambda, charge

class MPO:
    def __init__(self, n, m, loss, chi):
        self.n = n # Number of modes
        self.m = m # Number of photons
        self.d = int(m + 1) # Local Hilbert space dimension
        self.loss = loss
        self.chi = chi # Maximum bond dimension
        self.TotalProbPar = np.zeros([n], dtype = 'float32')
        self.SingleProbPar = np.zeros([n], dtype = 'float32')
        self.EEPar = np.zeros([n - 1,n], dtype = 'float32')
        self.update_time = 0
        self.U_time = 0
        self.svd_time = 0
        self.theta_time = 0
        self.align_init_time = 0
        self.align_info_time = 0
        self.index_time = 0
        self.copy_time = 0
        self.align_time = 0
        self.other_time = 0
        self.largest_C = 0
        self.largest_T = 0

    def MPOInitialization(self):
        self.Gamma = np.zeros([self.n, self.chi, self.chi], dtype=data_type)# modes, alpha, alpha
        self.Lambda = np.zeros([self.n - 1, self.chi], dtype=float_type)# modes - 1, alpha
        self.Lambda_edge = np.ones(self.chi, dtype=float_type) # edge lambda (for first and last site) don't exists and are ones
        self.charge = self.d * np.ones([self.n + 1, self.chi, 2], dtype=int_type) # Initialize initial charges for all bonds to the impossible charge d.
        self.charge[:, 0, :] = 0
        
        Gamma, Lambda, charge = canonicalize(chi, self.m, self.loss)
        
        self.Gamma[:self.m, :, :] = Gamma
        self.Lambda[:self.m - 1, :] = Lambda
        self.charge[:self.m + 1, :, 0] = charge
        self.charge[:self.m + 1, :, 1] = charge
        
        for i in range(self.m, self.n):
            self.Gamma[i, 0, 0] = 1

        for i in range(self.m - 1, self.n - 1):
            self.Lambda[i, 0] = 1

        for i in range(self.n + 1):
            idx = np.lexsort((self.charge[i, :, 1], self.charge[i, :, 0]))
            self.charge[i] = self.charge[i, idx]
            if i > 0:
                self.Gamma[i - 1] = self.Gamma[i - 1][:, idx]
                if i < n:
                    self.Lambda[i - 1] = self.Lambda[i - 1, idx]
            if i < n:
                self.Gamma[i] = self.Gamma[i, idx]

    # Gives the range of left, center and right hand side charge values when center charge is fixed to tau
    def charge_range(self, location, tau):
        # Speficying allowed left and right charges
        if location == 'left':
            min_charge_l = max_charge_l = self.m # The leftmost site must have all photons to the right, hence charge can only be m
        else:
            min_charge_l, max_charge_l = tau, self.d - 1 # Left must have more or equal photons to its right than center
        # Possible center site charge
        min_charge_c, max_charge_c = 0, self.d - 1 # The center charge is summed so returns 0 and maximum possible charge.
        # Possible right site charge
        if location == 'right':
            min_charge_r = max_charge_r = 0 # The rightmost site must have all photons to the left, hence charge can only be 0
        else:    
            min_charge_r, max_charge_r = 0, tau # Left must have more or equal photons to its right than center
        
        return min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r

    def MPOtwoqubitUpdate(self, l, r):
        seed = np.random.randint(0, 13579)
        self.MPOtwoqubitUpdateDevice(l, r, seed)

    #MPO update after a two-qudit gate        
    def MPOtwoqubitUpdateDevice(self, l, r, seed):
        
        # Initializing unitary matrix on GPU
        np.random.seed(seed)
        start = time.time()
        U = Rand_U(self.d, r)
        #print(U_r[0,0,0])
        self.U_time += time.time() - start

        # Determining the location of the two qubit gate
        left = "Left"
        center = "Center"
        right = "Right"
        if l == 0:
            location = left
            LL = self.Lambda_edge[:]
            LC = self.Lambda[l,:]
            LR = self.Lambda[l+1,:]
        elif l == self.n - 2:
            location = right
            LL = self.Lambda[l-1,:]
            LC = self.Lambda[l,:]
            LR = self.Lambda_edge[:]
        else:
            location = center
            LL = self.Lambda[l-1,:]
            LC = self.Lambda[l,:]
            LR = self.Lambda[l+1,:]
        
        Glc = self.Gamma[l,:]
        Gcr = self.Gamma[l+1,:]

        # charge of corresponding index (bond charge left/center/right)
        CL = self.charge[l]
        CC = self.charge[l+1]
        CR = self.charge[l+2]

        start = time.time()
        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.d, self.chi, CL, CC, CR)
        self.align_info_time += aligner.align_info_time
        self.index_time += aligner.index_time
        self.align_init_time += time.time() - start
        start = time.time()
        # Obtaining aligned charges
        cNewL_obj, cNewR_obj, change_charges_C, change_idx_C = aligner.cNewL, aligner.cNewR, aligner.change_charges_C, aligner.change_idx_C
        d_cNewL_obj, d_cNewR_obj = map(aligner.to_cupy, [cNewL_obj, cNewR_obj])
        #d_change_charges_C, d_change_idx_C = map(cp.array, [change_charges_C, change_idx_C])
        d_change_charges_C, d_change_idx_C = change_charges_C, change_idx_C
        self.copy_time = time.time() - start
        start = time.time()
        # Obtaining aligned data
        CC_obj, LL_obj, LC_obj, LR_obj, Glc_obj, Gcr_obj = map(aligner.make_data_obj, ['CC','LL','LC','LR','Glc','Gcr'], [True]*6, [CC, LL, LC, LR, Glc, Gcr], [[0]]*4+[[0,0]]*2)
        d_CC_obj, d_LL_obj, d_LC_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.to_cupy, [CC_obj, LL_obj, LC_obj, LR_obj, Glc_obj, Gcr_obj])
        d_LL_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.align_data, [d_LL_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj])
       
        self.align_time += time.time() - start

        # Storage of generated data
        d_new_Gamma_L = []
        d_new_Gamma_R = []
        new_Lambda = np.array([], dtype=float_type)
        new_charge_0 = np.array([], dtype=int_type)
        new_charge_1 = np.array([], dtype=int_type)
        tau_array = [0]

        for charge_c_0 in range(self.d):
            for charge_c_1 in range(self.d):
                    
                start = time.time()

                # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
                min_charge_l_0, max_charge_l_0, min_charge_c_0, max_charge_c_0, min_charge_r_0, max_charge_r_0 = self.charge_range(location, charge_c_0)
                min_charge_l_1, max_charge_l_1, min_charge_c_1, max_charge_c_1, min_charge_r_1, max_charge_r_1 = self.charge_range(location, charge_c_1)
                # Selecting data according to charge bounds
                d_cl_obj, d_cc_obj, d_cr_obj, d_ll_obj, d_lc_obj, d_lr_obj = map(aligner.select_data,
                                                                                [d_cNewL_obj, d_CC_obj, d_cNewR_obj, d_LL_obj, d_LC_obj, d_LR_obj],
                                                                                [min_charge_l_0, min_charge_c_0, min_charge_r_0]*2,
                                                                                [max_charge_l_0, max_charge_c_0, max_charge_r_0]*2,
                                                                                [min_charge_l_1, min_charge_c_1, min_charge_r_1]*2,
                                                                                [max_charge_l_1, max_charge_c_1, max_charge_r_1]*2)
                d_glc_obj = aligner.select_data(d_Glc_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1,
                                                           min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1)
                d_gcr_obj = aligner.select_data(d_Gcr_obj, min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1,
                                                           min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)

                # Skip if any selection must be empty
                if d_cl_obj.data.shape[0] * d_cr_obj.data.shape[0] * d_cl_obj.data.shape[1] * d_cr_obj.data.shape[1] == 0:
                    tau_array.append(0)
                    continue

                self.align_time += time.time() - start

                start = time.time()
                #print('glc: {}, gcr: {}, ll: {}, lc: {}, lr: {}, cl: {}, cc: {}, cr: {}.'.format(d_glc_obj.data, d_gcr_obj.data, d_ll_obj.data, d_lc_obj.data, d_lr_obj.data, d_cl_obj.data, d_cc_obj.data, d_cr_obj.data))
                d_C_obj = update_MPO(self.d, charge_c_0, charge_c_1, U, d_glc_obj, d_gcr_obj, d_cl_obj, d_cc_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
                d_T_obj = d_C_obj.clone()
                d_T_obj.data = np.multiply(d_C_obj.data, d_lr_obj.data)
                d_C = aligner.compact_data(d_C_obj)
                d_T = aligner.compact_data(d_T_obj)
                #print('T: ', d_T)
                # s.synchronize()
                dt = time.time() - start
                self.largest_T = max(dt, self.largest_T)
                self.theta_time += dt
                
                # SVD
                start = time.time()
                d_V, d_Lambda, d_W = np.linalg.svd(d_T, full_matrices = False)
                d_V = np.asarray(d_V)
                d_Lambda = np.asarray(d_Lambda)
                d_W = np.matmul(np.conj(d_V.T), d_C)
                Lambda = d_Lambda
                # Lambda = cp.asnumpy(d_Lambda)
                # s.synchronize()
                #V, W = map(cp.array, [V, W])
                self.svd_time += time.time() - start

                # Store new results
                #print('V: ', d_V)
                d_new_Gamma_L = d_new_Gamma_L + [d_V[:, i] for i in range(len(Lambda))]
                d_new_Gamma_R = d_new_Gamma_R + [d_W[i, :] for i in range(len(Lambda))]
                new_Lambda = np.append(new_Lambda, Lambda)
                new_charge_0 = np.append(new_charge_0, np.repeat(np.array(charge_c_0, dtype=int_type), len(Lambda)))
                new_charge_1 = np.append(new_charge_1, np.repeat(np.array(charge_c_1, dtype=int_type), len(Lambda)))
                tau_array.append(len(Lambda))
        
        start = time.time()

        # Number of singular values to save
        num_lambda = int(min(new_Lambda.shape[0], self.chi))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx_select = np.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
        else:
            idx_select = np.array([], dtype=int_type)
        
        # Initialize selected and sorted Gamma outputs
        Gamma0Out = Aligner.make_data_obj('Glc', False, np.zeros([self.chi, self.chi], dtype = data_type), [0, 0])
        Gamma1Out = Aligner.make_data_obj('Gcr', False, np.zeros([self.chi, self.chi], dtype = data_type), [0, 0])

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for charge_c_0 in range(self.d):
            for charge_c_1 in range(self.d):
                tau = charge_c_0 * self.d + charge_c_1 # charge at center
                # Selecting gamma that will be modified. Modifying gamma will modify Gamma (because they are pointers).
                min_charge_l_0, max_charge_l_0, _, _, min_charge_r_0, max_charge_r_0 = self.charge_range(location, charge_c_0)
                min_charge_l_1, max_charge_l_1, _, _, min_charge_r_1, max_charge_r_1 = self.charge_range(location, charge_c_1)
                idx_gamma0_0, idx_gamma0_1 = aligner.get_select_index(Gamma0Out, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 0, self.d, 0, self.d)
                idx_gamma1_0, idx_gamma1_1 = aligner.get_select_index(Gamma1Out, 0, self.d, 0, self.d, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
                # Finding bond indices (tau_idx) that are in the largest num_lambda singular values and for center charge tau.
                # idx_select[indices] = tau_idx
                tau_idx, indices, _ = np.intersect1d(idx_select, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

                if len(tau_idx) * idx_gamma0_0.shape[0] * idx_gamma0_1.shape[0] * idx_gamma1_0.shape[0] * idx_gamma1_1.shape[0] == 0:
                    continue

                # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
                # d_V = cp.array([d_new_Gamma_L[i] for i in tau_idx], dtype=data_type)
                # d_W = cp.array([d_new_Gamma_R[i] for i in tau_idx], dtype=data_type)
                # d_V = d_V.T
                # V, W = map(cp.asnumpy, [d_V, d_W])
                V = np.array([d_new_Gamma_L[i] for i in tau_idx], dtype = 'complex64')
                W = np.array([d_new_Gamma_R[i] for i in tau_idx], dtype = 'complex64')
                V = V.T

                # Calculating output gamma
                # Left
                Gamma0Out.data[idx_gamma0_0.reshape(-1,1), idx_gamma0_1[indices].reshape(1,-1)] = V
                # Right
                Gamma1Out.data[idx_gamma1_0[indices].reshape(-1,1), idx_gamma1_1.reshape(1,-1)] = W
                #print('V, gamma, Gamma: ', V, gamma1out.data, Gamma1Out.data)

        #print('Gamma: ', Gamma0Out.data, Gamma1Out.data)
        # Select charges that corresponds to the largest num_lambda singular values
        new_charge_0 = new_charge_0[idx_select]
        new_charge_1 = new_charge_1[idx_select]
        # Sort the new charges
        idx_sort = np.lexsort((new_charge_1, new_charge_0)) # Indices that will sort the new charges
        new_charge_0 = new_charge_0[idx_sort]
        new_charge_1 = new_charge_1[idx_sort]
        # Update charges (modifying CC modifies self.dcharge by pointer)
        CC[:num_lambda, 0] = new_charge_0
        CC[:num_lambda, 1] = new_charge_1
        CC[num_lambda:] = self.d # Charges beyond num_lambda are set to impossible values d
        
        # Selecting and sorting Lambda
        new_Lambda = new_Lambda[idx_select]
        new_Lambda = new_Lambda[idx_sort]

        if new_Lambda.shape[0] == 0:
            print(0)
        else:
            print(np.max(new_Lambda))

        LC[:num_lambda] = new_Lambda
        LC[num_lambda:] = 0

         # Sorting Gamma
        Gamma0Out.data[:, :num_lambda] = Gamma0Out.data[:, idx_sort]
        Gamma1Out.data[:num_lambda] = Gamma1Out.data[idx_sort]
        #print('Gamma: ', Gamma0Out.data, Gamma1Out.data)
        if location == left:
            self.Gamma[0, :, :] = Gamma0Out.data[:, :]; self.Gamma[1, :, :] = Gamma1Out.data[:, :]
        elif location == right:
            self.Gamma[self.n - 2, :, :min(chi, self.d ** 2)] = Gamma0Out.data[:, :min(chi, self.d ** 2)]; self.Gamma[self.n - 1, :min(chi, self.d ** 2), 0] = Gamma1Out.data[:min(chi, self.d ** 2), 0]
        else:
            self.Gamma[l, :, :] = Gamma0Out.data; self.Gamma[l + 1, :, :] = Gamma1Out.data
        
        #print(self.dGamma[0,0,0])
        self.other_time += time.time() - start



    def RCS1DOneCycleUpdate(self, k):
        if k < self.n / 2:
            temp1 = 2 * k + 1
            temp2 = 2
            l = 2 * k
            while l >= 0:
                if temp1 > 0:
                    T = my_cv.rvs(2 * k + 2, temp1)
                    temp1 -= 2
                else:
                    T = my_cv.rvs(2 * k + 2, temp2)
                    temp2 += 2
                self.MPOtwoqubitUpdate(l, np.sqrt(1 - T))
                l -= 1
        else:
            temp1 = 2 * self.n - (2 * k + 3)
            temp2 = 2
            l = self.n - 2
            for i in range(2 * self.n - 2 * k - 2):
                if temp1 >= 0:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp1)
                    temp1 -= 2
                else:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp2)
                    temp2 += 2
                self.MPOtwoqubitUpdate(l, np.sqrt(1 - T))
                l -= 1    
        
    def RCS1DMultiCycle(self):
        
        start = time.time()

        self.MPOInitialization()    
        self.TotalProbPar[0] = self.TotalProbFromMPO()
        self.EEPar[:, 0] = self.MPOEntanglementEntropy()
        
        for k in range(self.n - 1):
            self.RCS1DOneCycleUpdate(k)
            self.TotalProbPar[k + 1] = self.TotalProbFromMPO()
            self.EEPar[:, k + 1] = self.MPOEntanglementEntropy()
        
        print("m: {:.2f}. Total time: {:.2f}. Update time: {:.2f}. U time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. Align init time: {:.2f}. Align info time: {:.2f}. Index time: {:.2f}. Copy time: {:.2f}. Align time: {:.2f}. Other_time: {:.2f}. Largest array dimension: {:.2f}. Longest time for single matrix: {:.8f}".format(m, time.time()-start, self.update_time, self.U_time, self.theta_time, self.svd_time, self.align_init_time, self.align_info_time, self.index_time, self.copy_time, self.align_time, self.other_time, self.largest_C, self.largest_T))

        return self.TotalProbPar, self.EEPar
    
    def TotalProbFromMPO(self):
        R = self.Gamma[self.n - 1, :, 0]
        RTemp = np.copy(R)
        for k in range(self.n - 2):
            idx = np.array([], dtype = 'int32')
            for ch in range(self.d):
                idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 0] == ch), np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 1] == ch), np.nonzero(self.Lambda[self.n - 1 - k - 1, :] > 0))))
            R = np.matmul(self.Gamma[self.n - 1 - k - 1, :, idx].T, RTemp[idx].reshape(-1))
            RTemp = np.copy(R)
        idx = np.array([], dtype = 'int32')
        for ch in range(self.d):
            idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[1, :, 0] == ch), np.intersect1d(np.nonzero(self.charge[1, :, 1] == ch), np.nonzero(self.Lambda[0, :] > 0))))
        res = np.matmul(self.Gamma[0, :, idx].T, RTemp[idx].reshape(-1))
        return np.sum(res)
        
    
    def MPOEntanglementEntropy(self):      
        Output = np.zeros([self.n - 1])
        sq_lambda = np.copy(self.Lambda ** 2)
        for i in range(self.n - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i, :]))
        
        
        #Output = Output * (1 - self.loss ** self.m)
        return Output




class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1)
my_cv = my_pdf(a = 0, b = 1, name='my_pdf')

def EntropyFromColumn(InputColumn):
    Output = -np.nansum(InputColumn * np.log2(InputColumn))
    return Output

def RenyiFromColumn(InputColumn, alpha):
    Output = np.log2(np.nansum(InputColumn ** alpha)) / (1 - alpha)
    return Output

def ColumnSumToOne(InputColumn):
    return InputColumn / np.sum(InputColumn)

def multi_f(args):
    i, n, m, loss, chi = args
    boson = MPO(n, m, loss, chi)
    return boson.RCS1DMultiCycle()

def RCS1DMultiCycleAvg(NumSample, n, m, loss, chi):
    TotalProbAvg = np.zeros([n])
    EEAvg = np.zeros([n - 1, n])

    TotalProbTot = np.zeros([n])
    EETot = np.zeros([n - 1, n])
    
    res = multi_f([0, n, m, loss, chi])
    
    TotalProbTot += res[0]
    EETot += res[1]

    TotalProbAvg = TotalProbTot / NumSample
    EEAvg = EETot / NumSample

    return TotalProbAvg,  EEAvg

if __name__ == "__main__":
    EE_tot = []; prob_tot = []
    for m in [2]:
        NumSample = 1; n = 16; loss = 1 - 1.2 * m ** (1 / 2) / m
        chi = 4
        
        Totprob, EE = RCS1DMultiCycleAvg(NumSample, n, m, loss, chi)