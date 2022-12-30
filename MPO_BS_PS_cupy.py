'''Full simulation code containing the Device method (cupy, unified update)'''
import numpy as np
import cupy as cp
from scipy.stats import rv_continuous

from mpo_sort import Aligner
from cuda_kernels import Rand_U, update_MPO

import time
import os
import sys
import pickle
from itertools import combinations
from filelock import FileLock

from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size() - 1
# num_gpu_ranks = num_gpu_ranks // 16 * 4
rank = comm.Get_rank()
# print(rank)
# gpu = (rank%17-1)%num_gpus_per_node
gpu = rank % 4
# if rank % 17 != 0:
cp.cuda.Device(gpu).use()
print('rank {} using gpu {}'.format(rank, gpu))


# mempool.set_limit(size=2.5 * 10**9)  # 2.3 GiB

np.random.seed(1)
np.set_printoptions(precision=3)

data_type = np.complex64
float_type = np.float32
int_type = np.int32

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

def canonicalize(d, chi, n_ph, loss):
    #c = np.copy(state)
    A = np.zeros([n_ph, chi, chi], dtype = 'complex64')
    Lambda = np.zeros([n_ph - 1, chi], dtype = 'float32')
    charge = d*np.ones([n_ph + 1, chi], dtype = 'int32')
    charge_d = n_ph + 1
    
    if n_ph == 1:
        c = d*np.ones((2, 2))
        for lost_ph in range(n_ph + 1):
            c[lost_ph, :] = random_photon_state(n_ph, lost_ph, loss)
            charge[0, lost_ph] = n_ph - lost_ph
        A[n_ph - 1, 0, 0] = c[0, 1]
        A[n_ph - 1, 1, 0] = c[1, 0]
        return A, Lambda, charge
    
    size_ = np.array([n_ph + 1], dtype = 'int32')
    size_ = np.append(size_, [2] * n_ph)
    c = d*np.ones(size_)
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
    def __init__(self, n, m, d, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
        self.n = n
        self.m = m
        self.d = d
        self.K = m
        self.loss = loss
        self.init_chi = init_chi
        self.chi = chi
        self.errtol = errtol
        self.TotalProbPar = np.zeros([n], dtype = 'float32')
        self.SingleProbPar = np.zeros([n], dtype = 'float32')
        self.EEPar = np.zeros([n - 1, n], dtype = 'float32')      
        self.REPar = np.zeros([n - 1, n, 5], dtype = 'float32')
        self.PS = PS
        self.normalization = None
        self.update_time = 0
        self.U_time = 0
        self.svd_time = 0
        self.theta_time = 0
        self.align_init_time = 0
        self.align_info_time = 0
        self.index_time = 0
        self.copy_time = 0
        self.align_time = 0
        self.before_loop_other_time = 0
        self.segment1_time = 0
        self.segment2_time = 0
        self.segment3_time = 0
        self.largest_C = 0
        self.largest_T = 0

    def MPOInitialization1(self):
        
        chi = self.chi; init_chi = self.init_chi; d = self.d; K = self.K

        self.Lambda_edge = np.ones(chi, dtype = 'float32') # edge lambda (for first and last site) don't exists and are ones
        self.Lambda = np.zeros([init_chi, self.n - 1], dtype = 'float32')
        self.Gamma = np.zeros([init_chi, init_chi, self.n], dtype = 'complex64')  
        self.charge = d * np.ones([init_chi, self.n + 1, 2], dtype = 'int32')
        self.charge[0] = 0
        
        rho = np.zeros([d, d], dtype = 'complex64')
        rho[0, 0] = self.loss
        rho[1, 1] = 1 - self.loss

        if self.PS == None:
            for i in range(d):
                self.charge[i, 0, 0] = i
                self.charge[i, 0, 1] = i
            #pre_chi = d
            updated_bonds = np.array([bond for bond in range(d)])
        else:
            self.charge[0, 0, 0] = self.PS
            self.charge[0, 0, 1] = self.PS
            # pre_chi = 1
            updated_bonds = np.array([0])

        for i in range(K - 1):
            print('Initializing mode ', i)
            #chi_ = 0
            #for j in range(pre_chi):
            bonds_updated = np.zeros(d**2)
            for j in updated_bonds:
                if self.charge[j, i, 0] == d:
                    c1 = 0
                else:
                    c1 = self.charge[j, i, 0]
                for ch_diff1 in range(c1, -1, -1):
                    if self.charge[j, i, 1] == d:
                        c2 = 0
                    else:
                        c2 = self.charge[j, i, 1]
                    for ch_diff2 in range(c2, -1, -1):
                        if np.abs(rho[ch_diff1, ch_diff2]) <= self.errtol:
                            continue
                        #self.Gamma_temp[j, chi_, i] = sq[ch_diff1, ch_diff2]
                        #self.charge[chi_, i + 1, 0] = c1 - ch_diff1
                        #self.charge[chi_, i + 1, 1] = c2 - ch_diff2
                        self.Gamma[j, (c1 - ch_diff1) * d + c2 - ch_diff2, i] = rho[ch_diff1, ch_diff2]
                        self.charge[(c1 - ch_diff1) * d + c2 - ch_diff2, i + 1, 0] = c1 - ch_diff1
                        self.charge[(c1 - ch_diff1) * d + c2 - ch_diff2, i + 1, 1] = c2 - ch_diff2
                        bonds_updated[(c1 - ch_diff1) * d + c2 - ch_diff2] = 1
                        #chi_ += 1
            # self.Lambda[:chi_, i] = 1
            #pre_chi = chi_
            updated_bonds = np.where(bonds_updated == 1)[0]
            self.Lambda[updated_bonds, i] = 1
            # print('Chi ', chi_)

        print('Computing Gamma')
        # for j in range(pre_chi):
        for j in updated_bonds:
            if self.charge[j, K - 1, 0] == d:
                c0 = 0
            else:
                c0 = self.charge[j, K - 1, 0]
            if self.charge[j, K - 1, 1] == d:
                c1 = 0
            else:
                c1 = self.charge[j, K - 1, 1]
            self.Gamma[j, 0, K - 1] = rho[c0, c1]
        
        for i in range(self.m - 1, self.n - 1):
            self.Lambda[0, i] = 1
            self.charge[0, i + 1, 0] = 0
            self.charge[0, i + 1, 1] = 0
        
        print('Update gamma from gamme_temp')
        for i in range(self.m):
            self.Gamma[:, :, i] = np.multiply(self.Gamma[:, :, i], self.Lambda[:, i].reshape(1, -1))

        print('Update the rest of gamma values to 1')
        for i in range(self.m, self.n):
            self.Gamma[0, 0, i] = 1

        print('Array transposition')
        self.Gamma = np.transpose(self.Gamma, (2, 0, 1))
        self.Lambda = np.transpose(self.Lambda, (1, 0))
        self.charge = np.transpose(self.charge, (1, 0, 2))

        print('Start sorting')

        # Sorting bonds based on bond charges
        for i in range(self.n + 1):
            # print('Sorting')
            idx = np.lexsort((self.charge[i, :, 1], self.charge[i, :, 0]))
            # print('Indexing')
            self.charge[i] = self.charge[i, idx]
            if i > 0:
                self.Gamma[i - 1] = self.Gamma[i - 1][:, idx]
                if i < self.n:
                    self.Lambda[i - 1] = self.Lambda[i - 1, idx]
            if i < self.n:
                self.Gamma[i] = self.Gamma[i, idx]

        self.normalization = self.TotalProbFromMPO()
        print('Total probability normalization factor: ', self.normalization)

        charge_temp = np.copy(self.charge)
        Lambda_temp = np.copy(self.Lambda)
        Gamma_temp = np.copy(self.Gamma)
        self.charge = d * np.ones([self.n + 1, chi, 2], dtype = 'int32')
        self.Lambda = np.zeros([self.n - 1, chi], dtype = 'float32')
        self.Gamma = np.zeros([self.n, chi, chi], dtype = 'complex64')

        self.Gamma[:, :init_chi, :init_chi] = Gamma_temp
        self.Lambda[:, :init_chi] = Lambda_temp
        self.charge[:, :init_chi] = charge_temp

        print('Canonicalization update')
        for l in range(self.n - 2, -1, -1):
            self.MPOtwoqubitUpdate(l, 0)
        
        np.random.seed(1)

    def MPOInitialization(self):
        self.Gamma = np.zeros([self.n, self.chi, self.chi], dtype=data_type)# modes, alpha, alpha
        self.Lambda = np.zeros([self.n - 1, self.chi], dtype=float_type)# modes - 1, alpha
        self.Lambda_edge = np.ones(self.chi, dtype=float_type) # edge lambda (for first and last site) don't exists and are ones
        self.charge = self.d * np.ones([self.n + 1, self.chi, 2], dtype=int_type) # Initialize initial charges for all bonds to the impossible charge d.
        self.charge[:, 0, :] = 0
        
        Gamma, Lambda, charge = canonicalize(self.d, self.chi, self.m, self.loss)
        # print(Gamma, Lambda, charge)
        
        self.Gamma[:self.m, :, :] = Gamma
        self.Lambda[:self.m - 1, :] = Lambda
        self.charge[:self.m + 1, :, 0] = charge
        self.charge[:self.m + 1, :, 1] = charge
        
        for i in range(self.m, self.n):
            self.Gamma[i, 0, 0] = 1

        for i in range(self.m - 1, self.n - 1):
            self.Lambda[i, 0] = 1
            self.charge[i + 1, 0, 0] = 0
            self.charge[i + 1, 0, 1] = 0

        for i in range(self.n + 1):
            idx = np.lexsort((self.charge[i, :, 1], self.charge[i, :, 0]))
            self.charge[i] = self.charge[i, idx]
            if i > 0:
                self.Gamma[i - 1] = self.Gamma[i - 1][:, idx]
                if i < self.n:
                    self.Lambda[i - 1] = self.Lambda[i - 1, idx]
            if i < self.n:
                self.Gamma[i] = self.Gamma[i, idx]
        

    # Gives the range of left, center and right hand side charge values when center charge is fixed to tau
    def charge_range(self, location, tau):
        # Speficying allowed left and right charges
        # if location == 'left':
        #     min_charge_l = max_charge_l = self.d - 1 # The leftmost site must have all photons to the right, hence charge can only be m
        # else:
        min_charge_l, max_charge_l = tau, self.d - 1 # Left must have more or equal photons to its right than center
        # Possible center site charge
        min_charge_c, max_charge_c = 0, self.d - 1 # The center charge is summed so returns 0 and maximum possible charge.
        # Possible right site charge
        # if location == 'right':
        #     min_charge_r = max_charge_r = 0 # The rightmost site must have all photons to the left, hence charge can only be 0
        # else:    
        min_charge_r, max_charge_r = 0, tau # Left must have more or equal photons to its right than center
        
        return min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r

    def MPOtwoqubitUpdate(self, l, r):
        seed = np.random.randint(0, 13579)
        # print(seed)
        self.MPOtwoqubitUpdateDevice(l, r, seed)

    #MPO update after a two-qudit gate        
    def MPOtwoqubitUpdateDevice(self, l, r, seed):

        # print('At beginning of mode ', l, ', memory use is ', mempool.used_bytes())
        # for obj in gc.get_objects():
        #     try:
        #         if type(obj) is cp._core.core.ndarray:
        #             print(type(obj), obj.shape)
        #     except:
        #         pass
        # mempool.free_all_blocks()

        chi = self.chi
        # if r == 0:
            # chi = self.init_chi
        
        # Initializing unitary matrix on GPU
        np.random.seed(seed)
        start = time.time()
        d_U_r, d_U_i = Rand_U(self.d, r)
        #print(U_r[0,0,0])
        self.U_time += time.time() - start


        LC = self.Lambda[l,:]
        # Determining the location of the two qubit gate
        left = "Left"
        center = "Center"
        right = "Right"
        if l == 0:
            location = left
            LR = self.Lambda[l+1,:]
        elif l == self.n - 2:
            location = right
            LR = self.Lambda_edge[:]
        else:
            location = center
            LR = self.Lambda[l+1,:]
        
        Glc = self.Gamma[l,:]
        Gcr = self.Gamma[l+1,:]

        # charge of corresponding index (bond charge left/center/right)
        CL = self.charge[l]
        CC = self.charge[l+1]
        CR = self.charge[l+2]

        start = time.time()
        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.d, CL, CC, CR)
        self.align_info_time += aligner.align_info_time
        self.index_time += aligner.index_time
        self.align_init_time += time.time() - start
        start = time.time()
        # Obtaining aligned charges
        cNewL_obj, cNewR_obj, change_charges_C, change_idx_C = aligner.cNewL, aligner.cNewR, aligner.change_charges_C, aligner.change_idx_C
        d_cNewL_obj, d_cNewR_obj = map(aligner.to_cupy, [cNewL_obj, cNewR_obj])
        #d_change_charges_C, d_change_idx_C = map(cp.array, [change_charges_C, change_idx_C])
        d_change_charges_C, d_change_idx_C = cp.array(change_charges_C), cp.array(change_idx_C)
        self.copy_time = time.time() - start
        start = time.time()
        # Obtaining aligned data
        LR_obj, Glc_obj, Gcr_obj = map(aligner.make_data_obj, ['LR','Glc','Gcr'], [True]*3, [LR, Glc, Gcr], [ [0],[0,0],[0,0] ])
        d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.to_cupy, [LR_obj, Glc_obj, Gcr_obj])
        d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.align_data, [d_LR_obj, d_Glc_obj, d_Gcr_obj])
        # print(d_Glc_obj.data.shape, d_Gcr_obj.data.shape)
       
        self.align_time += time.time() - start

        # Storage of generated data
        new_Gamma_L = []
        new_Gamma_R = []
        new_Lambda = np.array([], dtype=float_type)
        new_charge_0 = np.array([], dtype=int_type)
        new_charge_1 = np.array([], dtype=int_type)
        tau_array = [0]

        valid_idx_l_0 = np.where(CL[:, 0] != self.d)[0]
        valid_idx_l_1 = np.where(CL[:, 1] != self.d)[0]
        if valid_idx_l_0.shape[0] > 0:
            largest_cl_0 = np.max(CL[valid_idx_l_0, 0])
        else:
            largest_cl_0 = 0
        if valid_idx_l_1.shape[0] > 0:
            largest_cl_1 = np.max(CL[valid_idx_l_1, 1])
        else:
            largest_cl_1 = 0
        smallest_cr_0 = np.min(CR[:, 0])
        smallest_cr_1 = np.min(CR[:, 1])
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                    
                start = time.time()

                # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
                min_charge_l_0, max_charge_l_0, min_charge_c_0, max_charge_c_0, min_charge_r_0, max_charge_r_0 = self.charge_range(location, charge_c_0)
                min_charge_l_1, max_charge_l_1, min_charge_c_1, max_charge_c_1, min_charge_r_1, max_charge_r_1 = self.charge_range(location, charge_c_1)
                # Selecting data according to charge bounds
                d_cl_obj = aligner.select_data(d_cNewL_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1)
                d_cr_obj = aligner.select_data(d_cNewR_obj, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
                d_lr_obj = aligner.select_data(d_LR_obj, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
                d_glc_obj = aligner.select_data(d_Glc_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1,
                                                           min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1)
                d_gcr_obj = aligner.select_data(d_Gcr_obj, min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1,
                                                           min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)

                self.align_time += time.time() - start
                # Skip if any selection must be empty
                if d_cl_obj.data.shape[0] * d_cr_obj.data.shape[0] * d_cl_obj.data.shape[1] * d_cr_obj.data.shape[1] == 0:
                    tau_array.append(0)
                    del d_cl_obj.data, d_cr_obj.data, d_lr_obj.data, d_glc_obj.data, d_gcr_obj.data
                    continue

                

                start = time.time()
                # print('charge_c_0: {}, charge_c_1:{}, U_r: {}, U_i:{}, glc: {}, gcr: {}, lr: {}, cl: {}, cr: {}.'.format(charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj.data, d_gcr_obj.data, d_lr_obj.data, d_cl_obj.data, d_cr_obj.data))
                # print('charge info: ', d_change_charges_C, d_change_idx_C)
                d_C_obj = update_MPO(self.d, charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj, d_gcr_obj, d_cl_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
                d_T_obj = d_C_obj.clone()
                d_T_obj.data = cp.multiply(d_C_obj.data, d_lr_obj.data)
                d_C = aligner.compact_data(d_C_obj)
                # print('T: ', d_T_obj.data)
                d_T = aligner.compact_data(d_T_obj)
                
                dt = time.time() - start
                self.largest_T = max(dt, self.largest_T)
                self.theta_time += dt
                
                # SVD
                start = time.time()
                d_V, d_Lambda, d_W = cp.linalg.svd(d_T, full_matrices = False)
                d_W = cp.matmul(cp.conj(d_V.T), d_C)
                Lambda = cp.asnumpy(d_Lambda)
                # d_V, d_Lambda, d_W = np.linalg.svd(cp.asnumpy(d_T), full_matrices = False)
                # d_W = np.matmul(np.conj(d_V.T), cp.asnumpy(d_C))
                # Lambda = d_Lambda
                
                self.svd_time += time.time() - start

                #d_V, d_W = map(cp.asnumpy, [d_V, d_W])
                # Store new results
                new_Gamma_L = new_Gamma_L + [d_V[:, i] for i in range(len(Lambda))]
                new_Gamma_R = new_Gamma_R + [d_W[i, :] for i in range(len(Lambda))]
                
                new_Lambda = np.append(new_Lambda, Lambda)
                new_charge_0 = np.append(new_charge_0, np.repeat(np.array(charge_c_0, dtype=int_type), len(Lambda)))
                new_charge_1 = np.append(new_charge_1, np.repeat(np.array(charge_c_1, dtype=int_type), len(Lambda)))
                tau_array.append(len(Lambda))

                # del d_cl_obj.data, d_cr_obj.data, d_lr_obj.data, d_glc_obj.data, d_gcr_obj.data, d_C_obj.data, d_T_obj.data
        
        # del d_cNewL_obj.data, d_cNewR_obj.data, d_LR_obj.data, d_Glc_obj.data, d_Gcr_obj.data

        # Number of singular values to save
        num_lambda = int(min(new_Lambda.shape[0], self.chi))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx_select = np.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
        else:
            idx_select = np.array([], dtype=int_type)
        
        # Initialize selected and sorted Gamma outputs
        d_Gamma0Out = Aligner.make_data_obj('Glc', False, cp.zeros([chi, chi], dtype = data_type), [0, 0])
        d_Gamma1Out = Aligner.make_data_obj('Gcr', False, cp.zeros([chi, chi], dtype = data_type), [0, 0])

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        cp.cuda.stream.get_current_stream().synchronize()
        #print('in ', time.time())
        other_start = time.time()
        
        tau = 0
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                start = time.time()
                # Selecting gamma that will be modified. Modifying gamma will modify Gamma (because they are pointers).
                min_charge_l_0, max_charge_l_0, _, _, min_charge_r_0, max_charge_r_0 = self.charge_range(location, charge_c_0)
                min_charge_l_1, max_charge_l_1, _, _, min_charge_r_1, max_charge_r_1 = self.charge_range(location, charge_c_1)
                idx_gamma0_0, idx_gamma0_1 = aligner.get_select_index(d_Gamma0Out, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 0, self.d, 0, self.d)
                idx_gamma1_0, idx_gamma1_1 = aligner.get_select_index(d_Gamma1Out, 0, self.d, 0, self.d, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
                # Finding bond indices (tau_idx) that are in the largest num_lambda singular values and for center charge tau.
                # idx_select[indices] = tau_idx
                tau_idx, indices, _ = np.intersect1d(idx_select, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
                tau += 1 # This line MUST be before the continue statement

                if len(tau_idx) * idx_gamma0_0.shape[0] * idx_gamma0_1.shape[0] * idx_gamma1_0.shape[0] * idx_gamma1_1.shape[0] == 0:
                    continue
                cp.cuda.stream.get_current_stream().synchronize()
                self.segment1_time += time.time() - start
                # start = time.time()
                # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
                d_V = cp.array([new_Gamma_L[i] for i in tau_idx], dtype = 'complex64')
                d_W = cp.array([new_Gamma_R[i] for i in tau_idx], dtype = 'complex64')
                d_V = d_V.T
                cp.cuda.stream.get_current_stream().synchronize()
                self.segment2_time += time.time() - start
                # start = time.time()

                # Calculating output gamma
                # Left
                d_Gamma0Out.data[idx_gamma0_0.reshape(-1,1), idx_gamma0_1[indices].reshape(1,-1)] = d_V
                # Right
                d_Gamma1Out.data[idx_gamma1_0[indices].reshape(-1,1), idx_gamma1_1.reshape(1,-1)] = d_W
                
                cp.cuda.stream.get_current_stream().synchronize()
                self.segment3_time += time.time() - start
                #print(time.time())
        cp.cuda.stream.get_current_stream().synchronize()
        self.before_loop_other_time += time.time() - other_start
        #print('out ', time.time())

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

        # if new_Lambda.shape[0] == 0:
        #     print(0)
        # else:
        #     print(np.max(new_Lambda))

        LC[:num_lambda] = new_Lambda
        LC[num_lambda:] = 0

         # Sorting Gamma
        d_Gamma0Out.data[:, :num_lambda] = d_Gamma0Out.data[:, idx_sort]
        d_Gamma1Out.data[:num_lambda] = d_Gamma1Out.data[idx_sort]

        if location == right:
            self.Gamma[self.n - 2, :, :min(chi, self.d ** 2)] = cp.asnumpy(d_Gamma0Out.data[:, :min(chi, self.d ** 2)])
            self.Gamma[self.n - 1, :min(chi, self.d ** 2), 0] = cp.asnumpy(d_Gamma1Out.data[:min(chi, self.d ** 2), 0])
        else:
            self.Gamma[l, :, :] = cp.asnumpy(d_Gamma0Out.data)
            self.Gamma[l + 1, :, :] = cp.asnumpy(d_Gamma1Out.data)

        # del d_Gamma0Out.data, d_Gamma1Out.data
        # for i in range(len(new_Gamma_L)-1, -1, -1):
        #     del new_Gamma_L[i]
        # for i in range(len(new_Gamma_R)-1, -1, -1):
        #     del new_Gamma_R[i]


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
                start = time.time()
                self.MPOtwoqubitUpdate(l, np.sqrt(1 - T))
                self.update_time += time.time() - start
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
                start = time.time()
                self.MPOtwoqubitUpdate(l, np.sqrt(1 - T))
                self.update_time += time.time() - start
                l -= 1    
        
        
    def RCS1DMultiCycle(self):
        
        start = time.time()

        self.MPOInitialization()    
        self.TotalProbPar[0] = self.TotalProbFromMPO()
        self.EEPar[:, 0] = self.MPOEntanglementEntropy()
        alpha_array = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for i in range(5):
            self.REPar[:, 0, i] = self.MPORenyiEntropy(alpha_array[i])
        for k in range(self.n - 1):
            self.RCS1DOneCycleUpdate(k)
            self.TotalProbPar[k + 1] = self.TotalProbFromMPO()
            self.EEPar[:, k + 1] = self.MPOEntanglementEntropy()
            for i in range(5):
                self.REPar[:, k + 1, i] = self.MPORenyiEntropy(alpha_array[i])
            '''Initialial total time is much higher than simulation time due to initialization of cuda context.'''
            print("m: {:.2f}. Total time (unreliable): {:.2f}. Update time: {:.2f}. U time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. Align init time: {:.2f}. Align info time: {:.2f}. Index time: {:.2f}. Copy time: {:.2f}. Align time: {:.2f}. Before loop other_time: {:.2f}. Segment1_time: {:.2f}. Segment2_time: {:.2f}. Segment3_time: {:.2f}. Largest array dimension: {:.2f}. Longest time for single matrix: {:.8f}".format(self.m, time.time()-start, self.update_time, self.U_time, self.theta_time, self.svd_time, self.align_init_time, self.align_info_time, self.index_time, self.copy_time, self.align_time, self.before_loop_other_time, self.segment1_time, self.segment2_time, self.segment3_time, self.largest_C, self.largest_T))

        return self.TotalProbPar, self.EEPar, self.REPar

    def TotalProbFromMPO(self):
        R = self.Gamma[self.n - 1, :, 0]
        RTemp = np.copy(R)
        for k in range(self.n - 2):
            idx = np.array([], dtype = 'int32')
            for ch in range(self.d):
                idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 0] == ch), np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 1] == ch), np.nonzero(self.Lambda[self.n - 1 - k - 1] > 0))))
            R = np.matmul(self.Gamma[self.n - 1 - k - 1, :, idx].T, RTemp[idx].reshape(-1))
            RTemp = np.copy(R)
        idx = np.array([], dtype = 'int32')
        for ch in range(self.d):
            idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[1, :, 0] == ch), np.intersect1d(np.nonzero(self.charge[1, :, 1] == ch), np.nonzero(self.Lambda[0, :] > 0))))
        res = np.matmul(self.Gamma[0, :, idx].T, RTemp[idx].reshape(-1))
        tot_prob = np.sum(res)
        print('Probability: ', np.real(tot_prob))
        # if self.normalization != None:
        #     if tot_prob/self.normalization > 1.05 or tot_prob/self.normalization < 0.95:
        #         quit()
        return tot_prob
    
    def MPOEntanglementEntropy(self):      
        Output = np.zeros([self.n - 1])
        sq_lambda = np.copy(self.Lambda ** 2)
        for i in range(self.n - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i]))
        return Output

    def MPORenyiEntropy(self, alpha = 0.5):
        Output = np.zeros([self.n - 1])
        sq_lambda = np.copy(self.Lambda ** 2)
        for i in range(self.n - 1):
            Output[i] += RenyiFromColumn(ColumnSumToOne(sq_lambda[i]), alpha)
        return Output

    def getProb(self, outcome):
        tot_ch = np.sum(outcome)
        charge = [tot_ch]
        for i in range(len(outcome) - 1):
            charge.append(tot_ch - outcome[i])
            tot_ch = tot_ch - outcome[i]

        R = self.Gamma[self.n - 1, :, 0]
        RTemp = np.copy(R);            
            
        for k in range(self.n - 1):
            idx = np.array([], dtype = 'int32')
            idx = np.append(idx,np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 0] == charge[self.n - 1 - k]), np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 1] == charge[self.n - 1 - k]), np.nonzero(self.Lambda[self.n - 1 - k - 1, :] > 0))))
            R = np.matmul(self.Gamma[self.n - 1 - k - 1, :, idx], RTemp[idx].reshape(-1))
            RTemp = np.copy(R)
        idx = np.array([], dtype = 'int32')
        idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[0, :, 0] == np.sum(outcome)), np.nonzero(self.charge[0, :, 1] == np.sum(outcome))))
        return np.abs(np.sum(RTemp[idx]))




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

def RCS1DMultiCycleAvg(n, m, d, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n])
    EEAvg = np.zeros([n - 1, n])
    REAvg = np.zeros([n - 1, n, 5])

    TotalProbTot = np.zeros([n])
    EETot = np.zeros([n - 1, n])
    RETot = np.zeros([n - 1, n, 5])

    boson = MPO(n, m, d, loss, init_chi, chi, errtol, PS)
    Totprob, EE, RE = boson.RCS1DMultiCycle()
    TotalProbTot += Totprob;#TotalProbPar[:,i];
    EETot += EE;#EEPar[:,:,i];
    RETot += RE;#EEPar[:,:,i];
    
    TotalProbAvg = TotalProbTot
    EEAvg = EETot
    REAvg = RETot

    return TotalProbAvg,  EEAvg, REAvg

# if __name__ == "__main__":

#     t0 = time.time()

#     errtol = 10 ** (-7)

#     while True:
#         with FileLock("experiment.pickle.lock"):
#             # work with the file as it is now locked
#             print("Lock acquired.")
#             with open("experiment.pickle", 'rb') as experiment_file:
#                 experiments = pickle.load(experiment_file)
#             for exp_idx in range(len(experiments)):
#                 experiment = experiments[exp_idx]
#                 if experiment['status'] == 'incomplete':
#                     experiments[exp_idx]['status'] = 'in progress'
#                     # Update experiment track file
#                     with open('experiment.pickle', 'wb') as file:
#                         pickle.dump(experiments, file)
#                     # Break the loop once an incomplete experiment is found
#                     break
#             if exp_idx == len(experiments) - 1:
#                 # If loop never broke, no experiment was found
#                 print('All experiments already ran. Exiting.')
#             else:
#                 # experiment = {'n': 32, 'm': 9, 'beta': 1, 'r': 1.44, 'PS': 5, 'd': 6, 'status': 'in progress'}
#                 print('Running experiment: ', experiment)
#                 n, m, beta, r, PS, d = experiment['n'], experiment['m'], experiment['beta'], experiment['r'], experiment['PS'], experiment['d']

#         ideal_ave_photons = m*sinh(r)**2
#         lossy_ave_photons = beta*sqrt(ideal_ave_photons)
#         loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
#         init_chi = d**2
#         chi = int(max(64*2**PS, d**2, 512))
#         print('m is ', m, ', d is ', d, ', r is ', r, ', beta is ', beta, ', chi is ', chi)
#         if (chi > 2048):
#             print('Too large')
#             continue
        
#         begin_dir = './Dec_9/n_{}_m_{}_beta_{}_loss_{}_chi_{}_r_{}_PS_{}'.format(n, m, beta, loss, chi, r, PS)
#         if not os.path.isdir(begin_dir):
#             os.makedirs(begin_dir)

#         if not os.path.isfile(begin_dir + '/EE_{}.npy'.format(id)):
#             Totprob, EE, RE = RCS1DMultiCycleAvg(n, m, d, r, loss, init_chi, chi, errtol, PS)
#             print(Totprob)
#             print(EE)
            
#             np.save(begin_dir + '/EE_{}.npy'.format(id), EE)
#             np.save(begin_dir + '/Totprob_{}.npy'.format(id), Totprob)

#             print("Time cost", time.time() - t0)
#         else:
#             print("Simulation already ran.")
#             quit()

def main():
    exp_idx_beginning = 0
    while True:
    # Loop until all experiments are over
        # with FileLock("./experiment.pickle.lock"):
        #     # work with the file as it is now locked
        #     print("Lock acquired.")
        #     with open("./experiment.pickle", 'rb') as experiment_file:
        #         experiments = pickle.load(experiment_file)
        #     found_experiment = False
        #     for exp_idx in range(exp_idx_beginning, len(experiments)):
        #         experiment = experiments[exp_idx]
        #         if experiment['status'] == 'incomplete':
        #             found_experiment = True
        #             print('Found experiment: ', experiment)
        #             # Break the loop once an incomplete experiment is found
        #             break
        #     exp_idx_beginning = exp_idx + 1
        #     if not found_experiment:
        #         # If loop never broke, no experiment was found
        #         print('All experiments already ran. Exiting.')
        #         comm.Abort()
        #     else:
        #         n, m, beta, loss, PS = experiment['n'], experiment['m'], experiment['beta'], experiment['loss'], experiment['PS']
        #         d = PS + 1
        #         init_chi = d ** 2
        #         chi = int(max(32*2**d, 512))
        #         begin_dir = './SPBSPSResults/n_{}_m_{}_beta_{}_loss_{}_PS_{}'.format(n, m, beta, loss, PS)
        #         if os.path.isfile(begin_dir + 'chi.npy'):
        #             chi_array = np.load(begin_dir + 'chi.npy')
        #             chi = int(np.max(chi_array))
        #             prob = np.load(begin_dir + 'chi_{}_Totprob.npy'.format(chi))
        #             prob = prob[np.where(prob > 0)[0]]
        #             print('prob: ', prob)
        #             if min(prob) != 0:
        #                 error = np.max(prob)/np.min(prob) - 1
        #             error = np.max(error)
        #             print('error: ', error)
        #             if error > 0.1:
        #                 chi *= 2
        #                 print('chi was too small producing error {}. Increasing chi to {}'.format(error, chi))
        #                 status = 'run'
        #             else:
        #                 print('Simulation with suitable accuracy already ran.')
        #                 status = 'skip'
        #         else:
        #             status = 'run'

        #         print('Loss: {}. Chi: {}'.format(loss, chi))
                
        #         if status == 'run':
        #             if chi > 5000:
        #                 print('Required bond-dimension chi too large. Moving on to next experiment.')
        #                 status = 'skip'
        #             elif n > 100:
        #                 print('Too many modes. Moving on to next experiment.')
        #                 status = 'skip'
        #             else:
        #                 # Will run the first found incomplete experiment, set status to in progress
        #                 experiments[exp_idx]['status'] = 'in progress'
        #                 # Update experiment track file
        #                 with open('./experiment.pickle', 'wb') as file:
        #                     pickle.dump(experiments, file)
        #                 status = 'run'

        # if status == 'skip':
        #     continue

        t0 = time.time()
        n, m, beta, errtol = 10, 5, 1.0, 10**(-7)
        ideal_ave_photons = beta * m
        lossy_ave_photons = 0.5 * ideal_ave_photons
        loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
        PS = int((1-loss)*m)
        PS += 1
        d = PS + 1
        PS = None
        d = m+1
        init_chi = d**2
        chi = int(max(4*2**d, d**2, 512))
        chi = 512

        if True:
        # if not os.path.isfile(begin_dir + 'EE.npy'):
            try:
                Totprob, EE, RE = RCS1DMultiCycleAvg(n, m, d, loss, init_chi, chi, 10**(-7), PS)
                print(Totprob)
                print(EE)
                # Saving results
                # if os.path.isfile(begin_dir + 'chi.npy'):
                #     chi_array = np.load(begin_dir + 'chi.npy')
                # else:
                #     chi_array = np.array([])
                # assert not np.sum(chi_array == chi), 'chi {} already in chi array'.format(chi)
                # chi_array = np.append(chi_array, chi)
                # prob_file = begin_dir + 'chi_{}_Totprob.npy'.format(chi)
                # EE_file = begin_dir + 'chi_{}_EE.npy'.format(chi)
                # assert not os.path.isfile(prob_file), '{} exists already. Error.'.format(prob_file)
                # assert not os.path.isfile(EE_file), '{} exists already. Error.'.format(EE_file)
                # np.save(prob_file, Totprob)
                # np.save(EE_file, EE)
                # np.save(begin_dir + 'chi.npy', chi_array)
                # print("Time cost", time.time() - t0)
            except ValueError:
                print('Bad initialization. Next experiment.')
        else:
            print("Simulation already ran.")
        break


if __name__ == "__main__":
    # def mpiabort_excepthook(type, value, traceback):
    #     print('type: ', type)
    #     print('value: ', value)
    #     print('traceback: ', traceback)
    #     print('An exception occured. Aborting MPI')
    #     comm.Abort()
    # sys.excepthook = mpiabort_excepthook
    main()
    # sys.excepthook = sys.__excepthook__