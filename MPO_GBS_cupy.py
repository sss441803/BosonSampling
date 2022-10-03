'''Full simulation code containing the Device method (cupy, unified update)'''
import numpy as np
import cupy as cp

from scipy.stats import rv_continuous
from qutip import squeeze, thermal_dm

import time

from mpo_sort import Aligner
from cuda_kernels import Rand_U, update_MPO

np.random.seed(1)
np.set_printoptions(precision=3)

s = cp.cuda.Stream()
s1 = cp.cuda.Stream(non_blocking=True)

data_type = np.complex64
float_type = np.float32
int_type = np.int32

class MPO:
    def __init__(self, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
        self.n = n
        self.m = m
        self.d = d
        self.r = r
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

    def MPOInitialization(self):
        
        chi = self.init_chi; d = self.d; K = self.K

        self.Lambda_edge = np.ones(chi, dtype = 'float32') # edge lambda (for first and last site) don't exists and are ones
        self.Lambda = np.zeros([chi, self.n - 1], dtype = 'float32')
        self.Gamma = cp.zeros([chi, chi, self.n], dtype = 'complex64')
        self.Gamma_temp = np.zeros([chi, chi, self.n], dtype = 'complex64')        
        self.charge = d * np.ones([chi, self.n + 1, 2], dtype = 'int32')
        self.charge[0] = 0
        
        am = (1 - self.loss) * np.exp(- 2 * self.r) + self.loss
        ap = (1 - self.loss) * np.exp(2 * self.r) + self.loss
        s = 1 / 4 * np.log(ap / am)
        n_th = 1 / 2 * (np.sqrt(am * ap) - 1)
        nn = 40
        
        sq = (squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()).full()[:(d + 1), :(d + 1)]

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
                        if np.abs(sq[ch_diff1, ch_diff2]) <= errtol:
                            continue
                        #self.Gamma_temp[j, chi_, i] = sq[ch_diff1, ch_diff2]
                        #self.charge[chi_, i + 1, 0] = c1 - ch_diff1
                        #self.charge[chi_, i + 1, 1] = c2 - ch_diff2
                        self.Gamma_temp[j, (c1 - ch_diff1) * d + c2 - ch_diff2, i] = sq[ch_diff1, ch_diff2]
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
            self.Gamma_temp[j, 0, K - 1] = sq[c0, c1]
        
        for i in range(self.m - 1, self.n - 1):
            self.Lambda[0, i] = 1
            self.charge[0, i + 1, 0] = 0
            self.charge[0, i + 1, 1] = 0
        
        print('Update gamma from gamme_temp')
        for i in range(self.m):
            self.Gamma[:, :, i] = cp.multiply(cp.array(self.Gamma_temp[:, :, i]), cp.array(self.Lambda[:, i].reshape(1, -1)))

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
            print('Sorting')
            idx = np.lexsort((self.charge[i, :, 1], self.charge[i, :, 0]))
            print('Indexing')
            self.charge[i] = self.charge[i, idx]
            if i > 0:
                self.Gamma[i - 1] = self.Gamma[i - 1][:, idx]
                if i < n:
                    self.Lambda[i - 1] = self.Lambda[i - 1, idx]
            if i < n:
                self.Gamma[i] = self.Gamma[i, idx]
        
        print('Canonicalization update')
        for l in range(self.n - 1):
            self.MPOtwoqubitUpdate(l, 0)

        self.Gamma = self.Gamma[:, :self.chi, :self.chi]
        self.Gamma = cp.asnumpy(self.Gamma)
        self.Lambda = self.Lambda[:, :self.chi]
        self.charge = self.charge[:, :self.chi]

    # Gives the range of left, center and right hand side charge values when center charge is fixed to tau
    def charge_range(self, location, tau):
        # Speficying allowed left and right charges
        if location == 'left':
            min_charge_l = max_charge_l = self.d - 1 # The leftmost site must have all photons to the right, hence charge can only be m
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
        # print(seed)
        self.MPOtwoqubitUpdateDevice(l, r, seed)

    #MPO update after a two-qudit gate        
    def MPOtwoqubitUpdateDevice(self, l, r, seed):

        chi = self.chi
        if r == 0:
            chi = self.init_chi
        
        # Initializing unitary matrix on GPU
        np.random.seed(seed)
        start = time.time()
        d_U_r, d_U_i = Rand_U(self.d, r)
        #print(U_r[0,0,0])
        s.synchronize()
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
                    continue

                

                start = time.time()
                #print('glc: {}, gcr: {}, ll: {}, lc: {}, lr: {}, cl: {}, cc: {}, cr: {}.'.format(d_glc_obj.data, d_gcr_obj.data, d_ll_obj.data, d_lc_obj.data, d_lr_obj.data, d_cl_obj.data, d_cc_obj.data, d_cr_obj.data))
                d_C_obj = update_MPO(self.d, charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj, d_gcr_obj, d_cl_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
                d_T_obj = d_C_obj.clone()
                d_T_obj.data = cp.multiply(d_C_obj.data, d_lr_obj.data)
                d_C = aligner.compact_data(d_C_obj)
                d_T = aligner.compact_data(d_T_obj)
                # print('T: ', d_T)
                s.synchronize()
                dt = time.time() - start
                self.largest_T = max(dt, self.largest_T)
                self.theta_time += dt
                
                # SVD
                start = time.time()
                d_V, d_Lambda, d_W = cp.linalg.svd(d_T, full_matrices = False)
                # d_V = cp.asarray(d_V)
                # d_Lambda = cp.asarray(d_Lambda)
                d_W = cp.matmul(cp.conj(d_V.T), d_C)
                #Lambda = d_Lambda
                Lambda = cp.asnumpy(d_Lambda)
                s.synchronize()
                self.svd_time += time.time() - start

                with s1:
                    d_V, d_W = map(cp.asnumpy, [d_V, d_W])
                    # Store new results
                    if r == 0:
                        new_Gamma_L = new_Gamma_L + [d_V[:, i] for i in range(len(Lambda))]
                        new_Gamma_R = new_Gamma_R + [d_W[i, :] for i in range(len(Lambda))]
                    else:
                        new_Gamma_L = new_Gamma_L + [d_V[:, i] for i in range(len(Lambda))]
                        new_Gamma_R = new_Gamma_R + [d_W[i, :] for i in range(len(Lambda))]
                
                new_Lambda = np.append(new_Lambda, Lambda)
                new_charge_0 = np.append(new_charge_0, np.repeat(np.array(charge_c_0, dtype=int_type), len(Lambda)))
                new_charge_1 = np.append(new_charge_1, np.repeat(np.array(charge_c_1, dtype=int_type), len(Lambda)))
                tau_array.append(len(Lambda))
        
        

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
                tau += 1
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

        if r == 0:
            if location == right:
                self.Gamma[self.n - 2, :, :min(chi, self.d ** 2)] = d_Gamma0Out.data[:, :min(chi, self.d ** 2)]
                self.Gamma[self.n - 1, :min(chi, self.d ** 2), 0] = d_Gamma1Out.data[:min(chi, self.d ** 2), 0]
            else:
                self.Gamma[l, :, :] = d_Gamma0Out.data
                self.Gamma[l + 1, :, :] = d_Gamma1Out.data
        else:
            if location == right:
                self.Gamma[self.n - 2, :, :min(chi, self.d ** 2)] = cp.asnumpy(d_Gamma0Out.data[:, :min(chi, self.d ** 2)])
                self.Gamma[self.n - 1, :min(chi, self.d ** 2), 0] = cp.asnumpy(d_Gamma1Out.data[:min(chi, self.d ** 2), 0])
            else:
                self.Gamma[l, :, :] = cp.asnumpy(d_Gamma0Out.data)
                self.Gamma[l + 1, :, :] = cp.asnumpy(d_Gamma1Out.data)


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
            print("m: {:.2f}. Total time (unreliable): {:.2f}. Update time: {:.2f}. U time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. Align init time: {:.2f}. Align info time: {:.2f}. Index time: {:.2f}. Copy time: {:.2f}. Align time: {:.2f}. Before loop other_time: {:.2f}. Segment1_time: {:.2f}. Segment2_time: {:.2f}. Segment3_time: {:.2f}. Largest array dimension: {:.2f}. Longest time for single matrix: {:.8f}".format(m, time.time()-start, self.update_time, self.U_time, self.theta_time, self.svd_time, self.align_init_time, self.align_info_time, self.index_time, self.copy_time, self.align_time, self.before_loop_other_time, self.segment1_time, self.segment2_time, self.segment3_time, self.largest_C, self.largest_T))

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
        return np.sum(res)
    
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

def RCS1DMultiCycleAvg(NumSample, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n])
    EEAvg = np.zeros([n - 1, n])
    REAvg = np.zeros([n - 1, n, 5])

    TotalProbTot = np.zeros([n])
    EETot = np.zeros([n - 1, n])
    RETot = np.zeros([n - 1, n, 5])

    for i in range(NumSample):
        print("Sample number", i)
        boson = MPO(n, m, d, r, loss, init_chi, chi, errtol, PS)
        Totprob, EE, RE = boson.RCS1DMultiCycle()
        TotalProbTot += Totprob;#TotalProbPar[:,i];
        EETot += EE;#EEPar[:,:,i];
        RETot += RE;#EEPar[:,:,i];
    
    TotalProbAvg = TotalProbTot / NumSample
    EEAvg = EETot / NumSample
    REAvg = RETot / NumSample

    return TotalProbAvg,  EEAvg, REAvg

if __name__ == "__main__":
    t0 = time.time() 
    NumSample = 1; n = 10; m = 4; loss = 0.5; init_chi = 2000; chi = 2000; r = 0.2; d = 5; errtol = 10 ** (-7)
    PS = 4
    Totprob, EE, RE = RCS1DMultiCycleAvg(NumSample, n, m, d, r, loss, init_chi, chi, errtol, PS)
    print(Totprob)
    print(EE)
    print("Time cost", time.time() - t0)
