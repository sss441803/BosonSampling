import time

import cupy as cp
import numpy as np

from cuda_kernels import Rand_U, update_MPO
from mpo_sort import Aligner

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
gpu = rank % 4
if rank != 0:
    cp.cuda.Device(gpu).use()
    print('rank {} using gpu {}'.format(rank, gpu))


data_type = np.complex64
float_type = np.float32
int_type = np.int32



class ChildMPO:

    def __init__(self, n_modes, local_hilbert_space_dimension, bond_dimension):
        self.n_modes = n_modes
        self.local_hilbert_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = bond_dimension

    # Gives the range of left, center and right hand side charge values when center charge is fixed to tau
    def charge_range(self, tau):
        # Speficying allowed left and right charges
        # if location in ['left', 'Left']:
        #     min_charge_l = max_charge_l = self.d - 1 # The leftmost site must have all photons to the right, hence charge can only be m
        # else:
        min_charge_l, max_charge_l = tau, self.local_hilbert_space_dimension - 1 # Left must have more or equal photons to its right than center
        # Possible center site charge
        min_charge_c, max_charge_c = 0, self.local_hilbert_space_dimension - 1 # The center charge is summed so returns 0 and maximum possible charge.
        # Possible right site charge
        # if location == 'right':
        #     min_charge_r = max_charge_r = 0 # The rightmost site must have all photons to the left, hence charge can only be 0
        # else:    
        min_charge_r, max_charge_r = 0, tau # Left must have more or equal photons to its right than center
        
        return min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r
        

    def Childloop(self):
        status = comm.recv(source=0, tag=100)
        if status != 'Finished':
            if rank == 1:
                for _ in range(self.n_modes - 1):
                    self.canonicalize()
                    status = comm.recv(source=0, tag=100)
            # print('rank: {}, status: {}'.format(rank, status))
            while status != 'Finished':
                self.ChildProcess()
                status = comm.recv(source=0, tag=100)
                # print('rank: {}, status: {}'.format(rank, status))
        # while status != 'Finished':
        #     self.ChildProcess()
        #     status = comm.recv(source=0, tag=100)

    def canonicalize(self):
        chi = self.bond_dimension
        self.bond_dimension = self.local_hilbert_space_dimension ** 2
        self.ChildProcess()
        self.bond_dimension = chi


    def ChildProcess(self):

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
        self.comm_time = 0

        LC = np.empty(self.bond_dimension, dtype = 'float32')
        LR = np.empty(self.bond_dimension, dtype = 'float32')
        CL = np.empty([self.bond_dimension, 2], dtype = 'int32')
        CC = np.empty([self.bond_dimension, 2], dtype = 'int32')
        CR = np.empty([self.bond_dimension, 2], dtype = 'int32')
        Glc = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')  
        Gcr = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')
        
        comm.Recv([LC, MPI.FLOAT], source=0, tag=0)
        comm.Recv([LR, MPI.FLOAT], source=0, tag=1)
        comm.Recv([CL, MPI.INT], source=0, tag=2)
        comm.Recv([CC, MPI.INT], source=0, tag=3)
        comm.Recv([CR, MPI.INT], source=0, tag=4)
        comm.Recv([Glc, MPI.C_FLOAT_COMPLEX], source=0, tag=5)
        comm.Recv([Gcr, MPI.C_FLOAT_COMPLEX], source=0, tag=6)
        r = comm.recv(source=0, tag=7)
        seed = comm.recv(source=0, tag=8)
        # print('rank: {} got data'.format(rank))#, LC, LR, CL, CC, CR, Glc, Gcr, r, location, seed)

        # Initializing unitary matrix on GPU
        np.random.seed(seed)
        start = time.time()
        # print('reflectivity ', r)
        d_U_r, d_U_i = Rand_U(self.local_hilbert_space_dimension, r)
        self.U_time += time.time() - start

        start = time.time()
        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.local_hilbert_space_dimension, CL, CC, CR)
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
        valid_idx_l_0 = np.where(CL[:, 0] != self.local_hilbert_space_dimension)[0]
        valid_idx_l_1 = np.where(CL[:, 1] != self.local_hilbert_space_dimension)[0]
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
                min_charge_l_0, max_charge_l_0, min_charge_c_0, max_charge_c_0, min_charge_r_0, max_charge_r_0 = self.charge_range(charge_c_0)
                min_charge_l_1, max_charge_l_1, min_charge_c_1, max_charge_c_1, min_charge_r_1, max_charge_r_1 = self.charge_range(charge_c_1)
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
                #print('glc: {}, gcr: {}, ll: {}, lc: {}, lr: {}, cl: {}, cc: {}, cr: {}.'.format(d_glc_obj.data, d_gcr_obj.data, d_ll_obj.data, d_lc_obj.data, d_lr_obj.data, d_cl_obj.data, d_cc_obj.data, d_cr_obj.data))
                d_C_obj = update_MPO(self.local_hilbert_space_dimension, charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj, d_gcr_obj, d_cl_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
                d_T_obj = d_C_obj.clone()
                d_T_obj.data = cp.multiply(d_C_obj.data, d_lr_obj.data)
                d_C = aligner.compact_data(d_C_obj)
                d_T = aligner.compact_data(d_T_obj)
                # print('T: ', d_T)
                
                dt = time.time() - start
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
        num_lambda = int(min(new_Lambda.shape[0], self.bond_dimension))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx_select = np.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
        else:
            idx_select = np.array([], dtype=int_type)
        
        # Initialize selected and sorted Gamma outputs
        d_Gamma0Out = Aligner.make_data_obj('Glc', False, cp.zeros([self.bond_dimension, self.bond_dimension], dtype = data_type), [0, 0])
        d_Gamma1Out = Aligner.make_data_obj('Gcr', False, cp.zeros([self.bond_dimension, self.bond_dimension], dtype = data_type), [0, 0])

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
                min_charge_l_0, max_charge_l_0, _, _, min_charge_r_0, max_charge_r_0 = self.charge_range(charge_c_0)
                min_charge_l_1, max_charge_l_1, _, _, min_charge_r_1, max_charge_r_1 = self.charge_range(charge_c_1)
                idx_gamma0_0, idx_gamma0_1 = aligner.get_select_index(d_Gamma0Out, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 0, self.local_hilbert_space_dimension, 0, self.local_hilbert_space_dimension)
                idx_gamma1_0, idx_gamma1_1 = aligner.get_select_index(d_Gamma1Out, 0, self.local_hilbert_space_dimension, 0, self.local_hilbert_space_dimension, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
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
        new_charge = np.concatenate([new_charge_0.reshape(-1, 1), new_charge_1.reshape(-1, 1)], axis=1)
        
        # Selecting and sorting Lambda
        new_Lambda = new_Lambda[idx_select]
        new_Lambda = new_Lambda[idx_sort]

        # if new_Lambda.shape[0] == 0:
        #     print(0)
        # else:
        #     print(np.max(new_Lambda))

        # Sorting Gamma
        d_Gamma0Out.data[:, :num_lambda] = d_Gamma0Out.data[:, idx_sort]
        d_Gamma1Out.data[:num_lambda] = d_Gamma1Out.data[idx_sort]

        Gamma0Out = cp.asnumpy(d_Gamma0Out.data)
        Gamma1Out = cp.asnumpy(d_Gamma1Out.data)

        # print('Rank {} finished processing'.format(rank))
        start = time.time()
        comm.Send(new_charge, 0, tag=10)
        comm.Send(new_Lambda, 0, tag=11)
        comm.Send(Gamma0Out, 0, tag=12)
        comm.Send(Gamma1Out, 0, tag=13)
        comm.send(self.U_time, 0, tag=14)
        comm.send(self.svd_time, 0, tag=15)
        comm.send(self.theta_time, 0, tag=16)
        comm.send(self.align_init_time, 0, tag=17)
        comm.send(self.align_info_time, 0, tag=18)
        comm.send(self.index_time, 0, tag=19)
        comm.send(self.copy_time, 0, tag=20)
        comm.send(self.align_time, 0, tag=21)
        comm.send(self.before_loop_other_time, 0, tag=22)
        comm.send(self.segment1_time, 0, tag=23)
        comm.send(self.segment2_time, 0, tag=24)
        comm.send(self.segment3_time, 0, tag=25)
        self.comm_time += time.time() - start
        comm.send(self.comm_time, 0, tag=26)

        # print('Rank {} sent data'.format(rank))