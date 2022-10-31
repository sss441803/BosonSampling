import time

import cupy as cp
import numpy as np

from cuda_kernels import Rand_U, update_MPO

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


data_type = np.complex64
float_type = np.float32
int_type = np.int32



class RankMPO:

    def __init__(self, node_control_rank, d, chi):
        self.node_control_rank = node_control_rank
        self.d = d
        self.chi = chi
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
        

    def FullRankloop(self):

        # Loop executing requests form node
        status = comm.recv(source=self.node_control_rank, tag=100)
        # print('rank: {}, node status: {}'.format(rank, status))
        while status != 'Full Finished':
            self.NodeRankloop()
            # print('Node finished')
            status = comm.recv(source=self.node_control_rank, tag=100)
            # print('rank: {}, node status: {}'.format(rank, status))

    def NodeRankloop(self):

        # Receiving data for the update
        change_charges_C = np.zeros([2, (self.d+1)**2], dtype='int32')
        change_idx_C = np.zeros((self.d+1)**2, dtype='int32')
        r = comm.recv(source=self.node_control_rank, tag=0)
        seed = comm.recv(source=self.node_control_rank, tag=1)
        comm.Recv([change_charges_C, MPI.INT], source=self.node_control_rank, tag=2)
        # print('received change charges: ', change_charges_C)
        comm.Recv([change_idx_C, MPI.INT], source=self.node_control_rank, tag=3)
        changes = comm.recv(source=self.node_control_rank, tag=4)
        cNewL_obj = comm.recv(source=self.node_control_rank, tag=5)
        cNewR_obj = comm.recv(source=self.node_control_rank, tag=6)
        LR_obj = comm.recv(source=self.node_control_rank, tag=7)
        Glc_obj = comm.recv(source=self.node_control_rank, tag=8)
        Gcr_obj = comm.recv(source=self.node_control_rank, tag=9)
        aligner = comm.recv(source=self.node_control_rank, tag=10)
        location = comm.recv(source=self.node_control_rank, tag=11)
        change_charges_C = change_charges_C[:, :changes]
        change_idx_C = change_idx_C[:changes]
        np.random.seed(seed)
        # print('seed: ', seed)
        d_U_r, d_U_i = Rand_U(self.d, r)

        # Moving data to GPU
        d_cNewL_obj, d_cNewR_obj = map(aligner.to_cupy, [cNewL_obj, cNewR_obj])
        d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.to_cupy, [LR_obj, Glc_obj, Gcr_obj])
        d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.align_data, [d_LR_obj, d_Glc_obj, d_Gcr_obj])
        d_change_charges_C, d_change_idx_C = cp.array(change_charges_C), cp.array(change_idx_C)

        # Loop executing requests form node
        status = comm.recv(source=self.node_control_rank, tag=101)
        # print('rank: {}, status: {}'.format(rank, status))
        while status != 'Node Finished':
            charge_c_0 = comm.recv(source=self.node_control_rank, tag=0)
            charge_c_1 = comm.recv(source=self.node_control_rank, tag=1)
            self.RankProcess(charge_c_0, charge_c_1, d_U_r, d_U_i, d_change_charges_C, d_change_idx_C, d_cNewL_obj, d_cNewR_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj, aligner, location)
            # print('Rank finished')
            status = comm.recv(source=self.node_control_rank, tag=101)
            # print('rank: {}, status: {}'.format(rank, status))


    def RankProcess(self, charge_c_0, charge_c_1, d_U_r, d_U_i, d_change_charges_C, d_change_idx_C, d_cNewL_obj, d_cNewR_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj, aligner, location):
                    
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

        start = time.time()
        # print('charge_c_0: {}, charge_c_1:{}, U_r: {}, U_i:{}, glc: {}, gcr: {}, lr: {}, cl: {}, cr: {}.'.format(charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj.data, d_gcr_obj.data, d_lr_obj.data, d_cl_obj.data, d_cr_obj.data))
        # print('charge info: ', d_change_charges_C, d_change_idx_C)
        d_C_obj = update_MPO(self.d, charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj, d_gcr_obj, d_cl_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
        d_T_obj = d_C_obj.clone()
        d_T_obj.data = cp.multiply(d_C_obj.data, d_lr_obj.data)
        d_C = aligner.compact_data(d_C_obj)
        d_T = aligner.compact_data(d_T_obj)
        # print('T: ', d_T)
        
        dt = time.time() - start
        self.largest_T = max(dt, self.largest_T)
        self.theta_time += dt
        
        # SVD
        start = time.time()
        d_V, d_Lambda, d_W = cp.linalg.svd(d_T, full_matrices = False)
        d_W = cp.matmul(cp.conj(d_V.T), d_C)
        self.svd_time += time.time() - start
        
        # Sending results back to node
        comm.Send([cp.asnumpy(d_V), MPI.C_FLOAT_COMPLEX], self.node_control_rank, tag=0)
        comm.Send([cp.asnumpy(d_W), MPI.C_FLOAT_COMPLEX], self.node_control_rank, tag=1)
        comm.Send([cp.asnumpy(d_Lambda), MPI.FLOAT], self.node_control_rank, tag=2)