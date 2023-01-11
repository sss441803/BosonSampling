import time

import cupy as cp
mempool = cp.get_default_memory_pool()

import numpy as np

from ..cuda_kernels import Rand_U, update_MPO
from ..mpo_sort import charge_range

from mpi4py import MPI
comm = MPI.COMM_WORLD


class RankWorker:

    def __init__(self, node_control_rank):
        self.node_control_rank = node_control_rank


    def ExperimentInit(self, n_modes, local_hilbert_space_dimension, bond_dimension):
        self.local_hilbert_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = bond_dimension
        self.svd_time = 0
        self.theta_time = 0
        

    def Simulate(self):
        # Loop executing requests form node
        status = comm.recv(source=self.node_control_rank, tag=100)
        while status != 'Full Finished':
            self.NodeRankloop()
            status = comm.recv(source=self.node_control_rank, tag=100)


    def NodeRankloop(self):

        self.theta_time = 0
        self.svd_time = 0

        # Receiving data for the update
        change_charges_C = np.zeros([2, (self.local_hilbert_space_dimension+1)**2], dtype='int32')
        change_idx_C = np.zeros((self.local_hilbert_space_dimension+1)**2, dtype='int32')
        r = comm.recv(source=self.node_control_rank, tag=0)
        seed = comm.recv(source=self.node_control_rank, tag=1)
        comm.Recv([change_charges_C, MPI.INT], source=self.node_control_rank, tag=2)
        comm.Recv([change_idx_C, MPI.INT], source=self.node_control_rank, tag=3)
        changes = comm.recv(source=self.node_control_rank, tag=4)
        cNewL_obj = comm.recv(source=self.node_control_rank, tag=5)
        cNewR_obj = comm.recv(source=self.node_control_rank, tag=6)
        LR = np.empty(self.bond_dimension, dtype = 'float32')
        Glc = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')  
        Gcr = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')
        comm.Recv([LR, MPI.FLOAT], source=self.node_control_rank, tag=7)
        comm.Recv([Glc, MPI.C_FLOAT_COMPLEX], source=self.node_control_rank, tag=8)
        comm.Recv([Gcr, MPI.C_FLOAT_COMPLEX], source=self.node_control_rank, tag=9)
        aligner = comm.recv(source=self.node_control_rank, tag=10)

        change_charges_C = change_charges_C[:, :changes]
        change_idx_C = change_idx_C[:changes]
        np.random.seed(seed)
        d_U_r, d_U_i = Rand_U(self.local_hilbert_space_dimension, r)

        # Moving data to GPU
        d_cNewL_obj, d_cNewR_obj = map(aligner.to_cupy, [cNewL_obj, cNewR_obj])
        LR_obj, Glc_obj, Gcr_obj = map(aligner.make_data_obj, ['LR','Glc','Gcr'], [True]*3, [LR, Glc, Gcr], [ [0],[0,0],[0,0] ])
        d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.to_cupy, [LR_obj, Glc_obj, Gcr_obj])
        d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.align_data, [d_LR_obj, d_Glc_obj, d_Gcr_obj])
        d_change_charges_C, d_change_idx_C = cp.array(change_charges_C), cp.array(change_idx_C)

        # Loop executing requests form node
        status = comm.recv(source=self.node_control_rank, tag=101)
        # print('rank: {}, status: {}'.format(rank, status))
        while status != 'Node Finished':
            charge_c_0 = comm.recv(source=self.node_control_rank, tag=0)
            charge_c_1 = comm.recv(source=self.node_control_rank, tag=1)
            self.RankProcess(charge_c_0, charge_c_1, d_U_r, d_U_i, d_change_charges_C, d_change_idx_C, d_cNewL_obj, d_cNewR_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj, aligner)
            status = comm.recv(source=self.node_control_rank, tag=101)
        comm.send(self.theta_time, self.node_control_rank, tag=3)
        comm.send(self.svd_time, self.node_control_rank, tag=4)
        del d_LR_obj, d_Glc_obj,d_Gcr_obj, d_change_charges_C, d_change_idx_C
        mempool.free_all_blocks()


    def RankProcess(self, charge_c_0, charge_c_1, d_U_r, d_U_i, d_change_charges_C, d_change_idx_C, d_cNewL_obj, d_cNewR_obj, d_LR_obj, d_Glc_obj, d_Gcr_obj, aligner):
                    
        start = time.time()

        # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
        min_charge_l_0, max_charge_l_0, min_charge_c_0, max_charge_c_0, min_charge_r_0, max_charge_r_0 = charge_range(self.local_hilbert_space_dimension, charge_c_0)
        min_charge_l_1, max_charge_l_1, min_charge_c_1, max_charge_c_1, min_charge_r_1, max_charge_r_1 = charge_range(self.local_hilbert_space_dimension, charge_c_1)
        # Selecting data according to charge bounds
        d_cl_obj = aligner.select_data(d_cNewL_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1)
        d_cr_obj = aligner.select_data(d_cNewR_obj, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
        d_lr_obj = aligner.select_data(d_LR_obj, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
        d_glc_obj = aligner.select_data(d_Glc_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1,
                                                min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1)
        d_gcr_obj = aligner.select_data(d_Gcr_obj, min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1,
                                                min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)

        start = time.time()
        d_C_obj = update_MPO(self.local_hilbert_space_dimension, charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj, d_gcr_obj, d_cl_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
        d_T_obj = d_C_obj.clone()
        d_T_obj.data = cp.multiply(d_C_obj.data, d_lr_obj.data)
        d_C = aligner.compact_data(d_C_obj)
        d_T = aligner.compact_data(d_T_obj)
        self.theta_time += time.time() - start
        
        # SVD
        start = time.time()
        d_V, d_Lambda, d_W = cp.linalg.svd(d_T, full_matrices = False)
        d_W = cp.matmul(cp.conj(d_V.T), d_C)
        self.svd_time += time.time() - start
        
        # Sending results back to node
        comm.Send([cp.asnumpy(d_V), MPI.C_FLOAT_COMPLEX], self.node_control_rank, tag=0)
        comm.Send([cp.asnumpy(d_W), MPI.C_FLOAT_COMPLEX], self.node_control_rank, tag=1)
        comm.Send([cp.asnumpy(d_Lambda), MPI.FLOAT], self.node_control_rank, tag=2)

        del d_V, d_W, d_T, d_C
        mempool.free_all_blocks()