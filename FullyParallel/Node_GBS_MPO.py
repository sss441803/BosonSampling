import time

import cupy as cp
import numpy as np

from cuda_kernels import Rand_U
from mpo_sort import Aligner

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


data_type = np.complex64
float_type = np.float32
int_type = np.int32



class NodeMPO:

    def __init__(self, node_gpu_ranks, d, chi):
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
        self.available_ranks = node_gpu_ranks
        self.running_charges_and_rank = []
        self.num_ranks = len(node_gpu_ranks)
        self.requests = [[[] for _ in range(self.d)] for _ in range(self.d)]
        self.requests_buf = [[[] for _ in range(self.d)] for _ in range(self.d)]


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
        

    def Request(self, charge_c_0, charge_c_1, left_size, right_size, target_rank):
        # Sending information
        comm.send('New data coming', target_rank, tag=101) # Telling rank there is new task to run instead of finishing
        comm.send(charge_c_0, target_rank, tag=0) # Telling rank which charges to compute
        comm.send(charge_c_1, target_rank, tag=1)

        # Receiving results (non-blocking)
        d_V = cp.empty([left_size, left_size], dtype='complex64')
        d_W = cp.empty([right_size, right_size], dtype='complex64')
        d_Lambda = cp.empty(min(left_size, right_size), dtype='float32')
        requests = []
        requests.append(comm.Irecv(d_V, source=target_rank, tag=0))
        requests.append(comm.Irecv(d_W, source=target_rank, tag=1))
        requests.append(comm.Irecv(d_Lambda, source=target_rank, tag=2))
        self.requests[charge_c_0][charge_c_1] = requests
        self.requests_buf[charge_c_0][charge_c_1] = [d_V, d_W, d_Lambda]


    def Check(self, charge_c_0, charge_c_1) -> bool:

        # Determining if slave computational results are ready
        completed = MPI.Request.testall(self.requests[charge_c_0][charge_c_1])
        if not completed[0]:
            # print('Not done')
            return False
        else:
            return True


    def update_rank_status(self):
        new_running_charges_and_rank = []
        for charge_c_0, charge_c_1, rank in self.running_charges_and_rank:
            if self.Check(charge_c_0, charge_c_1):
                self.available_ranks.append(rank)
            else:
                new_running_charges_and_rank.append([charge_c_0, charge_c_1, rank])
        self.running_charges_and_rank = new_running_charges_and_rank
        # print(self.running_l_and_node)


    def NodeProcess(self):

        for target_rank in self.available_ranks:
            comm.send('New data coming', target_rank, tag=100)

        LC = np.empty(self.chi, dtype = 'float32')
        LR = np.empty(self.chi, dtype = 'float32')
        CL = np.empty([self.chi, 2], dtype = 'int32')
        CC = np.empty([self.chi, 2], dtype = 'int32')
        CR = np.empty([self.chi, 2], dtype = 'int32')
        Glc = np.empty([self.chi, self.chi], dtype = 'complex64')  
        Gcr = np.empty([self.chi, self.chi], dtype = 'complex64')
        comm.Recv([LC, MPI.FLOAT], source=0, tag=0)
        comm.Recv([LR, MPI.FLOAT], source=0, tag=1)
        comm.Recv([CL, MPI.INT], source=0, tag=2)
        comm.Recv([CC, MPI.INT], source=0, tag=3)
        comm.Recv([CR, MPI.INT], source=0, tag=4)
        comm.Recv([Glc, MPI.C_FLOAT_COMPLEX], source=0, tag=5)
        comm.Recv([Gcr, MPI.C_FLOAT_COMPLEX], source=0, tag=6)
        r = comm.recv(source=0, tag=7)
        location = comm.recv(source=0, tag=8)
        seed = comm.recv(source=0, tag=9)
        # print('rank: {} got data'.format(rank))#, LC, LR, CL, CC, CR, Glc, Gcr, r, location, seed)

        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.d, CL, CC, CR)
        # Obtain and align data
        cNewL_obj, cNewR_obj, change_charges_C_data, change_idx_C = aligner.cNewL, aligner.cNewR, aligner.change_charges_C, aligner.change_idx_C
        changes = change_idx_C.shape[0]
        change_charges_C = np.zeros([2, (self.d+1)**2], dtype='int32')
        change_charges_C[:, :changes] = change_charges_C_data
        LR_obj, Glc_obj, Gcr_obj = map(aligner.make_data_obj, ['LR','Glc','Gcr'], [True]*3, [LR, Glc, Gcr], [ [0],[0,0],[0,0] ])
        
        requests = []

        # print('node sending data to ranks')
        for target_rank in self.available_ranks:
            requests.append(comm.isend(r, target_rank, tag=0))
            requests.append(comm.isend(seed, target_rank, tag=1))
            requests.append(comm.Isend([change_charges_C, MPI.INT], target_rank, tag=2))
            requests.append(comm.Isend([change_idx_C, MPI.INT], target_rank, tag=3))
            requests.append(comm.isend(changes, target_rank, tag=4))
            requests.append(comm.isend(cNewL_obj, target_rank, tag=5))
            requests.append(comm.isend(cNewR_obj, target_rank, tag=6))
            requests.append(comm.isend(LR_obj, target_rank, tag=7))
            requests.append(comm.isend(Glc_obj, target_rank, tag=8))
            requests.append(comm.isend(Gcr_obj, target_rank, tag=9))
            requests.append(comm.isend(aligner, target_rank, tag=10))
            requests.append(comm.isend(location, target_rank, tag=11))

        MPI.Request.Testall(requests)

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
                # Determining size of buffer for computed resutls
                # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
                min_charge_l_0, max_charge_l_0, min_charge_c_0, max_charge_c_0, min_charge_r_0, max_charge_r_0 = self.charge_range(location, charge_c_0)
                min_charge_l_1, max_charge_l_1, min_charge_c_1, max_charge_c_1, min_charge_r_1, max_charge_r_1 = self.charge_range(location, charge_c_1)
                # Selecting data according to charge bounds
                left_indices = aligner.get_indices(min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 'left', False)
                right_indices = aligner.get_indices(min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1, 'right', False)
                left_size = left_indices.shape[0]
                right_size = right_indices.shape[0]
                # Continue to next charge if no data is selected for this size
                if left_size * right_size == 0:
                    self.requests_buf[charge_c_0][charge_c_1] = None
                    continue

                while len(self.available_ranks) == 0:
                    # print('checking avaiable')
                    self.update_rank_status()
                    time.sleep(0.1)
                target_rank = self.available_ranks.pop(0)
                self.Request(charge_c_0, charge_c_1, left_size, right_size, target_rank)
                # print('finished request')
                self.running_charges_and_rank.append([charge_c_0, charge_c_1, target_rank])

        # print('finished all requests')

        while len(self.available_ranks) != self.num_ranks:
            self.update_rank_status()
            # print('waiting finish layer')
            time.sleep(0.1)

        for target_rank in self.available_ranks:
            comm.send('Node Finished', target_rank, tag=101)

        # Collect calculated data from ranks
        # Initialize
        new_Gamma_L = []
        new_Gamma_R = []
        new_Lambda = np.array([], dtype=float_type)
        new_charge_0 = np.array([], dtype=int_type)
        new_charge_1 = np.array([], dtype=int_type)
        tau_array = [0]
        # Compile results from buffers
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                if self.requests_buf[charge_c_0][charge_c_1] == None:
                    continue
                d_V, d_W, d_Lambda = self.requests_buf[charge_c_0][charge_c_1]
                Lambda = cp.asnumpy(d_Lambda)
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
        d_Gamma0Out = Aligner.make_data_obj('Glc', False, cp.zeros([self.chi, self.chi], dtype = data_type), [0, 0])
        d_Gamma1Out = Aligner.make_data_obj('Gcr', False, cp.zeros([self.chi, self.chi], dtype = data_type), [0, 0])

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        cp.cuda.stream.get_current_stream().synchronize()
        #print('in ', time.time())
        other_start = time.time()
        
        tau = 0
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                if self.requests_buf[charge_c_0][charge_c_1] == None:
                    continue
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

                if len(tau_idx) == 0:
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

        comm.Send([new_charge, MPI.INT], 0, tag=10)
        comm.Send([new_Lambda, MPI.FLOAT], 0, tag=11)
        comm.Send([Gamma0Out, MPI.C_FLOAT_COMPLEX], 0, tag=12)
        comm.Send([Gamma1Out, MPI.C_FLOAT_COMPLEX], 0, tag=13)
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

        # print('Rank {} sent data'.format(rank))
    
    
    def Nodeloop(self):

        status = comm.recv(source=0, tag=100)
        # print('rank: {}, status: {}'.format(rank, status))
        while status != 'Finished':
            self.NodeProcess()
            status = comm.recv(source=0, tag=100)
            # print('rank: {}, status: {}'.format(rank, status))

        for target_rank in self.available_ranks:
            comm.send('Full Finished', target_rank, tag=100)