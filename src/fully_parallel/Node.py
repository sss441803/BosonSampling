import time

import cupy as cp
mempool = cp.get_default_memory_pool()

import numpy as np

from ..mpo_sort import Aligner, charge_range

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


data_type = np.complex64
float_type = np.float32
int_type = np.int32



class NodeWorker:

    def __init__(self, nodes, ranks_per_node, this_node):
        
        self.nodes = nodes
        self.this_node = this_node
        self.available_ranks = [i + 2 for i in range(ranks_per_node * this_node, ranks_per_node * this_node + ranks_per_node - 2)]
        self.num_ranks = ranks_per_node - 2
        self.data_ranks = [node * ranks_per_node for node in range(nodes)]
        

    def ExperimentInit(self, n_modes, local_hilbert_space_dimension, bond_dimension):
        self.n_modes = n_modes
        self.local_hilbert_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = bond_dimension
        
        self.requests = [[[] for _ in range(self.local_hilbert_space_dimension)] for _ in range(self.local_hilbert_space_dimension)]
        self.requests_buf = [[[] for _ in range(self.local_hilbert_space_dimension)] for _ in range(self.local_hilbert_space_dimension)]
        self.running_charges_and_rank = []
        

    def Request(self, charge_c_0, charge_c_1, left_size, right_size, target_rank):
        # Sending information
        comm.send('New data coming', target_rank, tag=101) # Telling rank there is new task to run instead of finishing
        comm.send(charge_c_0, target_rank, tag=0) # Telling rank which charges to compute
        comm.send(charge_c_1, target_rank, tag=1)

        # Receiving results (non-blocking)
        num_eig = min(left_size, right_size)
        V = np.empty(left_size * num_eig, dtype='complex64')
        W = np.empty(num_eig * right_size, dtype='complex64')
        Lambda = np.empty(num_eig, dtype='float32')
        requests = []
        requests.append(comm.Irecv([V, MPI.C_FLOAT_COMPLEX], source=target_rank, tag=0))
        requests.append(comm.Irecv([W, MPI.C_FLOAT_COMPLEX], source=target_rank, tag=1))
        requests.append(comm.Irecv(Lambda, source=target_rank, tag=2))
        self.requests[charge_c_0][charge_c_1] = requests
        self.requests_buf[charge_c_0][charge_c_1] = [V, W, Lambda]


    def Check(self, charge_c_0, charge_c_1) -> bool:

        # Determining if slave computational results are ready
        completed = MPI.Request.testall(self.requests[charge_c_0][charge_c_1])
        if not completed[0]:
            return False
        else:
            return True


    def update_rank_status(self):
        new_all_requests = []
        for requests_info in self.all_requests:
            rank = requests_info[0]
            requests = requests_info[1:]
            completed = MPI.Request.testall(requests)
            if completed[0]:
                self.available_ranks.append(rank)
            else:
                new_all_requests.append(requests_info)
        self.all_requests = new_all_requests

        new_running_charges_and_rank = []
        for charge_c_0, charge_c_1, rank in self.running_charges_and_rank:
            if self.Check(charge_c_0, charge_c_1):
                self.available_ranks.append(rank)
            else:
                new_running_charges_and_rank.append([charge_c_0, charge_c_1, rank])
        self.running_charges_and_rank = new_running_charges_and_rank
        # print(self.running_l_and_node)


    def NodeProcess(self):

        svd_time = 0
        theta_time = 0

        for target_rank in self.available_ranks:
            comm.send('New data coming', target_rank, tag=100)

        # Receiving update parameters from host
        r = comm.recv(source=0, tag=7)
        l = comm.recv(source=0, tag=8)
        seed = comm.recv(source=0, tag=9)
        # Which node to find the data for the l'th site
        data_node_0 = l % self.nodes
        data_node_1 = (l + 1) % self.nodes
        data_node_2 = (l + 2) % self.nodes
        data_rank_0 = self.data_ranks[data_node_0]
        data_rank_1 = self.data_ranks[data_node_1]
        data_rank_2 = self.data_ranks[data_node_2]
        # Receiving data from data nodes
        LC = np.empty(self.bond_dimension, dtype = 'float32')
        LR = np.empty(self.bond_dimension, dtype = 'float32')
        CL = np.empty([self.bond_dimension, 2], dtype = 'int32')
        CC = np.empty([self.bond_dimension, 2], dtype = 'int32')
        CR = np.empty([self.bond_dimension, 2], dtype = 'int32')
        self.Glc = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')  
        self.Gcr = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')
        comm.Recv([LC, MPI.FLOAT], source=data_rank_0, tag=0)
        comm.Recv([LR, MPI.FLOAT], source=data_rank_1, tag=1)
        comm.Recv([CL, MPI.INT], source=data_rank_0, tag=2)
        comm.Recv([CC, MPI.INT], source=data_rank_1, tag=3)
        comm.Recv([CR, MPI.INT], source=data_rank_2, tag=4)
        comm.Recv([self.Glc, MPI.C_FLOAT_COMPLEX], source=data_rank_0, tag=5)
        comm.Recv([self.Gcr, MPI.C_FLOAT_COMPLEX], source=data_rank_1, tag=6)

        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.local_hilbert_space_dimension, CL, CC, CR)
        # Obtain and align data
        cNewL_obj, cNewR_obj, change_charges_C_data, change_idx_C = aligner.cNewL, aligner.cNewR, aligner.change_charges_C, aligner.change_idx_C

        changes = change_idx_C.shape[0]
        change_charges_C = np.zeros([2, (self.local_hilbert_space_dimension+1)**2], dtype='int32')
        change_charges_C[:, :changes] = change_charges_C_data

        self.all_requests = []
        for target_rank in self.available_ranks:
            requests_info = [target_rank]
            requests_info.append(comm.isend(r, target_rank, tag=0))
            requests_info.append(comm.isend(seed, target_rank, tag=1))
            requests_info.append(comm.Isend([change_charges_C, MPI.INT], target_rank, tag=2))
            requests_info.append(comm.Isend([change_idx_C, MPI.INT], target_rank, tag=3))
            requests_info.append(comm.isend(changes, target_rank, tag=4))
            requests_info.append(comm.isend(cNewL_obj, target_rank, tag=5))
            requests_info.append(comm.isend(cNewR_obj, target_rank, tag=6))
            requests_info.append(comm.Isend([LR, MPI.FLOAT], target_rank, tag=7))
            requests_info.append(comm.Isend([self.Glc, MPI.C_FLOAT_COMPLEX], target_rank, tag=8))
            requests_info.append(comm.Isend([self.Gcr, MPI.C_FLOAT_COMPLEX], target_rank, tag=9))
            requests_info.append(comm.isend(aligner, target_rank, tag=10))
            self.all_requests.append(requests_info)
        self.available_ranks = []

        # Storage of generated data
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

        svd_complexity_approx = []
        result_sizes = []
        charges = []

        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                # Determining size of buffer for computed resutls
                # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
                min_charge_l_0, max_charge_l_0, _,  _, min_charge_r_0, max_charge_r_0 = charge_range(self.local_hilbert_space_dimension, charge_c_0)
                min_charge_l_1, max_charge_l_1, _, _, min_charge_r_1, max_charge_r_1 = charge_range(self.local_hilbert_space_dimension, charge_c_1)
                # Selecting data according to charge bounds
                left_indices = aligner.get_indices(min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 'left', False)
                right_indices = aligner.get_indices(min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1, 'right', False)
                left_size = left_indices.shape[0]
                right_size = right_indices.shape[0]
                # Continue to next charge if no data is selected for this size
                if left_size * right_size == 0:
                    self.requests_buf[charge_c_0][charge_c_1] = None
                    continue
                charges.append([charge_c_0, charge_c_1])
                result_sizes.append([left_size, right_size])
                svd_complexity_approx.append(- left_size * right_size * min(left_size, right_size))

        sizes = [[None for _ in range(self.local_hilbert_space_dimension)] for _ in range(self.local_hilbert_space_dimension)]
        complexity_ordered_charge_ids = np.argsort(svd_complexity_approx) # Indices that sort in descending order. Largest first so that they are first ran
        for charge_id in complexity_ordered_charge_ids:
            left_size, right_size = result_sizes[charge_id]
            charge_c_0, charge_c_1 = charges[charge_id]
            while len(self.available_ranks) == 0:
                self.update_rank_status()
                time.sleep(0.01)
            target_rank = self.available_ranks.pop(0)
            self.Request(charge_c_0, charge_c_1, left_size, right_size, target_rank)
            sizes[charge_c_0][charge_c_1] = [left_size, right_size]
            self.running_charges_and_rank.append([charge_c_0, charge_c_1, target_rank])

        while len(self.available_ranks) != self.num_ranks:
            self.update_rank_status()
            time.sleep(0.01)

        self.Glc = None
        self.Gcr = None
        del self.Glc, self.Gcr

        for target_rank in self.available_ranks:
            comm.send('Node Finished', target_rank, tag=101)
            theta_time += comm.recv(source=target_rank, tag=3)
            svd_time += comm.recv(source=target_rank, tag=4)

        # Collect calculated data from ranks
        # Initialize
        new_Lambda = np.array([], dtype=float_type)
        new_charge_0 = np.array([], dtype=int_type)
        new_charge_1 = np.array([], dtype=int_type)
        tau_array = [0]
        # Compile results from buffers
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                if self.requests_buf[charge_c_0][charge_c_1] == None:
                    continue
                Lambda = self.requests_buf[charge_c_0][charge_c_1][2]
                new_Lambda = np.append(new_Lambda, Lambda)
                new_charge_0 = np.append(new_charge_0, np.repeat(np.array(charge_c_0, dtype=int_type), len(Lambda)))
                new_charge_1 = np.append(new_charge_1, np.repeat(np.array(charge_c_1, dtype=int_type), len(Lambda)))
                tau_array.append(len(Lambda))
        # Number of singular values to save
        num_lambda = int(min(new_Lambda.shape[0], self.bond_dimension))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx_select = np.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
        else:
            idx_select = np.array([], dtype=int_type)

        # Select charges that corresponds to the largest num_lambda singular values
        new_charge_0 = new_charge_0[idx_select]
        new_charge_1 = new_charge_1[idx_select]
        # Sort the new charges
        idx_sort = np.lexsort((new_charge_1, new_charge_0)) # Indices that will sort the new charges
        new_charge_0 = new_charge_0[idx_sort]
        new_charge_1 = new_charge_1[idx_sort]
        new_charge = self.local_hilbert_space_dimension * np.ones([self.bond_dimension, 2], dtype='int32')
        new_charge[:num_lambda, 0] = new_charge_0
        new_charge[:num_lambda, 1] = new_charge_1
        # Selecting and sorting Lambda
        new_Lambda = new_Lambda[idx_select]
        new_Lambda = new_Lambda[idx_sort]
        # Sending Lambda and charge
        comm.Send([new_charge, MPI.INT], data_rank_1, tag=10)
        comm.Send([new_Lambda, MPI.FLOAT], data_rank_0, tag=11)
        
        # Initialize selected and sorted Gamma outputs
        Gamma0Out = Aligner.make_data_obj('Glc', False, cp.zeros([1, 1], dtype = data_type), [0, 0])
        Gamma0OutData = cp.zeros([self.bond_dimension, self.bond_dimension], dtype = 'complex64')

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        tau = 0
        info_list = [[None for _ in range(self.local_hilbert_space_dimension)] for _ in range(self.local_hilbert_space_dimension)]
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                if self.requests_buf[charge_c_0][charge_c_1] == None:
                    continue
                
                # Finding bond indices (tau_idx) that are in the largest num_lambda singular values and for center charge tau.
                # idx_select[indices] = tau_idx
                tau_idx, indices, _ = np.intersect1d(idx_select, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
                tau += 1 # This line MUST be before the continue statement
                if len(tau_idx) == 0:
                    continue

                # Selecting gamma that will be modified. Modifying gamma will modify Gamma (because they are pointers).
                min_charge_l_0, max_charge_l_0, _, _, min_charge_r_0, max_charge_r_0 = charge_range(self.local_hilbert_space_dimension, charge_c_0)
                min_charge_l_1, max_charge_l_1, _, _, min_charge_r_1, max_charge_r_1 = charge_range(self.local_hilbert_space_dimension, charge_c_1)
                idx_gamma0_0, idx_gamma0_1 = aligner.get_select_index(Gamma0Out, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 0, self.local_hilbert_space_dimension, 0, self.local_hilbert_space_dimension)
                info_list[charge_c_0][charge_c_1] = [min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1, tau_idx, indices]
                
                # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
                left_size, right_size = sizes[charge_c_0][charge_c_1]
                num_eig = min(left_size, right_size)
                V = cp.array(self.requests_buf[charge_c_0][charge_c_1][0]).reshape(left_size, num_eig)
                V = V[:, tau_idx - tau_idx[0]]

                # Calculating output gamma
                # Left
                Gamma0OutData[idx_gamma0_0.reshape(-1,1), idx_gamma0_1[indices].reshape(1,-1)] = V
                del V
                self.requests_buf[charge_c_0][charge_c_1][0] = None
        
        Gamma0OutData[:, :num_lambda] = Gamma0OutData[:, idx_sort]

        comm.Send([cp.asnumpy(Gamma0OutData), MPI.C_FLOAT_COMPLEX], data_rank_0, tag=12)
        del Gamma0Out, Gamma0OutData
        mempool.free_all_blocks()

        # Initialize selected and sorted Gamma outputs
        Gamma1Out = Aligner.make_data_obj('Gcr', False, cp.zeros([1, 1], dtype = data_type), [0, 0])
        Gamma1OutData = cp.zeros([self.bond_dimension, self.bond_dimension], dtype = 'complex64')

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        tau = 0
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
            for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
                if self.requests_buf[charge_c_0][charge_c_1] == None:
                    continue
                
                if info_list[charge_c_0][charge_c_1] == None:
                    continue

                min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1, tau_idx, indices = info_list[charge_c_0][charge_c_1]
                idx_gamma1_0, idx_gamma1_1 = aligner.get_select_index(Gamma1Out, 0, self.local_hilbert_space_dimension, 0, self.local_hilbert_space_dimension, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
                # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
                left_size, right_size = sizes[charge_c_0][charge_c_1]
                num_eig = min(left_size, right_size)
                W = cp.array(self.requests_buf[charge_c_0][charge_c_1][1]).reshape(num_eig, right_size)
                W = W[tau_idx - tau_idx[0]]

                # Calculating output gamma
                # Right
                Gamma1OutData[idx_gamma1_0[indices].reshape(-1,1), idx_gamma1_1.reshape(1,-1)] = W
                del W
                self.requests_buf[charge_c_0][charge_c_1][1] = None

        # Sorting Gamma
        Gamma1OutData[:num_lambda] = Gamma1OutData[idx_sort]

        comm.Send([cp.asnumpy(Gamma1OutData), MPI.C_FLOAT_COMPLEX], data_rank_1, tag=13)
        del Gamma1Out, Gamma1OutData
        mempool.free_all_blocks()
        comm.send(svd_time, 0, tag=14)
        comm.send(theta_time, 0, tag=15)
    
    
    def Simulate(self):

        status = comm.recv(source=0, tag=100)
        while status != 'Finished':
            self.NodeProcess()
            status = comm.recv(source=0, tag=100)

        for target_rank in self.available_ranks:
            comm.send('Full Finished', target_rank, tag=100)