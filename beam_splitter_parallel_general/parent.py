'''Full simulation code containing the Device method (cupy, unified update)'''
import time
import numpy as np
from scipy.stats import rv_continuous
from qutip import squeeze, thermal_dm
from mpi4py import MPI

comm = MPI.COMM_WORLD


class ParentMPO:
    def __init__(self, num_gpu_ranks, input_state_type, n_modes, n_input_states, post_selected_photon_number,local_hilbert_space_dimension, bond_dimension, parameters, errtol = 10 ** (-6)):
        self.num_gpu_ranks = num_gpu_ranks
        self.n_modes = n_modes
        self.n_input_states = n_input_states
        self.post_selected_photon_number = post_selected_photon_number
        self.local_hilber_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = bond_dimension
        self.errtol = errtol
        self.TotalProbPar = np.zeros([self.n_modes + 1], dtype = 'float32')
        self.SingleProbPar = np.zeros([self.n_modes + 1], dtype = 'float32')
        self.EEPar = np.zeros([self.n_modes - 1, self.n_modes + 1], dtype = 'float32')      
        self.REPar = np.zeros([self.n_modes - 1, self.n_modes + 1, 5], dtype = 'float32')
        self.reflectivity = np.empty([self.n_modes, self.n_modes // 2])
        self.seeds = np.empty([self.n_modes, self.n_modes // 2], dtype=int)
        self.normalization = None
        self.update_time = 0
        self.U_time = 0
        self.theta_time = 0
        self.svd_time = 0
        self.ee_prob_cal_time = 0
        self.align_init_time = 0
        self.align_info_time = 0
        self.index_time = 0
        self.copy_time = 0
        self.align_time = 0
        self.array_cpu_time = 0
        self.before_loop_other_time = 0
        self.segment1_time = 0
        self.segment2_time = 0
        self.segment3_time = 0
        self.comm_time = 0
        self.largest_C = 0
        self.largest_T = 0

        self.requests = [None for _ in range(self.n_modes - 1)]
        self.requests_buf = [None for _ in range(self.n_modes - 1)]
        # self.available_ranks = [i + i//4 + 1 for i in range(self.num_ranks)]
        self.available_ranks = [i for i in range(1, self.num_gpu_ranks+1)]
        self.running_l_and_rank = []

        self.input_state_type = input_state_type
        # Loading input state type specific experiment configuration parameters
        if input_state_type == 'Gaussian':
            # beta: output photon number multiplication factor
            self.beta, self.loss, self.squeeze_parameter = parameters['beta'], parameters['loss'], parameters['r']
        elif input_state_type == 'Single photon':
            self.beta, self.loss = parameters['beta'], parameters['loss']
        elif input_state_type == 'Thermal':
            raise NotImplementedError
        else:
            raise ValueError('input_state_type {} not supported.'.format(input_state_type))

    def MPOInitialization(self):
        
        # Initialize the reflectivities of beam splitters to form a global Haar random array
        self.UpdateReflectivity()
        
        initial_bond_dimension = self.local_hilber_space_dimension ** 2
        self.Lambda_edge = np.ones(initial_bond_dimension, dtype = 'float32') # edge lambda (for first and last site) don't exists and are ones
        self.Lambda = np.zeros([initial_bond_dimension, self.n_modes - 1], dtype = 'float32')
        self.Gamma = np.zeros([initial_bond_dimension, initial_bond_dimension, self.n_modes], dtype = 'complex64')  
        self.charge = self.local_hilber_space_dimension * np.ones([initial_bond_dimension, self.n_modes + 1, 2], dtype = 'int32')
        self.charge[0] = 0
        
        # Initializing single input state density matrix
        if self.input_state_type == 'Gaussian':
            am = (1 - self.loss) * np.exp(- 2 * self.squeeze_parameter) + self.loss;
            ap = (1 - self.loss) * np.exp(2 * self.squeeze_parameter) + self.loss;
            s = 1 / 4 * np.log(ap / am);
            n_th = 1 / 2 * (np.sqrt(am * ap) - 1);
            nn = 40;
            single_input_density_matrix = (squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()).full()[:(self.local_hilber_space_dimension + 1), :(self.local_hilber_space_dimension + 1)];
        elif self.input_state_type == 'Single photon':
            single_input_density_matrix = np.zeros([self.local_hilber_space_dimension, self.local_hilber_space_dimension], dtype = 'complex64')
            single_input_density_matrix[0, 0] = self.loss
            single_input_density_matrix[1, 1] = 1 - self.loss
        elif self.input_state_type == 'Thermal':
            raise NotImplementedError
        else:
            raise ValueError('input_state_type {} not supported.'.format(self.input_state_type))

        # Filling MPO
        if self.post_selected_photon_number == None:
            for i in range(self.local_hilber_space_dimension):
                self.charge[i, 0, 0] = i
                self.charge[i, 0, 1] = i
            updated_bonds = np.array([bond for bond in range(self.local_hilber_space_dimension)])
        else:
            self.charge[0, 0, 0] = self.post_selected_photon_number
            self.charge[0, 0, 1] = self.post_selected_photon_number
            updated_bonds = np.array([0])

        for i in range(self.n_input_states - 1):
            bonds_updated = np.zeros(self.local_hilber_space_dimension**2)
            for j in updated_bonds:
                if self.charge[j, i, 0] == self.local_hilber_space_dimension:
                    c1 = 0
                else:
                    c1 = self.charge[j, i, 0]
                for ch_diff1 in range(c1, -1, -1):
                    if self.charge[j, i, 1] == self.local_hilber_space_dimension:
                        c2 = 0
                    else:
                        c2 = self.charge[j, i, 1]
                    for ch_diff2 in range(c2, -1, -1):
                        if np.abs(single_input_density_matrix[ch_diff1, ch_diff2]) <= self.errtol:
                            continue
                        self.Gamma[j, (c1 - ch_diff1) * self.local_hilber_space_dimension + c2 - ch_diff2, i] = single_input_density_matrix[ch_diff1, ch_diff2]
                        self.charge[(c1 - ch_diff1) * self.local_hilber_space_dimension + c2 - ch_diff2, i + 1, 0] = c1 - ch_diff1
                        self.charge[(c1 - ch_diff1) * self.local_hilber_space_dimension + c2 - ch_diff2, i + 1, 1] = c2 - ch_diff2
                        bonds_updated[(c1 - ch_diff1) * self.local_hilber_space_dimension + c2 - ch_diff2] = 1
            updated_bonds = np.where(bonds_updated == 1)[0]
            self.Lambda[updated_bonds, i] = 1

        for j in updated_bonds:
            if self.charge[j, self.n_input_states - 1, 0] == self.local_hilber_space_dimension:
                c0 = 0
            else:
                c0 = self.charge[j, self.n_input_states - 1, 0]
            if self.charge[j, self.n_input_states - 1, 1] == self.local_hilber_space_dimension:
                c1 = 0
            else:
                c1 = self.charge[j, self.n_input_states - 1, 1]
            self.Gamma[j, 0, self.n_input_states - 1] = single_input_density_matrix[c0, c1]
        
        for i in range(self.n_input_states - 1, self.n_modes - 1):
            self.Lambda[0, i] = 1
            self.charge[0, i + 1, 0] = 0
            self.charge[0, i + 1, 1] = 0
        
        for i in range(self.n_input_states):
            self.Gamma[:, :, i] = np.multiply(self.Gamma[:, :, i], self.Lambda[:, i].reshape(1, -1))

        for i in range(self.n_input_states, self.n_modes):
            self.Gamma[0, 0, i] = 1

        self.Gamma = np.transpose(self.Gamma, (2, 0, 1))
        self.Lambda = np.transpose(self.Lambda, (1, 0))
        self.charge = np.transpose(self.charge, (1, 0, 2))

        # Sorting bonds based on bond charges
        for i in range(self.n_modes + 1):
            idx = np.lexsort((self.charge[i, :, 1], self.charge[i, :, 0]))
            self.charge[i] = self.charge[i, idx]
            if i > 0:
                self.Gamma[i - 1] = self.Gamma[i - 1][:, idx]
                if i < self.n_modes:
                    self.Lambda[i - 1] = self.Lambda[i - 1, idx]
            if i < self.n_modes:
                self.Gamma[i] = self.Gamma[i, idx]
        
        self.Gamma = np.ascontiguousarray(self.Gamma)
        self.Lambda = np.ascontiguousarray(self.Lambda)
        self.charge = np.ascontiguousarray(self.charge)
        self.normalization = np.real(TotalProbFromMPO(self.n_modes, self.local_hilber_space_dimension, self.Gamma, self.Lambda, self.charge))
        self.TotalProbPar[0] = self.normalization
        if not self.normalization > 0:
            print('Configuration yields invalid probabilities. Exiting')
            return False
        # print(self.Lambda, self.charge)
        self.canonicalize = True
        print('Canonicalization update')
        for l in range(self.n_modes - 2, -1, -1):
            self.ParentRequest(0, l, 0, 1)
            done = False
            while not done:
                done = self.ParentCheck(l)
                time.sleep(0.01)
        self.canonicalize = False
        # print(self.Lambda, self.charge)

        self.normalization = np.real(TotalProbFromMPO(self.n_modes, self.local_hilber_space_dimension, self.Gamma, self.Lambda, self.charge))
        print('Total probability normalization factor: ', self.normalization)
        if not self.normalization > 0:
            print('Configuration yields invalid probabilities. Exiting')
            return False

        charge_temp = np.copy(self.charge)
        Lambda_temp = np.copy(self.Lambda)
        Gamma_temp = np.copy(self.Gamma)
        self.charge = self.local_hilber_space_dimension * np.ones([self.n_modes + 1, self.bond_dimension, 2], dtype = 'int32')
        self.Lambda = np.zeros([self.n_modes - 1, self.bond_dimension], dtype = 'float32')
        self.Lambda_edge = np.ones(self.bond_dimension, dtype = 'float32')
        self.Gamma = np.zeros([self.n_modes, self.bond_dimension, self.bond_dimension], dtype = 'complex64')
        self.Gamma[:, :initial_bond_dimension, :initial_bond_dimension] = Gamma_temp
        self.Lambda[:, :initial_bond_dimension] = Lambda_temp
        self.charge[:, :initial_bond_dimension] = charge_temp

        return True


    # MPO update of a two-qudit gate        
    def ParentRequest(self, seed, mode, reflectivity, target_rank):

        # Determining the location of the two qubit gate
        LC = self.Lambda[mode,:]
        if mode == 0:
            location = 'left'
            LR = self.Lambda[mode+1,:]
        elif mode == self.n_modes - 2:
            location = 'right'
            LR = self.Lambda_edge[:]
        else:
            location = 'center'
            LR = self.Lambda[mode+1,:]
        
        Glc = self.Gamma[mode,:]
        Gcr = self.Gamma[mode+1,:]

        # charge of corresponding index (bond charge left/center/right)
        CL = self.charge[mode]
        CC = self.charge[mode+1]
        CR = self.charge[mode+2]

        # print('Sending info to rank ', target_rank)

        # print('sending data to ', target_rank)
        comm.isend('New data coming', target_rank, tag=100)
        comm.Isend([LC, MPI.FLOAT], target_rank, tag=0)
        comm.Isend([LR, MPI.FLOAT], target_rank, tag=1)
        comm.Isend([CL, MPI.INT], target_rank, tag=2)
        comm.Isend([CC, MPI.INT], target_rank, tag=3)
        comm.Isend([CR, MPI.INT], target_rank, tag=4)
        comm.Isend([Glc, MPI.C_FLOAT_COMPLEX], target_rank, tag=5)
        comm.Isend([Gcr, MPI.C_FLOAT_COMPLEX], target_rank, tag=6)
        comm.isend(reflectivity, target_rank, tag=7)
        comm.isend(seed, target_rank, tag=8)

        # print('Sent info to rank ', target_rank)
        if self.canonicalize:
            bond_dimension = self.local_hilber_space_dimension ** 2
        else:
            bond_dimension = self.bond_dimension
        new_charge = self.local_hilber_space_dimension * np.ones([bond_dimension, 2], dtype='int32')
        new_Lambda = np.zeros(bond_dimension, dtype='float32')
        Gamma0Out = np.zeros([bond_dimension, bond_dimension], dtype='complex64')
        Gamma1Out = np.zeros([bond_dimension, bond_dimension], dtype='complex64')
        new_charge_req = comm.Irecv(new_charge, source=target_rank, tag=10)
        new_Lambda_req = comm.Irecv(new_Lambda, source=target_rank, tag=11)
        Gamma0Out_req = comm.Irecv(Gamma0Out, source=target_rank, tag=12)
        Gamma1Out_req = comm.Irecv(Gamma1Out, source=target_rank, tag=13)

        # print('Requested info from rank ', target_rank)
        U_time_req = comm.irecv(source=target_rank, tag=14)
        svd_time_req = comm.irecv(source=target_rank, tag=15)
        theta_time_req = comm.irecv(source=target_rank, tag=16)
        align_init_time_req = comm.irecv(source=target_rank, tag=17)
        align_info_time_req = comm.irecv(source=target_rank, tag=18)
        index_time_req = comm.irecv(source=target_rank, tag=19)
        copy_time_req = comm.irecv(source=target_rank, tag=20)
        align_time_req = comm.irecv(source=target_rank, tag=21)
        before_loop_other_time_req = comm.irecv(source=target_rank, tag=22)
        segment1_time_req = comm.irecv(source=target_rank, tag=23)
        segment2_time_req = comm.irecv(source=target_rank, tag=24)
        segment3_time_req = comm.irecv(source=target_rank, tag=25)
        comm_time_req = comm.irecv(source=target_rank, tag=26)
        self.requests[mode] = [new_charge_req, new_Lambda_req, Gamma0Out_req, Gamma1Out_req, U_time_req, svd_time_req, theta_time_req, align_init_time_req, align_info_time_req, index_time_req, copy_time_req, align_time_req, before_loop_other_time_req, segment1_time_req, segment2_time_req, segment3_time_req, comm_time_req]
        self.requests_buf[mode] = [new_charge, new_Lambda, Gamma0Out, Gamma1Out]
        # print('Parent request over with rank ', target_rank)


    def ParentCheck(self, l) -> bool:

        # Determining if slave computational results are ready
        completed = MPI.Request.testall(self.requests[l])
        if not completed[0]:
            # print('Not done')
            return False

        # Loading slave computational results
        new_charge, new_Lambda, Gamma0Out, Gamma1Out = self.requests_buf[l]
        _, _, _, _, U_time, svd_time, theta_time, align_init_time, align_info_time, index_time, copy_time, align_time, before_loop_other_time, segment1_time, segment2_time, segment3_time, comm_time = completed[1]
        self.U_time += U_time
        self.svd_time += svd_time
        self.theta_time += theta_time
        self.align_init_time += align_init_time
        self.align_info_time += align_info_time
        self.index_time += index_time
        self.copy_time += copy_time
        self.align_time += align_time
        self.before_loop_other_time += before_loop_other_time
        self.segment1_time += segment1_time
        self.segment2_time += segment2_time
        self.segment3_time += segment3_time
        self.comm_time += comm_time

        # Update charges (modifying CC modifies self.dcharge by pointer)
        start = time.time()
        self.charge[l + 1] = new_charge
        self.Lambda[l] = new_Lambda
        if l == self.n_modes - 2:
            self.Gamma[self.n_modes - 2, :, :min(self.bond_dimension, self.local_hilber_space_dimension ** 2)] = Gamma0Out[:, :min(self.bond_dimension, self.local_hilber_space_dimension ** 2)]
            self.Gamma[self.n_modes - 1, :min(self.bond_dimension, self.local_hilber_space_dimension ** 2), 0] = Gamma1Out[:min(self.bond_dimension, self.local_hilber_space_dimension ** 2), 0]
        else:
            self.Gamma[l, :, :] = Gamma0Out
            self.Gamma[l + 1, :, :] = Gamma1Out
        self.array_cpu_time += time.time() - start

        return True


    def UpdateReflectivity(self):
        np.random.seed(1)
        for k in range(self.n_modes - 1):
            # print('k, ', k)
            if k < self.n_modes / 2:
                temp1 = 2 * k + 1
                temp2 = 2
                l = 2 * k
                i = 0
                while l >= 0:
                    if temp1 > 0:
                        T = my_cv.rvs(2 * k + 2, temp1)
                        temp1 -= 2
                    else:
                        T = my_cv.rvs(2 * k + 2, temp2)
                        temp2 += 2
                    self.reflectivity[i, k-(i+1)//2] = np.sqrt(1 - T)
                    seed = np.random.randint(0, 13579)
                    self.seeds[i, k-(i+1)//2] = seed
                    # np.random.seed(seed)
                    # print(seed)
                    l -= 1
                    i += 1
            else:
                temp1 = 2 * self.n_modes - (2 * k + 3)
                temp2 = 2
                l = self.n_modes - 2
                first_layer = 2 * k - self.n_modes + 2
                for i in range(2 * self.n_modes - 2 * k - 2):
                    if temp1 >= 0:
                        T = my_cv.rvs(2 * self.n_modes - (2 * k + 1), temp1)
                        temp1 -= 2
                    else:
                        T = my_cv.rvs(2 * self.n_modes - (2 * k + 1), temp2)
                        temp2 += 2
                    self.reflectivity[first_layer + i, self.n_modes//2-1-(i+1)//2] = np.sqrt(1 - T)
                    seed = np.random.randint(0, 13579)
                    self.seeds[first_layer + i, self.n_modes//2-1-(i+1)//2] = seed
                    # np.random.seed(seed)
                    # print(seed)
                    l -= 1    


    def update_rank_status(self):
        new_running_l_and_rank = []
        for l, rank in self.running_l_and_rank:
            if self.ParentCheck(l):
                self.available_ranks.append(rank)
            else:
                new_running_l_and_rank.append([l, rank])
        self.running_l_and_rank = new_running_l_and_rank
        # print('running l and rank: ', self.running_l_and_rank)

    def LayerUpdate(self, k):
        # print('updating layer ', k)
        start = time.time()
        # self.U_time = 0
        # self.svd_time = 0
        # self.theta_time = 0
        # self.align_init_time = 0
        # self.align_info_time = 0
        # self.index_time = 0
        # self.copy_time = 0
        # self.align_time = 0
        # self.before_loop_other_time = 0
        # self.segment1_time = 0
        # self.segment2_time = 0
        # self.segment3_time = 0
        # self.comm_time = 0
        for i, l in enumerate(range(k % 2, self.n_modes - 1, 2)):
            reflectivity = self.reflectivity[k, i]
            seed = self.seeds[k, i]
            # print('finished reflectivity')
            # print(self.available_ranks)
            while len(self.available_ranks) == 0:
                # print('checking avaiable')
                self.update_rank_status()
                time.sleep(0.01)
            target_rank = self.available_ranks.pop(0)
            self.ParentRequest(seed, l, reflectivity, target_rank)
            # print('finished request')
            self.running_l_and_rank.append([l, target_rank])
        while len(self.available_ranks) != self.num_gpu_ranks:
            self.update_rank_status()
            # print('waiting finish layer')
            time.sleep(0.01)
        self.update_time += time.time() - start


    def FullUpdate(self):

        initialization_successful = self.MPOInitialization()

        if initialization_successful:
            self.EEPar[:, 0] = self.MPOEntanglementEntropy()
            # alpha_array = [0.5, 0.6, 0.7, 0.8, 0.9]
            # for i in range(5):
                # self.REPar[:, 0, i] = self.MPORenyiEntropy(alpha_array[i])
            full_start = time.time()

            for k in range(self.n_modes):
                self.LayerUpdate(k)
                start = time.time()
                self.TotalProbPar[k+1] = TotalProbFromMPO(self.n_modes, self.local_hilber_space_dimension, self.Gamma, self.Lambda, self.charge)
                self.EEPar[:, k+1] = self.MPOEntanglementEntropy()
                prob = self.TotalProbPar[np.where(self.TotalProbPar > 0)[0]]
                if np.max(prob)/np.min(prob) - 1 > 0.1:
                    print('Accuracy too low. Failed.')
                    break
                max_ee_idx = np.argmax(self.EEPar)
                if k - max_ee_idx % self.EEPar.shape[1] > 20:
                    print('EE did not increase for 20 epochs. Stopping')
                    break
                self.ee_prob_cal_time += time.time() - start
                # for i in range(5):
                    # self.REPar[:, k + 1, i] = self.MPORenyiEntropy(alpha_array[i])
                '''Initialial total time is much higher than simulation time due to initialization of cuda context.'''
                print("m: {:.2f}. Total time: {:.2f}. Update time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. EE_Prob_cal_time: {:.2f}. Align time: {:.2f}. Array cpu time: {:.2f}. Comm time: {:.2f}".format(self.n_input_states, time.time()-full_start, self.update_time, self.theta_time, self.svd_time, self.ee_prob_cal_time, self.align_time, self.array_cpu_time, self.comm_time))

            while len(self.available_ranks) != self.num_gpu_ranks:
                self.update_rank_status()
                time.sleep(0.01)

        for target_rank in self.available_ranks:
            comm.send('Finished', target_rank, tag=100)

        return self.TotalProbPar, self.EEPar, self.REPar
    
    
    def MPOEntanglementEntropy(self):      
        Output = np.zeros([self.n_modes - 1])
        sq_lambda = np.copy(self.Lambda ** 2)
        for i in range(self.n_modes - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i]))
        return Output

    def MPORenyiEntropy(self, alpha = 0.5):
        Output = np.zeros([self.n_modes - 1])
        sq_lambda = np.copy(self.Lambda ** 2)
        for i in range(self.n_modes - 1):
            Output[i] += RenyiFromColumn(ColumnSumToOne(sq_lambda[i]), alpha)
        return Output

    def getProb(self, outcome):
        tot_ch = np.sum(outcome)
        charge = [tot_ch]
        for i in range(len(outcome) - 1):
            charge.append(tot_ch - outcome[i])
            tot_ch = tot_ch - outcome[i]

        R = self.Gamma[self.n_modes - 1, :, 0]
        RTemp = np.copy(R);            
            
        for k in range(self.n_modes - 1):
            idx = np.array([], dtype = 'int32')
            idx = np.append(idx,np.intersect1d(np.nonzero(self.charge[self.n_modes - 1 - k, :, 0] == charge[self.n_modes - 1 - k]), np.intersect1d(np.nonzero(self.charge[self.n_modes - 1 - k, :, 1] == charge[self.n_modes - 1 - k]), np.nonzero(self.Lambda[self.n_modes - 1 - k - 1, :] > 0))))
            R = np.matmul(self.Gamma[self.n_modes - 1 - k - 1, :, idx], RTemp[idx].reshape(-1))
            RTemp = np.copy(R)
        idx = np.array([], dtype = 'int32')
        idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[0, :, 0] == np.sum(outcome)), np.nonzero(self.charge[0, :, 1] == np.sum(outcome))))
        return np.abs(np.sum(RTemp[idx]))




class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1)
my_cv = my_pdf(a = 0, b = 1, name='my_pdf')

def TotalProbFromMPO(n, d, Gamma, Lambda, charge):
    R = Gamma[n - 1, :, 0]
    RTemp = np.copy(R)
    for k in range(n - 2):
        idx = np.array([], dtype = 'int32')
        for ch in range(d):
            idx = np.append(idx, np.intersect1d(np.nonzero(charge[n - 1 - k, :, 0] == ch), np.intersect1d(np.nonzero(charge[n - 1 - k, :, 1] == ch), np.nonzero(Lambda[n - 1 - k - 1] > 0))))
        R = np.matmul(Gamma[n - 1 - k - 1, :, idx].T, RTemp[idx].reshape(-1))
        RTemp = np.copy(R)
    idx = np.array([], dtype = 'int32')
    for ch in range(d):
        idx = np.append(idx, np.intersect1d(np.nonzero(charge[1, :, 0] == ch), np.intersect1d(np.nonzero(charge[1, :, 1] == ch), np.nonzero(Lambda[0, :] > 0))))
    res = np.matmul(Gamma[0, :, idx].T, RTemp[idx].reshape(-1))
    tot_prob = np.sum(res)
    print('Probability: ', np.real(tot_prob))
    # if self.normalization != None:
    #     if tot_prob/self.normalization > 1.05 or tot_prob/self.normalization < 0.95:
    #         quit()
    return tot_prob

def EntropyFromColumn(InputColumn):
    Output = -np.nansum(InputColumn * np.log2(InputColumn))
    return Output

def RenyiFromColumn(InputColumn, alpha):
    Output = np.log2(np.nansum(InputColumn ** alpha)) / (1 - alpha)
    return Output

def ColumnSumToOne(InputColumn):
    return InputColumn / np.sum(InputColumn)