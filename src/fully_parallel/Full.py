'''Full simulation code containing the Device method (cupy, unified update)'''
import time

import numpy as np
np.set_printoptions(precision=3)

from mpi4py import MPI
comm = MPI.COMM_WORLD

from ..interferometer import ReflectivityAndSeeds
from ..MPO import Initialize
from ..summary import Probability, EntropyFromColumn, ColumnSumToOne


class FullWorker:
    def __init__(self, nodes, ranks_per_node):

        self.nodes = nodes
        self.this_node = 0
        self.available_nodes = [node for node in range(nodes)]
        self.compute_ranks = [node * ranks_per_node + 1 for node in range(nodes)]
        self.data_ranks = [node * ranks_per_node for node in range(nodes)]

        
    def ExperimentInit(self, input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, bond_dimension, parameters, seed=None):

        self.n_modes = n_modes
        self.n_input_states = n_input_states
        self.local_hilbert_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = bond_dimension
        self.prob = np.zeros([self.n_modes + 1], dtype = 'float32')
        self.EE = np.zeros([self.n_modes - 1, self.n_modes + 1], dtype = 'float32')      
        self.REPar = np.zeros([self.n_modes - 1, self.n_modes + 1, 5], dtype = 'float32')
        self.normalization = None
        self.update_time = 0
        self.theta_time = 0
        self.svd_time = 0
        self.ee_prob_cal_time = 0
        
        self.meta_data_requests = [None for _ in range(self.n_modes - 1)]
        self.requests_buf = [[None, None, None, None] for _ in range(self.n_modes - 1)]
        self.running_l_and_node = []
        
        # Initialize the reflectivities of beam splitters to form a global Haar random array
        self.reflectivity, self.seeds = ReflectivityAndSeeds(n_modes, seed)
        # Initialize the MPO
        self.initialization_successful = self.MPOInit(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters)


    def MPOInit(self, input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters):

        Gamma, Lambda, charge = Initialize(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters)

        self.prob[0] = np.real(Probability(n_modes, local_hilbert_space_dimension, Gamma, Lambda, charge))

        charge_temp = np.copy(charge)
        Lambda_temp = np.copy(Lambda)
        Gamma_temp = np.copy(Gamma)
        charge = self.local_hilbert_space_dimension * np.ones([self.n_modes + 1, self.bond_dimension, 2], dtype = 'int32')
        Lambda = np.zeros([self.n_modes - 1, self.bond_dimension], dtype = 'float32')
        self.Lambda_edge = np.ones(self.bond_dimension, dtype = 'float32')
        Lambda[:, :self.local_hilbert_space_dimension ** 2] = Lambda_temp
        charge[:, :self.local_hilbert_space_dimension ** 2] = charge_temp

        requests = [[] for _ in self.available_nodes]
        self.Gammas = []
        self.Lambdas= []
        self.Charges = []
        '''Stores all data for all sites'''
        for site in range(self.n_modes + 1):
        
            node = site % len(self.available_nodes)

            if node == 0:
                '''Stores in the host node when destination node is 0'''
                '''Gammas only go to self.n'''
                if site < self.n_modes:
                    Gamma = np.zeros([self.bond_dimension, self.bond_dimension], dtype = 'complex64')
                    Gamma[ :self.local_hilbert_space_dimension ** 2, :self.local_hilbert_space_dimension ** 2] = Gamma_temp[site]
                    self.Gammas.append(Gamma)
                '''Lambdas only go to self.n - 1'''
                if site < self.n_modes - 1:
                    self.Lambdas.append(Lambda[site])
                '''Charges go to self.n + 1'''
                self.Charges.append(charge[site])
                
            else:
                '''Stores in different nodes using MPI if destination node is not 0'''
                data_rank = self.data_ranks[node]
                '''Wait until all requests completed'''
                while not MPI.Request.testall(requests[node])[0]:
                    time.sleep(0.01)
                requests[node] = []
                '''Gammas only go to self.n'''
                if site < self.n_modes:
                    Gamma = np.zeros([self.bond_dimension, self.bond_dimension], dtype = 'complex64')
                    Gamma[ :self.local_hilbert_space_dimension ** 2, :self.local_hilbert_space_dimension ** 2] = Gamma_temp[site]
                    requests[node].append(comm.Isend([Gamma, MPI.C_FLOAT_COMPLEX], data_rank, tag=1))
                '''Lambdas only go to self.n - 1'''
                if site < self.n_modes - 1:
                    requests[node].append(comm.Isend([Lambda[site], MPI.FLOAT], data_rank, tag=2))
                '''Charges go to self.n + 1'''
                requests[node].append(comm.Isend([charge[site], MPI.INT], data_rank, tag=3))

        '''Wait until all requests completed'''
        for node in self.available_nodes:
            while not MPI.Request.testall(requests[node])[0]:
                time.sleep(0.01)

        if not self.prob[0] > 0:
            print('Configuration yields invalid probabilities. Exiting')
            return False
        return True


    # MPO update after a two-qudit gate        
    def Request(self, seed, l, r, compute_node):

        compute_rank = self.compute_ranks[compute_node]

        # Telling compute node to expect compute load and relevant compute parameters except data
        comm.isend('New data coming', compute_rank, tag=100)
        comm.isend(r, compute_rank, tag=7)
        comm.isend(l, compute_rank, tag=8)
        comm.isend(seed, compute_rank, tag=9)

        # Which node to find the data for the l'th site
        data_node_0 = l % self.nodes
        data_node_1 = (l + 1) % self.nodes
        data_node_2 = (l + 2) % self.nodes
        data_rank_0 = self.data_ranks[data_node_0]
        data_rank_1 = self.data_ranks[data_node_1]
        data_rank_2 = self.data_ranks[data_node_2]
        if data_node_0 == 0:
            # Sending data to compute node
            LC = self.Lambdas[l // self.nodes]
            CL = self.Charges[l // self.nodes]
            Glc = self.Gammas[l // self.nodes]
            comm.Isend([LC, MPI.FLOAT], compute_rank, tag=0)
            comm.Isend([CL, MPI.INT], compute_rank, tag=2)
            comm.Isend([Glc, MPI.C_FLOAT_COMPLEX], compute_rank, tag=5)
            # Receiving compute results
            new_Lambda = np.zeros(self.bond_dimension, dtype='float32')
            Gamma0Out = np.zeros([self.bond_dimension, self.bond_dimension], dtype='complex64')
            comm.Irecv(new_Lambda, source=compute_rank, tag=11)
            comm.Irecv(Gamma0Out, source=compute_rank, tag=12)
            self.requests_buf[l][:2] = [new_Lambda, Gamma0Out]
        else:
            # Telling data nodes to send data to compute node
            comm.send('Data needed', data_rank_0, tag=100)
            # Telling data nodes what site is updated (l'th site)
            comm.isend(l, data_rank_0, tag=0)
            # Telling data nodes which side of data is needed
            comm.isend(0, data_rank_0, tag=1)
            # Telling data nodes which compute nde to send to
            comm.isend(compute_rank, data_rank_0, tag=2)
        if data_node_1 == 0:
            if l != self.n_modes - 2:
                LR = self.Lambdas[(l + 1) // self.nodes]
            else:
                LR = self.Lambda_edge
            comm.Isend([LR, MPI.FLOAT], compute_rank, tag=1)
            CC = self.Charges[(l + 1) // self.nodes]
            Gcr = self.Gammas[(l + 1) // self.nodes]
            comm.Isend([CC, MPI.INT], compute_rank, tag=3)
            comm.Isend([Gcr, MPI.C_FLOAT_COMPLEX], compute_rank, tag=6)
            new_charge = np.empty([self.bond_dimension, 2], dtype='int32')
            Gamma1Out = np.zeros([self.bond_dimension, self.bond_dimension], dtype='complex64')
            comm.Irecv(new_charge, source=compute_rank, tag=10)
            comm.Irecv(Gamma1Out, source=compute_rank, tag=13)
            self.requests_buf[l][2:] = [new_charge, Gamma1Out]
        else:
            comm.send('Data needed', data_rank_1, tag=100)
            comm.isend(l, data_rank_1, tag=0)
            comm.isend(1, data_rank_1, tag=1)
            comm.isend(compute_rank, data_rank_1, tag=2)
        if data_node_2 == 0:
            CR = self.Charges[(l + 2) // self.nodes]
            comm.Isend([CR, MPI.INT], compute_rank, tag=4)
        else:
            comm.send('Data needed', data_rank_2, tag=100)
            comm.isend(l, data_rank_2, tag=0)
            comm.isend(2, data_rank_2, tag=1)
            comm.isend(compute_rank, data_rank_2, tag=2)

        # Receiving timing meta data
        svd_time_req = comm.irecv(source=compute_rank, tag=14)
        theta_time_req = comm.irecv(source=compute_rank, tag=15)
        self.meta_data_requests[l] = [svd_time_req, theta_time_req]


    def Check(self, l) -> bool:

        # Determining if slave computational results are ready
        completed = MPI.Request.testall(self.meta_data_requests[l])
        if not completed[0]:
            return False

        # Loading timing meta data
        svd_time, theta_time = completed[1]
        self.svd_time += svd_time
        self.theta_time += theta_time

        # Loading compute node results if updates data stored on node 0
        data_node_0 = l % self.nodes
        data_node_1 = (l + 1) % self.nodes
        if data_node_0 == 0:
            new_Lambda, Gamma0Out = self.requests_buf[l][:2]
            self.Lambdas[l // self.nodes] = new_Lambda
            if l == self.n_modes - 2:
                self.Gammas[l // self.nodes][:, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2)] = Gamma0Out[:, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2)]
            else:
                self.Gammas[l // self.nodes][:, :] = Gamma0Out
        if data_node_1 == 0:
            new_charge, Gamma1Out = self.requests_buf[l][2:]
            self.Charges[(l + 1) // self.nodes] = new_charge
            if l == self.n_modes - 2:
                self.Gammas[(l + 1) // self.nodes][:min(self.bond_dimension, self.local_hilbert_space_dimension ** 2), 0] = Gamma1Out[:min(self.bond_dimension, self.local_hilbert_space_dimension ** 2), 0]
            else:
                self.Gammas[(l + 1) // self.nodes] = Gamma1Out
        return True


    def update_node_status(self):
        new_running_l_and_node = []
        for l, node in self.running_l_and_node:
            if self.Check(l):
                self.available_nodes.append(node)
            else:
                new_running_l_and_node.append([l, node])
        self.running_l_and_node = new_running_l_and_node


    def LayerUpdate(self, layer):
        start = time.time()
        for i, mode in enumerate(range(layer % 2, self.n_modes - 1, 2)):
            if mode >= self.n_input_states + layer:
                continue
            reflectivity = self.reflectivity[layer, i]
            seed = self.seeds[layer, i]
            while len(self.available_nodes) == 0:
                self.update_node_status()
                time.sleep(0.01)
            target_node = self.available_nodes.pop(0)
            self.Request(seed, mode, reflectivity, target_node)
            self.running_l_and_node.append([mode, target_node])
        while len(self.available_nodes) != self.nodes:
            self.update_node_status()
            time.sleep(0.01)
        for data_rank in self.data_ranks:
            if data_rank != 0:
                comm.isend('Layer finished', data_rank, tag=100)
        self.update_time += time.time() - start


    def Simulate(self):
        
        success = True

        if self.initialization_successful:
            self.EE[:, 0] = self.EntanglementEntropyDistributed()
        full_start = time.time()
        
            for layer in range(self.n_modes):
                self.LayerUpdate(layer)
            start = time.time()
                if layer % 10 == 0 or layer == self.n_modes - 1:
                    self.prob[layer+1] = self.ProbabilityDistributed()
                    prob = self.prob[np.where(self.prob > 0)[0]]
                    # if np.max(prob)/np.min(prob) - 1 > 0.1:
                    if np.min(prob) < 0.9:
                    print('Accuracy too low. Failed.')
                        success = False
                    break
                self.EE[:, layer+1] = self.EntanglementEntropyDistributed()
                max_ee_idx = np.argmax(self.EE)
                if layer - max_ee_idx % self.EE.shape[1] > 10:
                    print('EE did not increase for 10 epochs. Stopping')
                break
            self.ee_prob_cal_time += time.time() - start
            '''Initialial total time is much higher than simulation time due to initialization of cuda context.'''
                print("Total time: {:.2f}. Update time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. EE_Prob_cal_time: {:.2f}.".format(time.time()-full_start, self.update_time, self.theta_time, self.svd_time, self.ee_prob_cal_time))

        while len(self.available_nodes) != self.nodes:
            self.update_node_status()
            time.sleep(0.01)

        for target_node in self.available_nodes:
            comm.send('Finished', self.compute_ranks[target_node], tag=100)
            comm.send('Finished', self.data_ranks[target_node], tag=100)

        return success, self.prob, self.EE, self.REPar
    
    






    '''Utilities for producing summaries (Entanglement entropy and probability)'''
    
    
    def Sync(self, send_back: bool):
        requests = [[] for _ in self.available_nodes]
        Lambda = []
        Charge = []

        for site in range(self.n_modes + 1):
        
            node = site % self.nodes

            if node == 0:
                '''Stores in the host node when destination node is 0'''
                '''Lambdas only go to self.n - 1'''
                if site < self.n_modes - 1:
                    Lambda.append(self.Lambdas[site // self.nodes])
                '''Charges go to self.n + 1'''
                Charge.append(self.Charges[site // self.nodes])
                
            else:
                '''Stores in different nodes using MPI if destination node is not 0'''
                data_rank = self.data_ranks[node]
                '''Wait until all requests completed'''
                while not MPI.Request.testall(requests[node])[0]:
                    time.sleep(0.01)
                requests[node] = []
                '''Lambdas only go to self.n - 1'''
                if site < self.n_modes - 1:
                    Lambda_buf = np.zeros(self.bond_dimension, dtype = 'float32')
                    requests[node].append(comm.Irecv([Lambda_buf, MPI.FLOAT], data_rank, tag=2))
                    Lambda.append(Lambda_buf)
                '''Charges go to self.n + 1'''
                Charge_buf = self.local_hilbert_space_dimension * np.ones([self.bond_dimension, 2], dtype = 'int32')
                requests[node].append(comm.Irecv([Charge_buf, MPI.INT], data_rank, tag=3))
                Charge.append(Charge_buf)

        '''Wait until all requests completed'''
        for node in self.available_nodes:
            while not MPI.Request.testall(requests[node])[0]:
                time.sleep(0.01)

        '''Compile results and send to all data ranks'''
        Lambda = np.array(Lambda)
        Charge = np.array(Charge)

        if send_back:
            requests = []
            for data_rank in self.data_ranks:
                if data_rank != 0:
                    requests.append(comm.Isend(Lambda, data_rank, tag=0))
                    requests.append(comm.Isend(Charge, data_rank, tag=1))
            
            '''Wait until all requests completed'''
            while not MPI.Request.testall(requests)[0]:
                time.sleep(0.01)

        return Lambda, Charge


    def EntanglementEntropyDistributed(self):

        for data_rank in self.data_ranks:
            if data_rank != 0:
                comm.send('Compute EE', data_rank, tag=100)

        Lambda, _ = self.Sync(send_back=False)
        Output = np.zeros([self.n_modes - 1])
        sq_lambda = np.copy(Lambda ** 2)
        for i in range(self.n_modes - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i]))
        return Output


    def ProbabilityDistributed(self):
        '''Computing the total probability (see if it is conserved) from distributed Gamma, Lambda and charges.
        This uses a log-depth reduction approach. Probability computation is a chain matrix multiplication of Gammas,
        where indices of Gammas are selected based on Charge and Lambda.
        At each level, Gammas at all sites are sent to the first half of the sites,
        where pair-wise matrix multiplication reduces the number of gammas by half.
        This progresses to the next level until all gammas are reduced to a number.'''

        # Telling all data ranks to send information needed for probability computation
        for data_rank in self.data_ranks:
            if data_rank != 0:
                comm.send('Compute Prob', data_rank, tag=100)

        # Obtaining Lambda and Charge from all data ranks.
        # These arrays are small so it's okay to synchronize (no need for distributed storage)
        Lambda, Charge = self.Sync(send_back=True) # send_back=True means all data ranks will have synchronized Lambda and Charge as well
        length = self.n_modes
        GammaResults = self.Gammas

        # Selecting indices of gammas based on charge and lambda happens only at the first level, so first round of reduction is special
        first_round = True

        while length != 1: # length is the remaining number of unreduced gammas

            # Receiving size info (sizes of gammas after index selection)
            size_req_list = [[] for _ in range(length)]
            for site in range((length - 1) // 2 + 1): # Only the first half will receive data and therefore receive size info
                # Skip this site if this data node is not in charge of this site
                if self.this_node != site % self.nodes:
                    continue
                # Receiving size of first gamma
                size_req_list[site].append(comm.irecv(source = self.data_ranks[(2 * site) % self.nodes], tag = 102 + 2 * site))
                # Receive size of second gamma only if there is a second gamma needed
                # not needed for the last site when there are a odd number of sites
                if not (site == (length - 1) // 2 and length % 2 == 1):
                    size_req_list[site].append(comm.irecv(source = self.data_ranks[(2 * site + 1) % self.nodes], tag = 102 + 2 * site + 1))
            
            # Sending size info
            idx_list = [None for _ in range(length)]
            for site in range(length):
                if self.this_node != site % self.nodes:
                    continue
                target_site = site // 2 # Destination site sending gamma to
                if first_round:
                    # Selecting indices based on charge and lambda at the first level only
                    lambda_select = np.where(Lambda[site - 1] > 0)[0] if site != 0 else np.arange(self.bond_dimension)
                    charge_select = np.where(Charge[site, :, 0] == Charge[site, :, 1])[0]
                    idx1 = np.intersect1d(charge_select, lambda_select)
                    lambda_select = np.where(Lambda[site] > 0)[0] if site != self.n_modes - 1 else np.arange(self.bond_dimension)
                    charge_select = np.where(Charge[site + 1, :, 0] == Charge[site + 1, :, 1])[0]
                    idx2 = np.intersect1d(charge_select, lambda_select)
                    idx_list[site] = [idx1, idx2]
                    comm.send([idx1.shape[0], idx2.shape[0]], self.data_ranks[target_site % self.nodes], tag = 102 + site)
                else:
                    # Otherwise, simply send the shape of gamma
                    comm.send(GammaResults[site // self.nodes].shape, self.data_ranks[target_site % self.nodes], tag = 102 + site)

            # Receiving Gammas
            req = []
            gamma1s = []
            gamma2s = []
            size_list = []
            for site in range((length - 1) // 2 + 1):
                self.nodes
                if self.this_node != site % self.nodes:
                    continue
                # Only continue if size information arrived
                completed = MPI.Request.testall(size_req_list[site])
                while not completed[0]:
                    time.sleep(0.01)
                    completed = MPI.Request.testall(size_req_list[site])
                sizes = completed[1]
                size_list.append(sizes)
                try:
                size0, size1 = sizes[0]
                except:
                    print(sizes, completed)
                    comm.Abort()
                # Receiving gammas (reshaped to 1 dimension so that arrays don't get unexpectedly transposed)
                gamma1 = np.empty(size0 * size1, dtype = 'complex64')
                req.append(comm.Irecv([gamma1, MPI.C_FLOAT_COMPLEX], source = self.data_ranks[(2 * site) % self.nodes], tag = 102 + 2 * site))
                if site == (length - 1) // 2 and length % 2 == 1:
                    # set gamma2 to identity if second gamma is not needed for being the last gamma for odd lengths
                    gamma2 = np.identity(size1, dtype = 'complex64').reshape(-1)
                else:
                    size2, size3 = sizes[1]
                    gamma2 = np.empty(size2 * size3, dtype = 'complex64')
                    req.append(comm.Irecv([gamma2, MPI.C_FLOAT_COMPLEX], source = self.data_ranks[(2 * site + 1) % self.nodes], tag = 102 + 2 * site + 1))
                gamma1s.append(gamma1)
                gamma2s.append(gamma2)

            send_requests = []
            # Sending Gammas
            for site in range(length):
                if self.this_node != site % self.nodes:
                    continue
                target_site = site // 2
                gamma = GammaResults[site // self.nodes]
                if first_round:
                    idx1, idx2 = idx_list[site]
                    gamma = gamma[idx1][:, idx2]
                gamma = gamma.reshape(-1)
                send_requests.append(comm.Isend([gamma, MPI.C_FLOAT_COMPLEX], self.data_ranks[target_site % self.nodes], tag = 102 + site))

            GammaResults = [None for _ in range(len(gamma1s))]

            # Computing
            MPI.Request.waitall(req) # Continue only all gammas are received
            for i in range(len(gamma1s)):
                # Retrieve the sizes of gammas to reshape from 1d to 2d
                sizes = size_list[i]
                size0, size1 = sizes[0]
                if len(sizes) == 1: # Case where the second gamma is not needed
                    size2 = size3 = size1
                else:
                    size2, size3 = sizes[1]
                gamma1, gamma2 = gamma1s[i].reshape(size0, size1), gamma2s[i].reshape(size2, size3)

                gamma1 = np.matmul(gamma1, gamma2)
                GammaResults[i] = gamma1

            # Next iteration should have roughly half the length
            length = (length - 1) // 2 + 1
            # Next iteration is not the first round anymore
            first_round = False
            MPI.Request.waitall(send_requests)

        prob = np.sum(np.real(GammaResults[0]))
        print('Probability: ', prob)

        return prob