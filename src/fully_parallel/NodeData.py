import time

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD


class NodeDataWorker:

    def __init__(self, nodes, ranks_per_node, this_node):
        self.nodes = nodes
        self.this_node = this_node
        self.data_ranks = [node * ranks_per_node for node in range(nodes)]

    def ExperimentInit(self, n_modes, local_hilbert_space_dimension, bond_dimension):
        
        self.n_modes = n_modes
        self.local_hilbert_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = bond_dimension
        self.running_charges_and_rank = []
        self.sites = []
        self.sides = []
        self.requests = []
        self.requests_buf = []        

        self.Gammas = []
        self.Lambdas= []
        self.Lambda_edge = np.ones(self.bond_dimension, dtype = 'float32')
        self.Charges = []
        for site in range(self.n_modes + 1):
            node = site % self.nodes
            if node != self.this_node:
                continue
            '''Gammas only go to self.n'''
            if site < self.n_modes:
                site_gamma = np.empty([self.bond_dimension, self.bond_dimension], dtype = 'complex64')
                comm.Recv([site_gamma, MPI.C_FLOAT_COMPLEX], 0, tag=1)
                self.Gammas.append(site_gamma)
            '''Lambdas only go to self.n - 1'''
            if site < self.n_modes - 1:
                site_Lambda = np.empty(self.bond_dimension, dtype = 'float32')
                comm.Recv([site_Lambda, MPI.FLOAT], 0, tag=2)
                self.Lambdas.append(site_Lambda)
            '''Charges go to self.n + 1'''
            site_charge = np.empty([self.bond_dimension, 2], dtype = 'int32')
            comm.Recv([site_charge, MPI.INT], 0, tag=3)
            self.Charges.append(site_charge)


    def NodeProcess(self):

        l = comm.recv(source=0, tag=0)
        side = comm.recv(source=0, tag=1)
        compute_rank = comm.recv(source=0, tag=2)

        if side == 0:
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
            new_Lambda_req = comm.Irecv([new_Lambda, MPI.FLOAT], source=compute_rank, tag=11)
            Gamma0Out_req = comm.Irecv([Gamma0Out, MPI.C_FLOAT_COMPLEX], source=compute_rank, tag=12)
            self.requests.append([new_Lambda_req, Gamma0Out_req])
            self.requests_buf.append([new_Lambda, Gamma0Out])
            self.sides.append(0)
            self.sites.append(l)
        if side == 1:
            if l != self.n_modes - 2:
                LR = self.Lambdas[(l + 1) // self.nodes]
            else:
                LR = self.Lambda_edge
            CC = self.Charges[(l + 1) // self.nodes]
            Gcr = self.Gammas[(l + 1) // self.nodes]
            comm.Isend([LR, MPI.FLOAT], compute_rank, tag=1)
            comm.Isend([CC, MPI.INT], compute_rank, tag=3)
            comm.Isend([Gcr, MPI.C_FLOAT_COMPLEX], compute_rank, tag=6)
            new_charge = np.empty([self.bond_dimension, 2], dtype='int32')
            Gamma1Out = np.zeros([self.bond_dimension, self.bond_dimension], dtype='complex64')
            new_charge_req = comm.Irecv([new_charge, MPI.INT], source=compute_rank, tag=10)
            Gamma1Out_req = comm.Irecv([Gamma1Out, MPI.C_FLOAT_COMPLEX], source=compute_rank, tag=13)
            self.requests.append([new_charge_req, Gamma1Out_req])
            self.requests_buf.append([new_charge, Gamma1Out])
            self.sides.append(1)
            self.sites.append(l)
        if side == 2:
            CR = self.Charges[(l + 2) // self.nodes]
            comm.Isend([CR, MPI.INT], compute_rank, tag=4)
    
    
    def Simulate(self):

        status_req = comm.irecv(source=0, tag=100)
        completed = MPI.Request.test(status_req)
        while not completed[0]:
            self.check()
            time.sleep(0.01)
            completed = MPI.Request.test(status_req)
        status = completed[1]
        
        while status != 'Finished':
            if status == 'Data needed':
                self.NodeProcess()
            elif status == 'Layer finished':
                self.complete_all()
            elif status == 'Compute EE':
                self.ComputeEE()
            elif status == 'Compute Prob':
                self.ComputeProb()
            else:
                raise Exception("Not a valid status from full_gbs_mpo.")
            status_req = comm.irecv(source=0, tag=100)
            completed = MPI.Request.test(status_req)
            
            while not completed[0]:
                self.check()
                time.sleep(0.01)
                completed = MPI.Request.test(status_req)
            status = completed[1]

    
    '''Synchronize Lambda and Charge with host'''
    def Sync(self, receive_back: bool):
        for site in range(self.n_modes + 1):
            node = site % self.nodes
            if node != self.this_node:
                continue
            
            '''Lambdas only go to self.n - 1'''
            if site < self.n_modes - 1:
                comm.Send([self.Lambdas[site // self.nodes], MPI.FLOAT], 0, tag=2)
            '''Charges go to self.n + 1'''
            comm.Send([self.Charges[site // self.nodes], MPI.INT], 0, tag=3)

        if receive_back:
            Lambda = np.empty([self.n_modes - 1, self.bond_dimension], dtype = 'float32')
            Charge = np.empty([self.n_modes + 1, self.bond_dimension, 2], dtype = 'int32')
            comm.Recv([Lambda, MPI.FLOAT], source=0, tag=0)
            comm.Recv([Charge, MPI.INT], source=0, tag=1)
            return Lambda, Charge

    def ComputeEE(self):
        self.Sync(receive_back=False)

    def ComputeProb(self):
        '''Computing the total probability (see if it is conserved) from distributed Gamma, Lambda and charges.
        This uses a log-depth reduction approach. Probability computation is a chain matrix multiplication of Gammas,
        where indices of Gammas are selected based on Charge and Lambda.
        At each level, Gammas at all sites are sent to the first half of the sites,
        where pair-wise matrix multiplication reduces the number of gammas by half.
        This progresses to the next level until all gammas are reduced to a number.'''

        # Obtaining Lambda and Charge from all data ranks.
        # These arrays are small so it's okay to synchronize (no need for distributed storage)
        # Lambda, Charge, Gamma = self.Sync(receive_back=True) # send_back=True means all data ranks will have synchronized Lambda and Charge as well
        Lambda, Charge = self.Sync(receive_back=True)
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
                size0, size1 = sizes[0]
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

            # Sending Gammas
            send_requests = []
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

        

    def check(self):
        
        new_sites = []
        new_sides = []
        new_requests = []
        new_requests_buf = []
        while len(self.sites) > 0:
            site = self.sites.pop(0)
            side = self.sides.pop(0)
            req = self.requests.pop(0)
            buf = self.requests_buf.pop(0)
            completed = MPI.Request.testall(req)
            if completed[0]:
                # Loading compute node results if updates data stored on node 0
                if side == 0:
                    new_Lambda, Gamma0Out = buf
                    self.Lambdas[site // self.nodes] = new_Lambda
                    if site == self.n_modes - 2:
                        self.Gammas[site // self.nodes][:, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2)] = Gamma0Out[:, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2)]
                    else:
                        self.Gammas[site // self.nodes] = Gamma0Out
                if side == 1:
                    new_charge, Gamma1Out = buf
                    self.Charges[(site + 1) // self.nodes] = new_charge
                    if site == self.n_modes - 2:
                        self.Gammas[(site + 1) // self.nodes][:min(self.bond_dimension, self.local_hilbert_space_dimension ** 2), 0] = Gamma1Out[:min(self.bond_dimension, self.local_hilbert_space_dimension ** 2), 0]
                    else:
                        self.Gammas[(site + 1) // self.nodes] = Gamma1Out
            else:
                new_sites.append(site)
                new_sides.append(side)
                new_requests.append(req)
                new_requests_buf.append(buf)
        self.sites = new_sites
        self.sides = new_sides
        self.requests = new_requests
        self.requests_buf = new_requests_buf


    def complete_all(self):
        while not self.is_all_buf_clear():
            self.check()
        self.requests_buf = []
        self.requests = []
        self.sides = []
        self.sites = []

    def is_all_buf_clear(self):
        for buf in self.requests_buf:
            if buf != None:
                return False
        return True