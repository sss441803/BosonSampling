'''Full simulation code containing the Device method (cupy, unified update)'''
import time

import numpy as np
from qutip import squeeze, thermal_dm
from scipy.stats import rv_continuous

from mpi4py import MPI
comm = MPI.COMM_WORLD

from Canonicalize import Canonicalize

# mempool.set_limit(size=2.5 * 10**9)  # 2.3 GiB

# np.random.seed(1)
np.set_printoptions(precision=3)

data_type = np.complex64
float_type = np.float32
int_type = np.int32


class FullCompute:
    def __init__(self, nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):

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
        self.reflectivity = np.empty([self.n, self.n // 2])
        self.PS = PS
        self.normalization = None
        self.update_time = 0
        self.U_time = 0
        self.theta_time = 0
        self.svd_time = 0
        self.ee_prob_cal_time = 0
        self.before_rank_time = 0
        self.rank_time = 0
        self.after_rank_time = 0
        self.nodes = nodes
        self.this_node = 0
        self.meta_data_requests = [None for _ in range(self.n - 1)]
        # self.requests = [None for _ in range(self.n - 1)]
        self.requests_buf = [[None, None, None, None] for _ in range(self.n - 1)]
        self.available_nodes = [node for node in range(nodes)]
        self.num_nodes = nodes
        self.compute_ranks = [node * ranks_per_node + 1 for node in range(nodes)]
        self.data_ranks = [node * ranks_per_node for node in range(nodes)]
        self.running_l_and_node = []

    def MPOInitialization(self):
        
        chi = self.chi; init_chi = self.init_chi; d = self.d; K = self.K

        self.Lambda_edge = np.ones(chi, dtype = 'float32') # edge lambda (for first and last site) don't exists and are ones
        Lambda = np.zeros([init_chi, self.n - 1], dtype = 'float32')
        Gamma = np.zeros([init_chi, init_chi, self.n], dtype = 'complex64')  
        charge = d * np.ones([init_chi, self.n + 1, 2], dtype = 'int32')
        charge[0] = 0
        
        am = (1 - self.loss) * np.exp(- 2 * self.r) + self.loss
        ap = (1 - self.loss) * np.exp(2 * self.r) + self.loss
        s = 1 / 4 * np.log(ap / am)
        n_th = 1 / 2 * (np.sqrt(am * ap) - 1)
        nn = 40
        
        sq = (squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()).full()[:(d + 1), :(d + 1)]

        if self.PS == None:
            for i in range(d):
                charge[i, 0, 0] = i
                charge[i, 0, 1] = i
            #pre_chi = d
            updated_bonds = np.array([bond for bond in range(d)])
        else:
            charge[0, 0, 0] = self.PS
            charge[0, 0, 1] = self.PS
            # pre_chi = 1
            updated_bonds = np.array([0])

        for i in range(K - 1):
            print('Initializing mode ', i)
            #chi_ = 0
            #for j in range(pre_chi):
            bonds_updated = np.zeros(d**2)
            for j in updated_bonds:
                if charge[j, i, 0] == d:
                    c1 = 0
                else:
                    c1 = charge[j, i, 0]
                for ch_diff1 in range(c1, -1, -1):
                    if charge[j, i, 1] == d:
                        c2 = 0
                    else:
                        c2 = charge[j, i, 1]
                    for ch_diff2 in range(c2, -1, -1):
                        if np.abs(sq[ch_diff1, ch_diff2]) <= self.errtol:
                            continue
                        Gamma[j, (c1 - ch_diff1) * d + c2 - ch_diff2, i] = sq[ch_diff1, ch_diff2]
                        charge[(c1 - ch_diff1) * d + c2 - ch_diff2, i + 1, 0] = c1 - ch_diff1
                        charge[(c1 - ch_diff1) * d + c2 - ch_diff2, i + 1, 1] = c2 - ch_diff2
                        bonds_updated[(c1 - ch_diff1) * d + c2 - ch_diff2] = 1

            updated_bonds = np.where(bonds_updated == 1)[0]
            Lambda[updated_bonds, i] = 1

        print('Computing Gamma')
        for j in updated_bonds:
            if charge[j, K - 1, 0] == d:
                c0 = 0
            else:
                c0 = charge[j, K - 1, 0]
            if charge[j, K - 1, 1] == d:
                c1 = 0
            else:
                c1 = charge[j, K - 1, 1]
            Gamma[j, 0, K - 1] = sq[c0, c1]
        
        for i in range(self.m - 1, self.n - 1):
            Lambda[0, i] = 1
            charge[0, i + 1, 0] = 0
            charge[0, i + 1, 1] = 0
        
        print('Update gamma from gamme_temp')
        for i in range(self.m):
            Gamma[:, :, i] = np.multiply(Gamma[:, :, i], Lambda[:, i].reshape(1, -1))

        print('Update the rest of gamma values to 1')
        for i in range(self.m, self.n):
            Gamma[0, 0, i] = 1

        print('Array transposition')
        Gamma = np.transpose(Gamma, (2, 0, 1))
        Lambda = np.transpose(Lambda, (1, 0))
        charge = np.transpose(charge, (1, 0, 2))

        print('Start sorting')

        # Sorting bonds based on bond charges
        for i in range(self.n + 1):
            idx = np.lexsort((charge[i, :, 1], charge[i, :, 0]))
            charge[i] = charge[i, idx]
            if i > 0:
                Gamma[i - 1] = Gamma[i - 1][:, idx]
                if i < self.n:
                    Lambda[i - 1] = Lambda[i - 1, idx]
            if i < self.n:
                Gamma[i] = Gamma[i, idx]

        print('Canonicalization update')
        Canonicalize(self.n, self.d, self.init_chi, Gamma, Lambda, charge)

        charge_temp = np.copy(charge)
        Lambda_temp = np.copy(Lambda)
        Gamma_temp = np.copy(Gamma)
        charge = d * np.ones([self.n + 1, chi, 2], dtype = 'int32')
        Lambda = np.zeros([self.n - 1, chi], dtype = 'float32')
        Lambda[:, :init_chi] = Lambda_temp
        charge[:, :init_chi] = charge_temp

        requests = [[] for _ in self.available_nodes]
        self.Gammas = []
        self.Lambdas= []
        self.Charges = []
        '''Stores all data for all sites'''
        for site in range(self.n + 1):
        
            node = site % len(self.available_nodes)
            # print('Full: node {} site {}'.format(node, site))

            if node == 0:
                '''Stores in the host node when destination node is 0'''
                '''Gammas only go to self.n'''
                if site < self.n:
                    Gamma = np.zeros([self.chi, self.chi], dtype = 'complex64')
                    Gamma[ :init_chi, :init_chi] = Gamma_temp[site]
                    self.Gammas.append(Gamma)
                '''Lambdas only go to self.n - 1'''
                if site < self.n - 1:
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
                if site < self.n:
                    Gamma = np.zeros([self.chi, self.chi], dtype = 'complex64')
                    Gamma[ :init_chi, :init_chi] = Gamma_temp[site]
                    requests[node].append(comm.Isend([Gamma, MPI.C_FLOAT_COMPLEX], data_rank, tag=1))
                '''Lambdas only go to self.n - 1'''
                if site < self.n - 1:
                    requests[node].append(comm.Isend([Lambda[site], MPI.FLOAT], data_rank, tag=2))
                '''Charges go to self.n + 1'''
                requests[node].append(comm.Isend([charge[site], MPI.INT], data_rank, tag=3))

        '''Wait until all requests completed'''
        for node in self.available_nodes:
            while not MPI.Request.testall(requests[node])[0]:
                time.sleep(0.01)

        self.UpdateReflectivity()


    #MPO update after a two-qudit gate        
    def Request(self, l, r, compute_node):

        # print('In master request node ', target_node)

        seed = np.random.randint(0, 13579)
        np.random.seed(seed)

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
        # print('l: {}. Compute rank {}'.format(l, compute_rank))
        # print('data nodes: ', data_node_0, data_node_1, data_node_2)
        # print('data ranks: ', data_rank_0, data_rank_1, data_rank_2)
        if data_node_0 == 0:
            # Sending data to compute node
            LC = self.Lambdas[l // self.nodes]
            CL = self.Charges[l // self.nodes]
            Glc = self.Gammas[l // self.nodes]
            comm.Isend([LC, MPI.FLOAT], compute_rank, tag=0)
            comm.Isend([CL, MPI.INT], compute_rank, tag=2)
            comm.Isend([Glc, MPI.C_FLOAT_COMPLEX], compute_rank, tag=5)
            # Receiving compute results
            new_Lambda = np.zeros(self.chi, dtype='float32')
            Gamma0Out = np.zeros([self.chi, self.chi], dtype='complex64')
            comm.Irecv(new_Lambda, source=compute_rank, tag=11)
            comm.Irecv(Gamma0Out, source=compute_rank, tag=12)
            # self.requests[l] += [new_Lambda_req, Gamma0Out_req]
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
            if l != self.n - 2:
                LR = self.Lambdas[(l + 1) // self.nodes]
            else:
                LR = self.Lambda_edge
            comm.Isend([LR, MPI.FLOAT], compute_rank, tag=1)
            CC = self.Charges[(l + 1) // self.nodes]
            Gcr = self.Gammas[(l + 1) // self.nodes]
            comm.Isend([CC, MPI.INT], compute_rank, tag=3)
            comm.Isend([Gcr, MPI.C_FLOAT_COMPLEX], compute_rank, tag=6)
            new_charge = np.empty([self.chi, 2], dtype='int32')
            Gamma1Out = np.zeros([self.chi, self.chi], dtype='complex64')
            comm.Irecv(new_charge, source=compute_rank, tag=10)
            comm.Irecv(Gamma1Out, source=compute_rank, tag=13)
            # self.requests[l] += [new_charge_req, Gamma1Out_req]
            self.requests_buf[l][2:] = [new_charge, Gamma1Out]
        else:
            # print('sending status')
            comm.send('Data needed', data_rank_1, tag=100)
            # print('finished sending status')
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
        before_rank_time_req = comm.irecv(source=compute_rank, tag=16)
        rank_time_req = comm.irecv(source=compute_rank, tag=17)
        after_rank_time_req = comm.irecv(source=compute_rank, tag=18)
        self.meta_data_requests[l] = [svd_time_req, theta_time_req, before_rank_time_req, rank_time_req, after_rank_time_req]


    def Check(self, l) -> bool:

        # Determining if slave computational results are ready
        completed = MPI.Request.testall(self.meta_data_requests[l])
        if not completed[0]:
            return False

        # Loading timing meta data
        svd_time, theta_time, before_rank_time, rank_time, after_rank_time = completed[1]
        self.svd_time += svd_time
        self.theta_time += theta_time
        self.before_rank_time += before_rank_time
        self.rank_time += rank_time
        self.after_rank_time += after_rank_time

        # Loading compute node results if updates data stored on node 0
        data_node_0 = l % self.nodes
        data_node_1 = (l + 1) % self.nodes
        if data_node_0 == 0:
            new_Lambda, Gamma0Out = self.requests_buf[l][:2]
            self.Lambdas[l // self.nodes] = new_Lambda
            if l == self.n - 2:
                self.Gammas[l // self.nodes][:, :min(self.chi, self.d ** 2)] = Gamma0Out[:, :min(self.chi, self.d ** 2)]
            else:
                self.Gammas[l // self.nodes][:, :] = Gamma0Out
        if data_node_1 == 0:
            new_charge, Gamma1Out = self.requests_buf[l][2:]
            self.Charges[(l + 1) // self.nodes] = new_charge
            if l == self.n - 2:
                self.Gammas[(l + 1) // self.nodes][:min(self.chi, self.d ** 2), 0] = Gamma1Out[:min(self.chi, self.d ** 2), 0]
            else:
                self.Gammas[(l + 1) // self.nodes] = Gamma1Out
        return True


    def UpdateReflectivity(self):
        
        for k in range(self.n - 1):
            # print('k, ', k)
            if k < self.n / 2:
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
                    # print(i, k, k-(i+1)//2)
                    self.reflectivity[i, k-(i+1)//2] = np.sqrt(1 - T)
                    l -= 1
                    i += 1
            else:
                temp1 = 2 * self.n - (2 * k + 3)
                temp2 = 2
                l = self.n - 2
                first_layer = 2 * k - self.n + 2
                for i in range(2 * self.n - 2 * k - 2):
                    if temp1 >= 0:
                        T = my_cv.rvs(2 * self.n - (2 * k + 1), temp1)
                        temp1 -= 2
                    else:
                        T = my_cv.rvs(2 * self.n - (2 * k + 1), temp2)
                        temp2 += 2
                    self.reflectivity[first_layer + i, self.n//2-1-(i+1)//2] = np.sqrt(1 - T)
                    l -= 1    


    def update_node_status(self):
        new_running_l_and_node = []
        for l, node in self.running_l_and_node:
            if self.Check(l):
                self.available_nodes.append(node)
            else:
                new_running_l_and_node.append([l, node])
        self.running_l_and_node = new_running_l_and_node
        # print(self.running_l_and_node)

    def LayerUpdate(self, k):
        # print('updating layer ', k)
        start = time.time()
        for i, l in enumerate(range(k % 2, self.n - 1, 2)):
            if l >= self.m + k:
                continue
            reflectivity = self.reflectivity[k, i]
            # print('finished reflectivity')
            # print(self.available_nodes)
            while len(self.available_nodes) == 0:
                # print('checking avaiable')
                self.update_node_status()
                time.sleep(0.01)
            target_node = self.available_nodes.pop(0)
            self.Request(l, reflectivity, target_node)
            # print('finished request')
            self.running_l_and_node.append([l, target_node])
        while len(self.available_nodes) != self.num_nodes:
            self.update_node_status()
            # print('waiting finish layer')
            time.sleep(0.01)
        for data_rank in self.data_ranks:
            if data_rank != 0:
                comm.isend('Layer finished', data_rank, tag=100)
        self.update_time += time.time() - start


    def FullUpdate(self):
        
        full_start = time.time()
        
        self.MPOInitialization()
        self.TotalProbPar[0] = self.TotalProbFromMPO()
        self.EEPar[:, 0] = self.MPOEntanglementEntropy()
        # alpha_array = [0.5, 0.6, 0.7, 0.8, 0.9]
        # for i in range(5):
            # self.REPar[:, 0, i] = self.MPORenyiEntropy(alpha_array[i])
        for k in range(self.n - 1):
            self.LayerUpdate(k)
            start = time.time()
            if k % 10 == 9 or k == self.n - 2:
                self.TotalProbPar[k+1] = self.TotalProbFromMPO()
                self.EEPar[:, k+1] = self.MPOEntanglementEntropy()
                # print(self.EEPar[:, k+1])
            self.ee_prob_cal_time += time.time() - start
            # for i in range(5):
                # self.REPar[:, k + 1, i] = self.MPORenyiEntropy(alpha_array[i])
            '''Initialial total time is much higher than simulation time due to initialization of cuda context.'''
            print("m: {:.2f}. Total time: {:.2f}. Update time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. EE_Prob_cal_time: {:.2f}. Before rank time: {:.2f}. Rank time: {:.2f}. After rank time: {:.2f}".format(self.m, time.time()-full_start, self.update_time, self.theta_time, self.svd_time, self.ee_prob_cal_time, self.before_rank_time, self.rank_time, self.after_rank_time))

        while len(self.available_nodes) != self.num_nodes:
            self.update_node_status()
            time.sleep(0.01)

        for target_node in self.available_nodes:
            comm.send('Finished', self.compute_ranks[target_node], tag=100)
            comm.send('Finished', self.data_ranks[target_node], tag=100)

        return self.TotalProbPar, self.EEPar, self.REPar
    
    
    def Sync(self, send_back: bool):
        requests = [[] for _ in self.available_nodes]
        Lambda = []
        Charge = []
        # Gamma = []

        for site in range(self.n + 1):

            # print('Host site {}.'.format(site))
        
            node = site % self.nodes
            # print('Full: node {} site {}'.format(node, site))

            if node == 0:
                '''Stores in the host node when destination node is 0'''
                # '''Gammas only go to self.n'''
                # if site < self.n:
                #     Gamma.append(self.Gammas[site // self.nodes])
                '''Lambdas only go to self.n - 1'''
                if site < self.n - 1:
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
                # '''Gammas only go to self.n'''
                # if site < self.n:
                #     Gamma_buf = np.empty([self.chi, self.chi], dtype = 'complex64')
                #     requests[node].append(comm.Irecv([Gamma_buf, MPI.C_FLOAT_COMPLEX], data_rank, tag=4))
                #     Gamma.append(Gamma_buf)
                '''Lambdas only go to self.n - 1'''
                if site < self.n - 1:
                    Lambda_buf = np.zeros(self.chi, dtype = 'float32')
                    requests[node].append(comm.Irecv([Lambda_buf, MPI.FLOAT], data_rank, tag=2))
                    Lambda.append(Lambda_buf)
                '''Charges go to self.n + 1'''
                Charge_buf = self.d * np.ones([self.chi, 2], dtype = 'int32')
                requests[node].append(comm.Irecv([Charge_buf, MPI.INT], data_rank, tag=3))
                Charge.append(Charge_buf)

        '''Wait until all requests completed'''
        for node in self.available_nodes:
            while not MPI.Request.testall(requests[node])[0]:
                time.sleep(0.01)

        '''Compile results and send to all data ranks'''
        Lambda = np.array(Lambda)
        Charge = np.array(Charge)
        # Gamma = np.array(Gamma)

        if send_back:
            requests = []
            for data_rank in self.data_ranks:
                if data_rank != 0:
                    requests.append(comm.Isend(Lambda, data_rank, tag=0))
                    requests.append(comm.Isend(Charge, data_rank, tag=1))
                    # requests.append(comm.Isend(Gamma, data_rank, tag=2))
            
            '''Wait until all requests completed'''
            while not MPI.Request.testall(requests)[0]:
                time.sleep(0.01)

        # if send_back:
        #     return Lambda, Charge, Gamma

        # else:
        #     return Lambda, Charge
        return Lambda, Charge

    def MPOEntanglementEntropy(self):

        for data_rank in self.data_ranks:
            if data_rank != 0:
                comm.send('Compute EE', data_rank, tag=100)

        Lambda, _ = self.Sync(send_back=False)
        Output = np.zeros([self.n - 1])
        sq_lambda = np.copy(Lambda ** 2)
        for i in range(self.n - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i]))
        return Output

    def TotalProbFromMPO(self):
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
        length = self.n
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
                size_req_list[site].append(comm.irecv(source = self.data_ranks[(2 * site) % self.nodes], tag = 2 * site))
                # Receive size of second gamma only if there is a second gamma needed
                # not needed for the last site when there are a odd number of sites
                if not (site == (length - 1) // 2 and length % 2 == 1):
                    size_req_list[site].append(comm.irecv(source = self.data_ranks[(2 * site + 1) % self.nodes], tag = 2 * site + 1))
            
            # Sending size info
            idx_list = [None for _ in range(length)]
            for site in range(length):
                if self.this_node != site % self.nodes:
                    continue
                target_site = site // 2 # Destination site sending gamma to
                if first_round:
                    # Selecting indices based on charge and lambda at the first level only
                    lambda_select = np.where(Lambda[site - 1] > 0)[0] if site != 0 else np.arange(self.chi)
                    charge_select = np.where(Charge[site, :, 0] == Charge[site, :, 1])[0]
                    idx1 = np.intersect1d(charge_select, lambda_select)
                    lambda_select = np.where(Lambda[site] > 0)[0] if site != self.n - 1 else np.arange(self.chi)
                    charge_select = np.where(Charge[site + 1, :, 0] == Charge[site + 1, :, 1])[0]
                    idx2 = np.intersect1d(charge_select, lambda_select)
                    idx_list[site] = [idx1, idx2]
                    comm.send([idx1.shape[0], idx2.shape[0]], self.data_ranks[target_site % self.nodes], tag = site)
                else:
                    # Otherwise, simply send the shape of gamma
                    comm.send(GammaResults[site // self.nodes].shape, self.data_ranks[target_site % self.nodes], tag = site)

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
                req.append(comm.Irecv([gamma1, MPI.C_FLOAT_COMPLEX], source = self.data_ranks[(2 * site) % self.nodes], tag = 2 * site))
                if site == (length - 1) // 2 and length % 2 == 1:
                    # set gamma2 to identity if second gamma is not needed for being the last gamma for odd lengths
                    gamma2 = np.identity(size1, dtype = 'complex64').reshape(-1)
                else:
                    size2, size3 = sizes[1]
                    gamma2 = np.empty(size2 * size3, dtype = 'complex64')
                    req.append(comm.Irecv([gamma2, MPI.C_FLOAT_COMPLEX], source = self.data_ranks[(2 * site + 1) % self.nodes], tag = 2 * site + 1))
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
                # print('host gamma: ', gamma)
                send_requests.append(comm.Isend([gamma, MPI.C_FLOAT_COMPLEX], self.data_ranks[target_site % self.nodes], tag = site))

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
                # if first_round:
                #     site = 2 * (i * self.nodes + self.this_node)
                #     lambda_select = np.where(Lambda[site - 1] > 0)[0] if site != 0 else np.arange(self.chi)
                #     charge_select = np.where(Charge[site, :, 0] == Charge[site, :, 1])[0]
                #     idx1 = np.intersect1d(charge_select, lambda_select)
                #     lambda_select = np.where(Lambda[site] > 0)[0] if site != self.n - 1 else np.arange(self.chi)
                #     charge_select = np.where(Charge[site + 1, :, 0] == Charge[site + 1, :, 1])[0]
                #     idx2 = np.intersect1d(charge_select, lambda_select)
                #     if not np.allclose(gamma1, Gamma[site, idx1][:, idx2]):
                #         print(gamma1, Gamma[site, idx1][:, idx2])
                #     if site < self.n - 1:
                #         lambda_select = np.where(Lambda[site + 1] > 0)[0] if site + 1 != self.n - 1 else np.arange(self.chi)
                #         charge_select = np.where(Charge[site + 2, :, 0] == Charge[site + 2, :, 1])[0]
                #         idx3 = np.intersect1d(charge_select, lambda_select)
                #         if not np.allclose(gamma2, Gamma[site + 1, idx2][:, idx3]):
                #             print(gamma2, Gamma[site + 1, idx2][:, idx3])
                    # print(gamma1.shape, idx1.shape, idx2.shape)
                    
                # print('mat ', gamma1, gamma2)
                # print('mat shape: ', gamma1.shape, gamma2.shape)
                gamma1 = np.matmul(gamma1, gamma2)
                GammaResults[i] = gamma1

            # Next iteration should have roughly half the length
            length = (length - 1) // 2 + 1
            # Next iteration is not the first round anymore
            first_round = False
            MPI.Request.waitall(send_requests)

        prob = np.sum(GammaResults[0])
        print('Probability: ', prob)
        # return prob

        # # Gamma = np.array(self.Gammas)
        # R = Gamma[self.n - 1, :, 0]
        # pre_idx = np.array([0])
        # RTemp = np.copy(R)
        # for k in range(self.n - 2):
        #     idx = np.array([], dtype = 'int32')
        #     for ch in range(self.d):
        #         idx = np.append(idx, np.intersect1d(np.nonzero(Charge[self.n - 1 - k, :, 0] == ch), np.intersect1d(np.nonzero(Charge[self.n - 1 - k, :, 1] == ch), np.nonzero(Lambda[self.n - 1 - k - 1] > 0))))
        #     R = np.matmul(Gamma[self.n - 1 - k - 1, :, idx].T, RTemp[idx].reshape(-1))
        #     # print('baseline mat ', Gamma[self.n - 1 - k, idx][:, pre_idx])
        #     pre_idx = idx
        #     # print('Baseline idx shape: ', idx.shape[0], R.shape)
        #     RTemp = np.copy(R)
        # idx = np.array([], dtype = 'int32')
        # for ch in range(self.d):
        #     idx = np.append(idx, np.intersect1d(np.nonzero(Charge[1, :, 0] == ch), np.intersect1d(np.nonzero(Charge[1, :, 1] == ch), np.nonzero(Lambda[0, :] > 0))))
        # res = np.matmul(Gamma[0, :, idx].T, RTemp[idx].reshape(-1))
        # # print('Baseline idx shape: ', idx.shape[0], res.shape)
        # tot_prob = np.sum(res)
        # print('Probability: ', np.real(tot_prob))
        # # if self.normalization != None:
        # #     if tot_prob/self.normalization > 1.05 or tot_prob/self.normalization < 0.95:
        # #         quit()
        return prob
        

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