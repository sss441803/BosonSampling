'''Full simulation code containing the Device method (cupy, unified update)'''
import time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from ..interferometer import ReflectivityAndSeeds
from ..MPO import Initialize
from ..summary import Probability, EntanglementEntropy

class FullWorker:
    def __init__(self, nodes, ranks_per_node):
        
        self.available_ranks = [node * ranks_per_node + rank + node + 1 for node in range(nodes) for rank in range(ranks_per_node - 1)]
        # print('available ranks: ', self.available_ranks)
        self.num_gpu_ranks = nodes * (ranks_per_node - 1)


    def ExperimentInit(self, input_state_type, n_modes, n_input_states, post_selected_photon_number,local_hilbert_space_dimension, bond_dimension, parameters, seed=None):

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

        self.requests = [None for _ in range(self.n_modes - 1)]
        self.requests_buf = [None for _ in range(self.n_modes - 1)]
        self.running_l_and_rank = []

        # Initialize the reflectivities of beam splitters to form a global Haar random array
        self.reflectivity, self.seeds = ReflectivityAndSeeds(n_modes, seed)
        # Initialize the MPO
        self.initialization_successful = self.MPOInit(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters)


    def MPOInit(self, input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters):

        self.Gamma, self.Lambda, self.charge = Initialize(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters)
        
        self.prob[0] = np.real(Probability(self.n_modes, self.local_hilbert_space_dimension, self.Gamma, self.Lambda, self.charge))
        if not self.prob[0] > 0:
            print('Configuration yields invalid probabilities. Exiting')
            return False

        charge_temp = np.copy(self.charge)
        Lambda_temp = np.copy(self.Lambda)
        Gamma_temp = np.copy(self.Gamma)
        self.charge = self.local_hilbert_space_dimension * np.ones([self.n_modes + 1, self.bond_dimension, 2], dtype = 'int32')
        self.Lambda = np.zeros([self.n_modes - 1, self.bond_dimension], dtype = 'float32')
        self.Lambda_edge = np.ones(self.bond_dimension, dtype = 'float32')
        self.Gamma = np.zeros([self.n_modes, self.bond_dimension, self.bond_dimension], dtype = 'complex64')
        initial_bond_dimension = self.local_hilbert_space_dimension ** 2
        self.Gamma[:, :initial_bond_dimension, :initial_bond_dimension] = Gamma_temp
        self.Lambda[:, :initial_bond_dimension] = Lambda_temp
        self.charge[:, :initial_bond_dimension] = charge_temp

        return True


    # MPO update of a two-qudit gate        
    def Request(self, seed, mode, reflectivity, target_rank):

        # Determining the location of the two qubit gate
        LC = self.Lambda[mode,:]
        if mode == self.n_modes - 2:
            LR = self.Lambda_edge[:]
        else:
            LR = self.Lambda[mode+1,:]
        
        Glc = self.Gamma[mode,:]
        Gcr = self.Gamma[mode+1,:]

        # charge of corresponding index (bond charge left/center/right)
        CL = self.charge[mode]
        CC = self.charge[mode+1]
        CR = self.charge[mode+2]

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

        bond_dimension = self.bond_dimension
        new_charge = self.local_hilbert_space_dimension * np.ones([bond_dimension, 2], dtype='int32')
        new_Lambda = np.zeros(bond_dimension, dtype='float32')
        Gamma0Out = np.zeros([bond_dimension, bond_dimension], dtype='complex64')
        Gamma1Out = np.zeros([bond_dimension, bond_dimension], dtype='complex64')
        new_charge_req = comm.Irecv(new_charge, source=target_rank, tag=9)
        new_Lambda_req = comm.Irecv(new_Lambda, source=target_rank, tag=10)
        Gamma0Out_req = comm.Irecv(Gamma0Out, source=target_rank, tag=11)
        Gamma1Out_req = comm.Irecv(Gamma1Out, source=target_rank, tag=12)

        svd_time_req = comm.irecv(source=target_rank, tag=13)
        theta_time_req = comm.irecv(source=target_rank, tag=14)
        self.requests[mode] = [new_charge_req, new_Lambda_req, Gamma0Out_req, Gamma1Out_req, svd_time_req, theta_time_req]
        self.requests_buf[mode] = [new_charge, new_Lambda, Gamma0Out, Gamma1Out]
        

    def Check(self, mode) -> bool:

        # Determining if rank computational results are ready
        completed = MPI.Request.testall(self.requests[mode])
        if not completed[0]:
            # print('Not done')
            return False

        # Loading rank computational results
        new_charge, new_Lambda, Gamma0Out, Gamma1Out = self.requests_buf[mode]
        _, _, _, _, svd_time, theta_time = completed[1]
        self.svd_time += svd_time
        self.theta_time += theta_time

        # Update charges (modifying CC modifies self.dcharge by pointer)
        self.charge[mode + 1] = new_charge
        self.Lambda[mode] = new_Lambda
        if mode == self.n_modes - 2:
            self.Gamma[self.n_modes - 2, :, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2)] = Gamma0Out[:, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2)]
            self.Gamma[self.n_modes - 1, :min(self.bond_dimension, self.local_hilbert_space_dimension ** 2), 0] = Gamma1Out[:min(self.bond_dimension, self.local_hilbert_space_dimension ** 2), 0]
        else:
            self.Gamma[mode, :, :] = Gamma0Out
            self.Gamma[mode + 1, :, :] = Gamma1Out

        return True  


    def update_rank_status(self):
        new_running_l_and_rank = []
        for l, rank in self.running_l_and_rank:
            if self.Check(l):
                self.available_ranks.append(rank)
            else:
                new_running_l_and_rank.append([l, rank])
        self.running_l_and_rank = new_running_l_and_rank

    def LayerUpdate(self, layer):

        start = time.time()
        for i, mode in enumerate(range(layer % 2, self.n_modes - 1, 2)):
            if mode >= self.n_input_states + layer:
                continue
            reflectivity = self.reflectivity[layer, i]
            seed = self.seeds[layer, i]
            while len(self.available_ranks) == 0:
                self.update_rank_status()
                time.sleep(0.01)
            target_rank = self.available_ranks.pop(0)
            self.Request(seed, mode, reflectivity, target_rank)
            self.running_l_and_rank.append([mode, target_rank])
        while len(self.available_ranks) != self.num_gpu_ranks:
            self.update_rank_status()
            time.sleep(0.01)
        self.update_time += time.time() - start


    def Simulate(self):

        success = True

        if self.initialization_successful:
            self.EE[:, 0] = EntanglementEntropy(self.Lambda)
            full_start = time.time()

            for layer in range(self.n_modes):
                self.LayerUpdate(layer)
                start = time.time()
                self.prob[layer+1] = Probability(self.n_modes, self.local_hilbert_space_dimension, self.Gamma, self.Lambda, self.charge)
                self.EE[:, layer+1] = EntanglementEntropy(self.Lambda)
                # prob = self.prob[np.where(self.prob > 0)[0]]
                # if np.max(prob)/np.min(prob) - 1 > 0.1:
                #     print('Accuracy too low. Failed.')
                #     success = False
                #     break
                # max_ee_idx = np.argmax(self.EE)
                # if layer - max_ee_idx % self.EE.shape[1] > 10:
                #     print('EE did not increase for 10 epochs. Stopping')
                #     break
                self.ee_prob_cal_time += time.time() - start
                '''Initialial total time is much higher than simulation time due to initialization of cuda context.'''
                print("Total time: {:.2f}. Update time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. EE_Prob_cal_time: {:.2f}".format(time.time() - full_start, self.update_time, self.theta_time, self.svd_time, self.ee_prob_cal_time))

            while len(self.available_ranks) != self.num_gpu_ranks:
                self.update_rank_status()
                time.sleep(0.01)

        for target_rank in self.available_ranks:
            comm.send('Finished', target_rank, tag=100)

        return success, self.prob, self.EE, self.REPar