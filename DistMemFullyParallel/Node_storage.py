import time

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


data_type = np.complex64
float_type = np.float32
int_type = np.int32



class NodeData:

    def __init__(self, nodes, this_node, n, d, chi):
        self.n = n
        self.d = d
        self.chi = chi
        self.nodes = nodes
        self.this_node = this_node
        self.running_charges_and_rank = []
        self.sides = [None for _ in range(self.n - 1)]
        self.requests = [None for _ in range(self.n - 1)]
        self.requests_buf = [None for _ in range(self.n - 1)]

        # Gammas = [[] for _ in range(self.n // self.nodes + (self.this_node < self.n % self.nodes))]
        # Lambdas = [[] for _ in range((self.n - 1) // self.nodes + (self.this_node < (self.n - 1) % self.nodes))]
        # Charges = [[] for _ in range((self.n + 1) // self.nodes + (self.this_node < (self.n + 1) % self.nodes))]
        self.Gammas = []
        self.Lambdas= []
        self.Lambda_edge = np.ones(chi, dtype = 'float32')
        self.Charges = []
        for site in range(self.n + 1):
            # print('Node: node {} site {}'.format(this_node, site))
            node = site % self.nodes
            if node != self.this_node:
                continue
            '''Gammas only go to self.n'''
            if site < self.n:
                site_gamma = np.empty([self.chi, self.chi], dtype = 'complex64')
                comm.Recv([site_gamma, MPI.C_FLOAT_COMPLEX], 0, tag=1)
                self.Gammas.append(site_gamma)
            '''Lambdas only go to self.n - 1'''
            if site < self.n - 1:
                site_Lambda = np.empty(self.chi, dtype = 'float32')
                comm.Recv([site_Lambda, MPI.FLOAT], 0, tag=2)
                self.Lambdas.append(site_Lambda)
            '''Charges go to self.n + 1'''
            site_charge = np.empty([self.chi, 2], dtype = 'int32')
            comm.Recv([site_charge, MPI.INT], 0, tag=3)
            self.Charges.append(site_charge)


    def NodeProcess(self):

        l = comm.recv(source=0, tag=0)
        side = comm.recv(source=0, tag=1)
        compute_rank = comm.recv(source=0, tag=2)
        # print('Data node rank {} received l {}, side {}, compute_rank {}'.format(rank, l, side, compute_rank))
        self.sides[l] = side

        if side == 0:
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
            new_Lambda_req = comm.Irecv(new_Lambda, source=compute_rank, tag=11)
            Gamma0Out_req = comm.Irecv(Gamma0Out, source=compute_rank, tag=12)
            self.requests[l] = [new_Lambda_req, Gamma0Out_req]
            self.requests_buf[l] = [new_Lambda, Gamma0Out]
        if side == 1:
            if l != self.n - 2:
                LR = self.Lambdas[(l + 1) // self.nodes]
            else:
                LR = self.Lambda_edge
            CC = self.Charges[(l + 1) // self.nodes]
            Gcr = self.Gammas[(l + 1) // self.nodes]
            comm.Isend([LR, MPI.FLOAT], compute_rank, tag=1)
            comm.Isend([CC, MPI.INT], compute_rank, tag=3)
            comm.Isend([Gcr, MPI.C_FLOAT_COMPLEX], compute_rank, tag=6)
            new_charge = np.empty([self.chi, 2], dtype='int32')
            Gamma1Out = np.zeros([self.chi, self.chi], dtype='complex64')
            new_charge_req = comm.Irecv(new_charge, source=compute_rank, tag=10)
            Gamma1Out_req = comm.Irecv(Gamma1Out, source=compute_rank, tag=13)
            self.requests[l] = [new_charge_req, Gamma1Out_req]
            self.requests_buf[l] = [new_charge, Gamma1Out]
        if side == 2:
            CR = self.Charges[(l + 2) // self.nodes]
            comm.Isend([CR, MPI.INT], compute_rank, tag=4)
        
        # print('Node data data sent for site ', l)

    
    
    def NodeDataLoop(self):

        status_req = comm.irecv(source=0, tag=100)
        completed = MPI.Request.test(status_req)
        while not completed[0]:
            self.check()
            time.sleep(0.01)
            completed = MPI.Request.test(status_req)
        status = completed[1]
        # print('rank: {}, status: {}'.format(rank, status))
        while status != 'Finished':
            if status == 'Data needed':
                self.NodeProcess()
            elif status == 'Layer finished':
                self.complete_all()
            else:
                raise Exception("Not a valid status from full_gbs_mpo.")
            status_req = comm.irecv(source=0, tag=100)
            completed = MPI.Request.test(status_req)
            # print('rank {} checking'.format(rank))
            while not completed[0]:
                self.check()
                time.sleep(0.01)
                # print('checked')
                completed = MPI.Request.test(status_req)
            # print('rank {} done checking'.format(rank))
            status = completed[1]
            # print('rank: {}, status: {}'.format(rank, status))


    def check(self):
        for l, req, buf in zip(range(self.n - 1), self.requests, self.requests_buf):
            # print(l, len(buf))
            if buf != None:
                # print('checking site ', l)
                completed = MPI.Request.testall(req)
                if completed[0]:
                    # Loading compute node results if updates data stored on node 0
                    side = self.sides[l]
                    if side == 0:
                        new_Lambda, Gamma0Out = buf
                        self.Lambdas[l // self.nodes] = new_Lambda
                        if l == self.n - 2:
                            self.Gammas[l // self.nodes][:, :min(self.chi, self.d ** 2)] = Gamma0Out[:, :min(self.chi, self.d ** 2)]
                        else:
                            self.Gammas[l // self.nodes][:, :] = Gamma0Out
                    if side == 1:
                        new_charge, Gamma1Out = buf
                        self.Charges[(l + 1) // self.nodes] = new_charge
                        if l == self.n - 2:
                            self.Gammas[(l + 1) // self.nodes][:min(self.chi, self.d ** 2), 0] = Gamma1Out[:min(self.chi, self.d ** 2), 0]
                        else:
                            self.Gammas[(l + 1) // self.nodes] = Gamma1Out
                    self.requests_buf[l] = None

    def complete_all(self):
        while not self.is_all_buf_clear():
            self.check()

    def is_all_buf_clear(self):
        for buf in self.requests_buf:
            if buf != None:
                return False
        return True