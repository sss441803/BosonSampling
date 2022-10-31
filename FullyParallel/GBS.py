'''Full simulation code containing the Device method (cupy, unified update)'''
import argparse
import os
import time
from math import sinh, sqrt
# import sys

import cupy as cp
import numpy as np
from qutip import squeeze, thermal_dm

from Full_GBS_MPO import FullMPO
from Node_GBS_MPO import NodeMPO
from Rank_GBS_MPO import RankMPO

from mpi4py import MPI


def FullMultiCycle(nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n])
    EEAvg = np.zeros([n - 1, n])
    REAvg = np.zeros([n - 1, n, 5])

    TotalProbTot = np.zeros([n])
    EETot = np.zeros([n - 1, n])
    RETot = np.zeros([n - 1, n, 5])

    boson = FullMPO(nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol, PS)
    Totprob, EE, RE = boson.FullUpdate()
    TotalProbTot += Totprob;#TotalProbPar[:,i];
    EETot += EE;#EEPar[:,:,i];
    RETot += RE;#EEPar[:,:,i];
    
    TotalProbAvg = TotalProbTot
    EEAvg = EETot
    REAvg = RETot

    return TotalProbAvg,  EEAvg, REAvg


def NodeMultiCycle(node_gpu_ranks, d, chi):
    boson = NodeMPO(node_gpu_ranks, d, chi)
    boson.Nodeloop()
    # print('Slave loop finished')

def RankMultiCycle(node_control_rank, d, chi):
    boson = RankMPO(node_control_rank, d, chi)
    boson.FullRankloop()


def PS_dist(n, r, loss):
    am = (1 - loss) * np.exp(- 2 * r) + loss
    ap = (1 - loss) * np.exp(2 * r) + loss
    s = 1 / 4 * np.log(ap / am)
    n_th = 1 / 2 * (np.sqrt(am * ap) - 1)
    nn = 40
    single_dist = np.array(np.diag(squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()), dtype = float)
    prob_dist = np.array(np.diag(squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()), dtype = float)
    for i in range(n - 1):
        prob_dist = np.convolve(prob_dist, single_dist)
    return prob_dist


# def main():

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=0)
parser.add_argument('--ranks_per_node', type=int, default=0)
parser.add_argument('--gpus_per_node', type=int, default=0)
parser.add_argument('--ranks_per_gpu', type=int, default=0)
parser.add_argument('--n', type=int, help='Number of modes.')
parser.add_argument('--m', type=int, help='Number of squeezed states. One state per mode from the left.')
args = vars(parser.parse_args())

nodes = args['nodes']
ranks_per_node = args['ranks_per_node']
gpus_per_node = args['gpus_per_node']
ranks_per_gpu = args['ranks_per_gpu']
assert ranks_per_node == ranks_per_gpu * gpus_per_node + 2, 'Arguments specifying number of ranks per node/gpu, gpus per node are incompatible. Expecting ranks_per_node == ranks_per_gpu * gpus_per_node + 2. One additional rank for node level beam splitter update, and  one additional rank for overall management.'
n = args['n']
m = args['m']

comm = MPI.COMM_WORLD
ranks = comm.Get_size()
rank = comm.Get_rank()
assert ranks == nodes * ranks_per_node, 'Number of ranks for MPI is not compatible with the script arguments. Should be nodes * ranks_per_node'
gpu_ranks_per_node = ranks_per_gpu * gpus_per_node
node = rank // ranks_per_node
node_gpu_ranks = [i + 2 for i in range(ranks_per_node * node, ranks_per_node * node + gpu_ranks_per_node)]
node_control_rank = node * ranks_per_node + 1

if rank in node_gpu_ranks:
    gpu = (rank%ranks_per_node-2)%gpus_per_node
    cp.cuda.Device(gpu).use()
    print('rank {} using gpu {}'.format(rank, gpu))

t0 = time.time()

errtol = 10 ** (-7)

for i in range(1):
    for beta in [1.2]:
        for r in [1.144]:
            ideal_ave_photons = m#*sinh(r)**2
            lossy_ave_photons = beta*sqrt(ideal_ave_photons)
            loss = round(100*(1 - lossy_ave_photons/ideal_ave_photons))/100
            PS = int((1-loss)*m*sinh(r)**2); d = PS+1; init_chi = d**2
            chi = int(max(32*2**PS, d**2, 128))

            if rank == 0:
                print('m is ',  m, ', d is ', d, ', r is ', r, ', beta is ', beta, ', chi is ', chi)
            if chi > 8200:
                if rank == 0:
                    print('Too large')
                continue
            
            begin_dir = './EE_vs_modes/n_{}_m_{}_beta_{}_loss_{}_chi_{}_r_{}_PS_{}'.format(n, m, beta, loss, chi, r, PS)
            if not os.path.isdir(begin_dir) and rank == 0:
                os.makedirs(begin_dir)

            if not os.path.isfile(begin_dir + '/EE_{}.npy'.format(id)):
                if rank == 0:
                    Totprob, EE, RE = FullMultiCycle(nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol, PS)
                    print(Totprob)
                    print(EE)
                    np.save(begin_dir + '/EE_{}.npy'.format(id), EE)
                    np.save(begin_dir + '/Totprob_{}.npy'.format(id), Totprob)
                    print("Time cost", time.time() - t0)

                elif rank == node_control_rank:
                    NodeMultiCycle(node_gpu_ranks, d, chi)

                elif rank in node_gpu_ranks:
                    RankMultiCycle(node_control_rank, d, chi)

                else:
                    print('invalid rank ', rank)
                    quit()

            else:
                if rank == 0:
                    print("Simulation already ran.")
    m += 4


# if __name__ == "__main__":
#     # def mpiabort_excepthook(type, value, traceback):
#     #     comm.Abort()
#     #     sys.__excepthook__(type, value, traceback)
#     # # sys.excepthook = mpiabort_excepthook
#     main()
#     # sys.excepthook = sys.__excepthook__