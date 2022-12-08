'''Full simulation code containing the Device method (cupy, unified update)'''
import argparse
import pickle
import time
import os
import sys
from math import sinh, sqrt
from filelock import FileLock

import cupy as cp
import numpy as np
from qutip import squeeze, thermal_dm

from Full_GBS_MPO import FullCompute
from Node_storage import NodeData
from Node_GBS_MPO import NodeCompute
from Rank_GBS_MPO import RankCompute

from mpi4py import MPI


def FullComputeCycle(nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n + 1])
    EEAvg = np.zeros([n - 1, n + 1])
    REAvg = np.zeros([n - 1, n + 1, 5])

    TotalProbTot = np.zeros([n + 1])
    EETot = np.zeros([n - 1, n + 1])
    RETot = np.zeros([n - 1, n + 1, 5])

    boson = FullCompute(nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol, PS)
    Totprob, EE, RE = boson.FullUpdate()
    TotalProbTot += Totprob;#TotalProbPar[:,i];
    EETot += EE;#EEPar[:,:,i];
    RETot += RE;#EEPar[:,:,i];
    
    TotalProbAvg = TotalProbTot
    EEAvg = EETot
    REAvg = RETot

    return TotalProbAvg,  EEAvg, REAvg

def NodeDataCycle(nodes, ranks_per_node, this_node, n, d, chi):
    process = NodeData(nodes, ranks_per_node, this_node, n, d, chi)
    process.NodeDataLoop()

def NodeComputeCycle(nodes, this_node, ranks_per_node, node_gpu_ranks, n, d, chi):
    process = NodeCompute(nodes, this_node, ranks_per_node, node_gpu_ranks, n, d, chi)
    process.NodeComputeLoop()

def RankComputeCycle(node_control_rank, d, chi):
    process = RankCompute(node_control_rank, d, chi)
    process.FullRankloop()
    return process.start_timestamps, process.end_timestamps


def PS_dist(n, r, loss):
    am = (1 - loss) * np.exp(- 2 * r) + loss
    ap = (1 - loss) * np.exp(2 * r) + loss
    s = 1 / 4 * np.log(ap / am)
    n_th = 1 / 2 * (np.sqrt(am * ap) - 1)
    nn = 40
    single_dist = np.array(np.diag(squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()), dtype = float)
    prob_dist = np.array(np.diag(squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()), dtype = float)
    for _ in range(n - 1):
        prob_dist = np.convolve(prob_dist, single_dist)
    return prob_dist


parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=0)
parser.add_argument('--ranks_per_node', type=int, default=0)
parser.add_argument('--gpus_per_node', type=int, default=0)
parser.add_argument('--ranks_per_gpu', type=int, default=0)
args = vars(parser.parse_args())

nodes = args['nodes']
ranks_per_node = args['ranks_per_node']
gpus_per_node = args['gpus_per_node']
ranks_per_gpu = args['ranks_per_gpu']
assert ranks_per_node == ranks_per_gpu * gpus_per_node + 2, 'Arguments specifying number of ranks per node/gpu, gpus per node are incompatible. Expecting ranks_per_node == ranks_per_gpu * gpus_per_node + 2. One additional rank for node level beam splitter update, and  one additional rank for overall management.'

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


def main():

    while True:
    # Loop until all experiments are over
        if rank == 0:
            print('rank 0')
            with FileLock("experiment.pickle.lock"):
                # work with the file as it is now locked
                print("Lock acquired.")
                with open("experiment.pickle", 'rb') as experiment_file:
                    experiments = pickle.load(experiment_file)
                for exp_idx in range(len(experiments)):
                    experiment = experiments[exp_idx]
                    if experiment['status'] == 'incomplete':
                        print('1')
                        # Will run the first found incomplete experiment, set status to in progress
                        experiments[exp_idx]['status'] = 'in progress'
                        # Update experiment track file
                        with open('experiment.pickle', 'wb') as file:
                            pickle.dump(experiments, file)
                        # Break the loop once an incomplete experiment is found
                        break
            if exp_idx == len(experiments) - 1:
                # If loop never broke, no experiment was found
                print('All experiments already ran. Exiting.')
                status = 'Exiting'
            else:
                experiment = {'n': 32, 'm': 9, 'beta': 1, 'r': 1.44, 'PS': 5, 'd': 6, 'status': 'in progress'}
                print('Running experiment: ', experiment)
                n, m, beta, r, PS, d = experiment['n'], experiment['m'], experiment['beta'], experiment['r'], experiment['PS'], experiment['d']
                status = 'Run Experiment'
        else:
            status = n = m = beta = r = PS = d = None

        # Broadcasting to all ranks
        status = comm.bcast(status, root=0)
        if status == 'Exiting':
            comm.Abort()
        elif status == 'Run Experiment':
            n, m, beta, r, PS, d = comm.bcast([n, m, beta, r, PS, d], root=0)
        else:
            print('Invalid status {} received by rank {}.'.format(status, rank))
            comm.Abort()

        t0 = time.time()
        errtol = 10 ** (-7)

        ideal_ave_photons = m*sinh(r)**2
        lossy_ave_photons = beta*sqrt(ideal_ave_photons)
        loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
        init_chi = d**2
        chi = int(max(32*2**PS, d**2, 128))
        # chi = 128

        if chi > 20000:
            if rank == 0:
                print('Required bond-dimension chi too large. Moving on to next experiment.')
            continue
        
        begin_dir = './results/n_{}_m_{}_beta_{}_loss_{}_chi_{}_r_{}_PS_{}'.format(n, m, beta, loss, chi, r, PS)
        if not os.path.isdir(begin_dir) and rank == 0:
            os.makedirs(begin_dir)

        id = 0
        if not os.path.isfile(begin_dir + '/EE_{}.npy'.format(id)):
        # if True:
            if rank == 0:
                print('rank {} full compute'.format(rank))
                Totprob, EE, RE = FullComputeCycle(nodes, ranks_per_node, n, m, d, r, loss, init_chi, chi, errtol, PS)
                print(Totprob)
                print(EE)
                np.save(begin_dir + '/EE_{}.npy'.format(id), EE)
                np.save(begin_dir + '/Totprob_{}.npy'.format(id), Totprob)
                print("Time cost", time.time() - t0)

            elif rank == node_control_rank:
                print('rank {} node compute'.format(rank))
                NodeComputeCycle(nodes, node, ranks_per_node, node_gpu_ranks, n, d, chi)
            
            elif rank % ranks_per_node == 0:
                print('rank {} node data'.format(rank))
                NodeDataCycle(nodes, ranks_per_node, node, n, d, chi)

            elif rank in node_gpu_ranks:
                print('rank {} rank compute'.format(rank))
                _, _ = RankComputeCycle(node_control_rank, d, chi)

            else:
                print('invalid rank ', rank)
                comm.Abort()

        else:
            if rank == 0:
                print("Simulation already ran.")
            print('rank {}'.format(rank))
        
        if d > 9:
            comm.Abort()


if __name__ == "__main__":
    def mpiabort_excepthook(type, value, traceback):
        # sys.__excepthook__(type, value, traceback)
        print('An exception occured. Aborting MPI')
        comm.Abort()
    sys.excepthook = mpiabort_excepthook
    main()