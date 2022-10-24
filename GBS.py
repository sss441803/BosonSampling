'''Full simulation code containing the Device method (cupy, unified update)'''
import argparse
import os
import time
from math import sinh, sqrt

import cupy as cp
import numpy as np
from qutip import squeeze, thermal_dm

from Master_GBS_MPO import MasterMPO
from Slave_GBS_MPO import SlaveMPO

from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size()
num_gpu_ranks = num_gpu_ranks // 5 * 4
rank = comm.Get_rank()
# print(rank)
gpu = (rank%5-1)%num_gpus_per_node
if rank % 5 != 0:
    cp.cuda.Device(gpu).use()
    print('rank {} using gpu {}'.format(rank, gpu))
#     print(rank)
#     Rand_U(1,0.1)
#     glc_obj = Aligner.make_data_obj('glc', True, cp.zeros([8,8],dtype='complex64'), [0, 0])
#     gcr_obj = Aligner.make_data_obj('glc', True, cp.zeros([8,8],dtype='complex64'), [0, 0])
#     cl_obj = Aligner.make_data_obj('cl', True, cp.zeros([8,2], dtype='int32'), [0])
#     cr_obj = Aligner.make_data_obj('cl', True, cp.zeros([8,2], dtype='int32'), [0])
#     change_charges_C = cp.zeros([2,1], dtype='int32')
#     change_idx_C = cp.zeros([2,1], dtype='int32')
#     # update_MPO(1, cp.zeros(8, dtype='int32'), cp.zeros(8, dtype='int32'), cp.zeros([1,1],dtype='float32'), cp.zeros([1,1],dtype='float32'), glc_obj, gcr_obj, cl_obj, cr_obj, change_charges_C, change_idx_C)
#     print('rank {} successful'.format(rank))

# quit()

def MasterMultiCycle(num_ranks, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n+1])
    EEAvg = np.zeros([n - 1, n+1])
    REAvg = np.zeros([n - 1, n+1, 5])

    TotalProbTot = np.zeros([n+1])
    EETot = np.zeros([n - 1, n+1])
    RETot = np.zeros([n - 1, n+1, 5])

    boson = MasterMPO(num_ranks, n, m, d, r, loss, init_chi, chi, errtol, PS)
    Totprob, EE, RE = boson.FullUpdate()
    TotalProbTot += Totprob;#TotalProbPar[:,i];
    EETot += EE;#EEPar[:,:,i];
    RETot += RE;#EEPar[:,:,i];
    
    TotalProbAvg = TotalProbTot
    EEAvg = EETot
    REAvg = RETot

    return TotalProbAvg,  EEAvg, REAvg


def SlaveMultiCycle(d, chi):
    boson = SlaveMPO(d, chi)
    boson.Slaveloop()
    # print('Slave loop finished')


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


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help="Which GPU", default=0)
parser.add_argument('--id', type=int, help="ID of the file to generate corresponding to task number")
parser.add_argument('--n', type=int, help='Number of modes.')
parser.add_argument('--m', type=int, help='Number of squeezed states. One state per mode from the left.')
parser.add_argument('--loss', type=float, help='Photon loss rate.')
# parser.add_argument('--chi', type=int, help='Maximum allowed bond dimension')
parser.add_argument('--r', type=float, help='Squeezing parameter.')
args = vars(parser.parse_args())

gpu = args['gpu']
id = args['id']
n = args['n']
m = args['m']
loss = args['loss']
# chi = args['chi']
r = args['r']

# n = 6
# m = 2

t0 = time.time()

errtol = 10 ** (-7)
# # PS = m; d = PS + 1; chi = 8 * 2**m; init_chi = d**2
# prob_dist = PS_dist(m, r, loss)
# cum_prob = 0
# i = 0
# while cum_prob < 0.99:
#     cum_prob += prob_dist[i]
#     i += 1

# print(i)

for i in range(1):
    for beta in [1.2]:
        for r in [1.44]:
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
            if not os.path.isdir(begin_dir):
                os.makedirs(begin_dir)

            if not os.path.isfile(begin_dir + '/EE_{}.npy'.format(id)):
                if rank == 0:
                    Totprob, EE, RE = MasterMultiCycle(num_gpu_ranks, n, m, d, r, loss, init_chi, chi, errtol, PS)
                    print(Totprob)
                    print(EE)
                    
                    np.save(begin_dir + '/EE_{}.npy'.format(id), EE)
                    np.save(begin_dir + '/Totprob_{}.npy'.format(id), Totprob)

                    print("Time cost", time.time() - t0)
                else:
                    SlaveMultiCycle(d, chi)
            else:
                if rank == 0:
                    print("Simulation already ran.")
    m += 4

    
    # max_d = 11
        
    # for PS in range(max_d):
    #     if PS%2 == 1 and loss == 0:
    #         continue
    #     errtol = 10 ** (-7) #/ prob_dist[PS]
    #     d = PS + 1; chi = max(2**(PS+2*m-3), d**2, 128); init_chi = d**2
    #     if chi > 8200:
    #         continue
    #     print('m is ', m, ', d is ', d, ', errtol is ', errtol, ', chi is ', chi)
        
    #     begin_dir = './results/n_{}_m_{}_loss_{}_r_{}/chi_{}_PS_{}'.format(n, m, loss, r, chi, PS)
    #     if not os.path.isdir(begin_dir):
    #         os.makedirs(begin_dir)

    #     if not os.path.isfile(begin_dir + '/EE_{}.npy'.format(id)):
    #         Totprob, EE, RE = RCS1DMultiCycleAvg(n, m, d, r, loss, init_chi, chi, errtol, PS)
    #         print(Totprob)
    #         # print(EE)
            
    #         np.save(begin_dir + '/EE_{}.npy'.format(id), EE)
    #         np.save(begin_dir + '/Totprob_{}.npy'.format(id), Totprob)

    #         print("Time cost", time.time() - t0)
    #     else:
    #         print("Simulation already ran.")