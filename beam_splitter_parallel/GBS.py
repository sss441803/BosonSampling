'''Full simulation code containing the Device method (cupy, unified update)'''
import argparse
import os
import time
from math import sinh, sqrt
from filelock import FileLock
import pickle
import sys

import cupy as cp
import numpy as np
np.set_printoptions(3)
from qutip import squeeze, thermal_dm

from Master_GBS_MPO import MasterMPO
from Slave_GBS_MPO import SlaveMPO

from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size() - 1
# num_gpu_ranks = num_gpu_ranks // 16 * 4
rank = comm.Get_rank()
# print(rank)
# gpu = (rank%17-1)%num_gpus_per_node
gpu = rank % 4
# if rank % 17 != 0:
cp.cuda.Device(gpu).use()
print('rank {} using gpu {}'.format(rank, gpu))


def MasterMultiCycle(num_gpu_ranks, n, m, d, r, loss, init_chi, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n+1])
    EEAvg = np.zeros([n - 1, n+1])
    REAvg = np.zeros([n - 1, n+1, 5])

    TotalProbTot = np.zeros([n+1])
    EETot = np.zeros([n - 1, n+1])
    RETot = np.zeros([n - 1, n+1, 5])

    boson = MasterMPO(num_gpu_ranks, n, m, d, r, loss, init_chi, chi, errtol, PS)
    Totprob, EE, RE = boson.FullUpdate()
    TotalProbTot += Totprob;#TotalProbPar[:,i];
    EETot += EE;#EEPar[:,:,i];
    RETot += RE;#EEPar[:,:,i];
    
    TotalProbAvg = TotalProbTot
    EEAvg = EETot
    REAvg = RETot

    return TotalProbAvg,  EEAvg, REAvg


def SlaveMultiCycle(n, d, chi):
    boson = SlaveMPO(n, d, chi)
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


def main():

    exp_idx_beginning = 0
    while True:
    # Loop until all experiments are over
        if rank == 0:
            with FileLock("../experiment.pickle.lock"):
                # work with the file as it is now locked
                print("Lock acquired.")
                with open("../experiment.pickle", 'rb') as experiment_file:
                    experiments = pickle.load(experiment_file)
                found_experiment = False
                for exp_idx in range(exp_idx_beginning, len(experiments)):
                    experiment = experiments[exp_idx]
                    if experiment['status'] == 'incomplete':
                        found_experiment = True
                        print('Found experiment: ', experiment)
                        # Break the loop once an incomplete experiment is found
                        break
                exp_idx_beginning = exp_idx + 1
                if not found_experiment:
                    # If loop never broke, no experiment was found
                    print('All experiments already ran. Exiting.')
                    comm.Abort()
                else:
                    # experiment = {'n': 32, 'm': 9, 'beta': 1, 'r': 1.44, 'PS': 5, 'd': 6, 'status': 'in progress'}
                    n, m, beta, r, PS, d = experiment['n'], experiment['m'], experiment['beta'], experiment['r'], experiment['PS'], experiment['d']
                    ideal_ave_photons = m*sinh(r)**2
                    lossy_ave_photons = beta*sqrt(ideal_ave_photons)
                    loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
                    init_chi = d**2
                    chi = int(max(32*2**PS, d**2, 512))
                    begin_dir = '../NewResults/n_{}_m_{}_beta_{}_loss_{}_r_{}_PS_{}_'.format(n, m, beta, loss, r, PS)
                    if os.path.isfile(begin_dir + 'chi.npy'):
                        chi_array = np.load(begin_dir + 'chi.npy')
                        chi = int(np.max(chi_array))
                        prob = np.load(begin_dir + 'chi_{}_Totprob.npy'.format(chi))
                        prob = prob[np.where(prob != 0)[0]]
                        print('prob: ', prob)
                        if min(prob) != 0:
                            error = np.max(prob)/np.min(prob) - 1
                        error = np.max(error)
                        print('error: ', error)
                        if error > 0.1:
                            chi *= 2
                            print('chi was too small producing error {}. Increasing chi to {}'.format(error, chi))
                            status = 'run'
                        else:
                            print('Simulation with suitable accuracy already ran.')
                            status = 'skip'
                    else:
                        status = 'run'
                    
                    if status == 'run':
                        if chi > 10000:
                            print('Required bond-dimension chi too large. Moving on to next experiment.')
                            status = 'skip'
                        else:
                            # Will run the first found incomplete experiment, set status to in progress
                            experiments[exp_idx]['status'] = 'in progress'
                            # Update experiment track file
                            with open('../experiment.pickle', 'wb') as file:
                                pickle.dump(experiments, file)
                            status = 'run'
        else:
            status = n = m = d = beta = loss = chi = r = PS = None

        # Broadcasting to all ranks
        status, n, m, d, beta, loss, chi, r, PS = comm.bcast([status, n, m, d, beta, loss, chi, r, PS], root=0)
        if status == 'skip':
            continue

        t0 = time.time()
        errtol = 10 ** (-7)   

        begin_dir = '../NewResults/n_{}_m_{}_beta_{}_loss_{}_r_{}_PS_{}_'.format(n, m, beta, loss, r, PS)

        id = 0

        if not os.path.isfile(begin_dir + 'EE.npy'):
            if rank == 0:
                t0 = time.time()
                Totprob, EE, RE = MasterMultiCycle(num_gpu_ranks, n, m, d, r, loss, init_chi, chi, errtol, PS)
                print(Totprob)
                print(EE)
                # Saving results
                if os.path.isfile(begin_dir + 'chi.npy'):
                    chi_array = np.load(begin_dir + 'chi.npy')
                else:
                    chi_array = np.array([])
                assert not np.sum(chi_array == chi), 'chi {} already in chi array'.format(chi)
                chi_array = np.append(chi_array, chi)
                prob_file = begin_dir + 'chi_{}_Totprob.npy'.format(chi)
                EE_file = begin_dir + 'chi_{}_EE.npy'.format(chi)
                assert not os.path.isfile(prob_file), '{} exists already. Error.'.format(prob_file)
                assert not os.path.isfile(EE_file), '{} exists already. Error.'.format(EE_file)
                np.save(prob_file, Totprob)
                np.save(EE_file, EE)
                np.save(begin_dir + 'chi.npy', chi_array)
                print("Time cost", time.time() - t0)
            # elif rank % 5 != 0:
            elif rank != 0:
                SlaveMultiCycle(n, d, chi)

        else:
            if rank == 0:
                print("Simulation already ran.")


if __name__ == "__main__":
    def mpiabort_excepthook(type, value, traceback):
        print('type: ', type)
        print('value: ', value)
        print('traceback: ', traceback)
        print('An exception occured. Aborting MPI')
        comm.Abort()
    sys.excepthook = mpiabort_excepthook
    main()
    # sys.excepthook = sys.__excepthook__