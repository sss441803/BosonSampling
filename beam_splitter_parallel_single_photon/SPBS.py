'''Full simulation code containing the Device method (cupy, unified update)'''
import os
import time
from filelock import FileLock
import pickle
import sys

import numpy as np
np.set_printoptions(3)
from qutip import squeeze, thermal_dm

from Master_SPBS_MPO import MasterMPO
from Slave_SPBS_MPO import SlaveMPO

from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size() - 1
# num_gpu_ranks = num_gpu_ranks // 16 * 4
rank = comm.Get_rank()
# print(rank)
# gpu = (rank%17-1)%num_gpus_per_node
# if rank % 17 != 0:


def MasterMultiCycle(num_gpu_ranks, n, m, d, loss, chi, errtol = 10 ** (-6), PS=None):

    boson = MasterMPO(num_gpu_ranks, n, m, d, loss, chi, errtol, PS)
    Totprob, EE, RE = boson.FullUpdate()

    return Totprob, EE, RE


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
        # if rank == 0:
        #     with FileLock("../experiment.pickle.lock"):
        #         # work with the file as it is now locked
        #         print("Lock acquired.")
        #         with open("../experiment.pickle", 'rb') as experiment_file:
        #             experiments = pickle.load(experiment_file)
        #         found_experiment = False
        #         for exp_idx in range(exp_idx_beginning, len(experiments)):
        #             experiment = experiments[exp_idx]
        #             if experiment['status'] == 'incomplete':
        #                 found_experiment = True
        #                 print('Found experiment: ', experiment)
        #                 # Break the loop once an incomplete experiment is found
        #                 break
        #         exp_idx_beginning = exp_idx + 1
        #         if not found_experiment:
        #             # If loop never broke, no experiment was found
        #             print('All experiments already ran. Exiting.')
        #             comm.Abort()
        #         else:
        #             n, m, beta, loss, d = experiment['n'], experiment['m'], experiment['beta'], experiment['loss'], experiment['d']
        #             chi = int(max(16*2**(m+1), 2048))
        #             begin_dir = '../SPBSResults/n_{}_m_{}_beta_{}_loss_{}_'.format(n, m, beta, loss)
        #             if os.path.isfile(begin_dir + 'chi.npy'):
        #                 chi_array = np.load(begin_dir + 'chi.npy')
        #                 chi = int(np.max(chi_array))
        #                 prob = np.load(begin_dir + 'chi_{}_Totprob.npy'.format(chi))
        #                 prob = prob[np.where(prob > 0)[0]]
        #                 print('prob: ', prob)
        #                 if min(prob) != 0:
        #                     error = np.max(prob)/np.min(prob) - 1
        #                 error = np.max(error)
        #                 print('error: ', error)
        #                 if error > 0.1:
        #                     chi *= 2
        #                     print('chi was too small producing error {}. Increasing chi to {}'.format(error, chi))
        #                     status = 'run'
        #                 else:
        #                     print('Simulation with suitable accuracy already ran.')
        #                     status = 'skip'
        #             else:
        #                 status = 'run'

        #             print('Loss: {}. Chi: {}'.format(loss, chi))
                    
        #             if status == 'run':
        #                 if chi > 5000:
        #                     print('Required bond-dimension chi too large. Moving on to next experiment.')
        #                     status = 'skip'
        #                 elif n > 100:
        #                     print('Too many modes. Moving on to next experiment.')
        #                     status = 'skip'
        #                 else:
        #                     # Will run the first found incomplete experiment, set status to in progress
        #                     experiments[exp_idx]['status'] = 'in progress'
        #                     # Update experiment track file
        #                     with open('../experiment.pickle', 'wb') as file:
        #                         pickle.dump(experiments, file)
        #                     status = 'run'
        # else:
        #     status = n = m = d = beta = loss = chi = None

        # # Broadcasting to all ranks
        # status, n, m, d, beta, loss, chi = comm.bcast([status, n, m, d, beta, loss, chi], root=0)
        # if status == 'skip':
        #     continue

        n, m, beta, errtol = 10, 5, 1.0, 10**(-7)
        ideal_ave_photons = beta * m
        lossy_ave_photons = 0.5 * ideal_ave_photons
        loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
        PS = int((1-loss)*m)
        PS += 1
        d = PS + 1
        PS = None
        d = m+1
        init_chi = d**2
        chi = int(max(4*2**d, d**2, 512))
        # chi = 128
        print(n, m, beta, loss, PS, d, chi)

        t0 = time.time()
        errtol = 10 ** (-7)   

        begin_dir = '../NewResults/n_{}_m_{}_beta_{}_loss_{}_'.format(n, m, beta, loss)

        if not os.path.isfile(begin_dir + 'EE.npy'):
            if rank == 0:
                t0 = time.time()
                Totprob, EE, RE = MasterMultiCycle(num_gpu_ranks, n, m, m+1, loss, chi, errtol, PS)
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
                # if np.sum(Totprob > 0) > 0:
                #     np.save(prob_file, Totprob)
                #     np.save(EE_file, EE)
                #     np.save(begin_dir + 'chi.npy', chi_array)
                # else:
                #     print('Results invalid. Not saving.')
                print("Time cost", time.time() - t0)
            # elif rank % 5 != 0:
            elif rank != 0:
                SlaveMultiCycle(n, m+1, chi)

        else:
            if rank == 0:
                print("Simulation already ran.")

        break


if __name__ == "__main__":
    # def mpiabort_excepthook(type, value, traceback):
    #     print('type: ', type)
    #     print('value: ', value)
    #     print('traceback: ', traceback)
    #     print('An exception occured. Aborting MPI')
    #     comm.Abort()
    # sys.excepthook = mpiabort_excepthook
    main()
    # sys.excepthook = sys.__excepthook__