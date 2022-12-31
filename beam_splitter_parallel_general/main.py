'''Full simulation code containing the Device method (cupy, unified update)'''
import os
import time
from filelock import FileLock
import pickle
import sys

import numpy as np
np.set_printoptions(3)

from parent import ParentMPO
from child import ChildMPO

from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size() - 1
# num_gpu_ranks = num_gpu_ranks // 16 * 4
rank = comm.Get_rank()
# print(rank)
# gpu = (rank%17-1)%num_gpus_per_node
# if rank % 17 != 0:


def main():

    exp_idx_beginning = 0
    # Loop until all experiments are over
    while True:
        # Rank 0 (main process) searches which experiment to run and then tells all parent and child processes
        if rank == 0:
            # File experiment.pickle stores the experiments and their configuration
            # For multiple processes, each process should work on one experiment without overlap.
            # Therefore, a process will lock the file (prevent other processes from accessing it),
            # choose the experiment that has not been run yet, and write to the file that
            # the chosen experiment is running so that other processes won't run it.
            with FileLock("../experiment.pickle.lock"):
                # work with the file as it is now locked
                print("Lock acquired.")
                # Load experiments
                with open("../experiment.pickle", 'rb') as experiment_file:
                    experiments = pickle.load(experiment_file)
                # The file is a list of dictionaries that specify each experiment
                found_experiment = False
                # Loop over all unloaded experiments. Will not re-load experiments that have already been loaded
                for exp_idx in range(exp_idx_beginning, len(experiments)):
                    experiment = experiments[exp_idx]
                    if experiment['status'] == 'incomplete':
                        found_experiment = True
                        print('Found experiment: ', experiment)
                        # Break the loop once an incomplete experiment is found
                        break
                exp_idx_beginning = exp_idx + 1
                # If loop never broke, no experiment was found
                if not found_experiment:
                    print('All experiments already ran. Exiting.')
                    comm.Abort()

                # Obtaining information regarding the experiment
                # input_state_type: Gaussian, Single photon, Thermal, etc.
                # parameters: Further parameters needed to specify the input configuration, specific to each input state type. Is a string of dictionary.
                # local_hilbert_space_dimension: This is the maximum number of photons we can simulate in a single mode
                input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, bond_dimension, parameters = experiment['input_state_type'], experiment['n_modes'], experiment['n_input_states'], experiment['post_selected_photon_number'], experiment['local_hilbert_space_dimension'], experiment['bond_dimension'], experiment['parameters']

                # Specifying the filename and location to save the results to
                begin_directory = '../Results/{}/n_{}_m_{}_PS_{}_d_{}_p_{}_'.format(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters)

                # See if previous runs exist. If so, a file with the previously use bond dimensions should exist.
                # Determine if simulation ran is already good enough to needs to improve the accuracy
                if os.path.isfile(begin_directory + 'chi.npy'):
                    bond_dimension_array = np.load(begin_directory + 'chi.npy')
                    bond_dimension = int(np.max(bond_dimension_array))
                    probability = np.load(begin_directory + 'chi_{}_prob.npy'.format(bond_dimension))
                    probability = probability[np.where(probability > 0)[0]]
                    print('Probability: ', probability)
                    if min(probability) != 0:
                        error = np.max(probability)/np.min(probability) - 1
                    error = np.max(error)
                    print('Error: ', error)
                    if error > 0.1:
                        bond_dimension *= 2
                        print('Bond dimension was too small producing error {}. Increasing chi to {}'.format(error, bond_dimension))
                        status = 'run'
                    else:
                        print('Simulation with suitable accuracy already ran.')
                        status = 'skip'
                else:
                    status = 'run'
                
                # Check if simulation already ran
                if os.path.isfile(begin_directory + 'chi_{}_EE.npy'.format(bond_dimension)):
                    status = 'skip'
                    print("Simulation already ran.")

                # Check if simulation requirements are reasonable
                if status == 'run':
                    if bond_dimension > 5000:
                        print('Required bond-dimension chi too large. Moving on to next experiment.')
                        status = 'skip'
                    elif n_modes > 100:
                        print('Too many modes. Moving on to next experiment.')
                        status = 'skip'
                    else:
                        # Will run the first found incomplete experiment, set status to in progress
                        experiments[exp_idx]['status'] = 'in progress'
                        # Update experiment track file
                        with open('../experiment.pickle', 'wb') as file:
                            pickle.dump(experiments, file)
                        status = 'run'
                
        else:
            status = n_modes = n_input_states = local_hilbert_space_dimension = bond_dimension = None

        # Broadcasting to all ranks
        status, n_modes, n_input_states, local_hilbert_space_dimension, bond_dimension = comm.bcast([status, n_modes, n_input_states, local_hilbert_space_dimension, bond_dimension], root=0)
        if status == 'skip':
            continue

        # n, m, beta, errtol = 10, 5, 1.0, 10**(-7)
        # ideal_ave_photons = beta * m
        # lossy_ave_photons = 0.5 * ideal_ave_photons
        # loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
        # PS = int((1-loss)*m)
        # PS += 1
        # d = PS + 1
        # PS = None
        # d = m+1
        # init_chi = d**2
        # chi = int(max(4*2**d, d**2, 512))
        # # chi = 128
        # print(n, m, beta, loss, PS, d, chi)

        t0 = time.time()
        errtol = 10 ** (-7)   

        if rank == 0:
            t0 = time.time()
            parentMPO = ParentMPO(num_gpu_ranks, input_state_type, n_modes, n_input_states, post_selected_photon_number,local_hilbert_space_dimension, bond_dimension, parameters, errtol)
            Totprob, EE, RE = parentMPO.FullUpdate()
            print(Totprob)
            print(EE)
            # Saving results
            if os.path.isfile(begin_directory + 'chi.npy'):
                bond_dimension_array = np.load(begin_directory + 'chi.npy')
            else:
                bond_dimension_array = np.array([])
            assert not np.sum(bond_dimension_array == bond_dimension), 'chi {} already in chi array'.format(bond_dimension)
            bond_dimension_array = np.append(bond_dimension_array, bond_dimension)
            prob_file = begin_directory + 'chi_{}_prob.npy'.format(bond_dimension)
            EE_file = begin_directory + 'chi_{}_EE.npy'.format(bond_dimension)
            assert not os.path.isfile(prob_file), '{} exists already. Error.'.format(prob_file)
            assert not os.path.isfile(EE_file), '{} exists already. Error.'.format(EE_file)
            if np.sum(Totprob > 0) > 0:
                np.save(prob_file, Totprob)
                np.save(EE_file, EE)
                np.save(begin_directory + 'chi.npy', bond_dimension_array)
            else:
                print('Results invalid. Not saving.')
            print("Time cost", time.time() - t0)
        # elif rank % 5 != 0:
        elif rank != 0:
            childMPO = ChildMPO(n_modes, local_hilbert_space_dimension, bond_dimension)
            childMPO.Childloop()

        # break


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