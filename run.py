'''Full simulation code containing the Device method (cupy, unified update)'''
import os
import time
from filelock import FileLock
import pickle
import sys
import argparse

import cupy as cp
import numpy as np
np.set_printoptions(3)

from src import cpu_algorithm, gpu_algorithm, beam_splitter_parallel, fully_parallel


from mpi4py import MPI
comm = MPI.COMM_WORLD
ranks = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument('--al', type=str, help="Which algorithm/implemetation to use.", default=0)
parser.add_argument('--nodes', type=int, default=0)
parser.add_argument('--rpn', type=int, help="Ranks per node.", default=0)
parser.add_argument('--gpn', type=int, help="GPUs per node.", default=0)
parser.add_argument('--rpg', type=int, help="Ranks per GPU.", default=0)
args = vars(parser.parse_args())

algorithm = args['al']
nodes = args['nodes']
ranks_per_node = args['rpn']
gpus_per_node = args['gpn']
ranks_per_gpu = args['rpg']

if algorithm == 'cpu':
    rank = 0
    Worker = cpu_algorithm.Worker()

elif algorithm == 'gpu':
    rank = 0
    Worker = gpu_algorithm.Worker()

elif algorithm == 'beam_splitter_parallel':
    assert ranks == nodes * ranks_per_node, 'Number of ranks for MPI is not compatible with the script arguments. Should be nodes * ranks_per_node'
    if rank == 0:
        Worker = beam_splitter_parallel.FullWorker(nodes, ranks_per_node)
    else:
        Worker = beam_splitter_parallel.RankWorker()
        gpu = (rank % ranks_per_node - 1) % gpus_per_node
        cp.cuda.Device(gpu).use()

elif algorithm == 'fully_parallel':
    assert ranks_per_node == ranks_per_gpu * gpus_per_node + 2, 'Arguments specifying number of ranks per node/gpu, gpus per node are incompatible. Expecting ranks_per_node == ranks_per_gpu * gpus_per_node + 2. One additional rank for node level beam splitter update, and  one additional rank for overall management.'
    assert ranks == nodes * ranks_per_node, 'Number of ranks for MPI is not compatible with the script arguments. Should be nodes * ranks_per_node'
    gpu_ranks_per_node = ranks_per_gpu * gpus_per_node
    this_node = rank // ranks_per_node
    node_gpu_ranks = [i + 2 for i in range(ranks_per_node * this_node, ranks_per_node * this_node + gpu_ranks_per_node)]
    node_control_rank = this_node * ranks_per_node + 1

    if rank == 0:
        Worker = fully_parallel.FullWorker(nodes, ranks_per_node)
    elif rank == node_control_rank:
        Worker = fully_parallel.NodeWorker(nodes, ranks_per_node, this_node)
    elif rank % ranks_per_node == 0:
        Worker = fully_parallel.NodeDataWorker(nodes, ranks_per_node, this_node)
    elif rank in node_gpu_ranks:
        Worker = fully_parallel.RankWorker(node_control_rank)
        gpu = (rank % ranks_per_node - 2) % gpus_per_node
        cp.cuda.Device(gpu).use()
    else:
        print('Invalid rank encountered')
        comm.Abort()
    
else:
    print('Algorithm {} is not supported'.format(algorithm))
    comm.Abort()


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
            with FileLock("experiments.pickle.lock"):
                # work with the file as it is now locked
                print("Lock acquired.")
                # Load experiments
                with open("experiments.pickle", 'rb') as experiment_file:
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
                # If loop never broke, no experiment was found
                if not found_experiment:
                    # If searched all experiments in a single go without running any simulations, all are done. Terminate
                    if exp_idx_beginning == 0:
                        print('All experiments already ran. Exiting.')
                        comm.Abort()
                    # If previously ran experiments, search from top again since previously ran experiments might not have sufficient accuracy.
                    else:
                        print('Searching from 0th experiment again.')
                        exp_idx_beginning = 0
                else:
                    exp_idx_beginning = exp_idx + 1

                # Will run the first found incomplete experiment, set status to in progress
                experiments[exp_idx]['status'] = 'in progress'
                # Update experiment track file
                with open('experiments.pickle', 'wb') as file:
                    pickle.dump(experiments, file)
                experiments[exp_idx]['status'] = 'incomplete'
                status = 'run'

                # Obtaining information regarding the experiment
                # input_state_type: Gaussian, Single photon, Thermal, etc.
                # parameters: Further parameters needed to specify the input configuration, specific to each input state type. Is a string of dictionary.
                # local_hilbert_space_dimension: This is the maximum number of photons we can simulate in a single mode
                input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, bond_dimension, parameters = experiment['input_state_type'], experiment['n_modes'], experiment['n_input_states'], experiment['post_selected_photon_number'], experiment['local_hilbert_space_dimension'], experiment['bond_dimension'], experiment['parameters']

                # Specifying the filename and location to save the results to
                begin_directory = 'Results/{}/n_{}_m_{}_PS_{}_d_{}_p_{}_'.format(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters)

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

                # Check if simulation requirements are reasonable
                if status == 'run':
                    if bond_dimension > 10000:
                        print('Required bond-dimension chi too large. Moving on to next experiment.')
                        status = 'skip'
                    elif n_modes > 200:
                        print('Too many modes. Moving on to next experiment.')
                        status = 'skip'
                
        else:
            status = n_modes = n_input_states = local_hilbert_space_dimension = bond_dimension = None
            
        # Broadcasting to all ranks
        status, n_modes, n_input_states, local_hilbert_space_dimension, bond_dimension = comm.bcast([status, n_modes, n_input_states, local_hilbert_space_dimension, bond_dimension], root=0)
        if status == 'skip':
            continue

        # input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, bond_dimension, parameters = 'Gaussian', 10, 5, None, 20, 4096, {'beta': 1.0, 'loss': 0.1, "r": 0.88}

        t0 = time.time()

        # Main rank
        if rank == 0:
            t0 = time.time()
            Worker.ExperimentInit(input_state_type, n_modes, n_input_states, post_selected_photon_number,local_hilbert_space_dimension, bond_dimension, parameters)
            success, Totprob, EE, RE = Worker.Simulate()
            print(Totprob)
            print(EE)
            # Saving results
            if os.path.isfile(begin_directory + 'chi.npy'):
                bond_dimension_array = np.load(begin_directory + 'chi.npy')
            else:
                bond_dimension_array = np.array([])
            # assert not np.sum(bond_dimension_array == bond_dimension), 'chi {} already in chi array'.format(bond_dimension)
            bond_dimension_array = np.append(bond_dimension_array, bond_dimension)
            prob_file = begin_directory + 'chi_{}_prob.npy'.format(bond_dimension)
            EE_file = begin_directory + 'chi_{}_EE.npy'.format(bond_dimension)
            # assert not os.path.isfile(prob_file), '{} exists already. Error.'.format(prob_file)
            # assert not os.path.isfile(EE_file), '{} exists already. Error.'.format(EE_file)
            if np.sum(Totprob > 0) > 0:
                np.save(prob_file, Totprob)
                np.save(EE_file, EE)
                np.save(begin_directory + 'chi.npy', bond_dimension_array)
                if not success:
                    with FileLock("experiment.pickle.lock"):
                        # work with the file as it is now locked
                        print("Lock acquired.")
                        # Load experiments
                        with open('experiments.pickle', 'wb') as file:
                            pickle.dump(experiments, file)
            else:
                print('Results invalid. Not saving.')
            print("Time cost", time.time() - t0)

        # Sub ranks
        else:
            # print(rank, 'Init')
            Worker.ExperimentInit(n_modes, local_hilbert_space_dimension, bond_dimension)
            # print('Simulate')
            Worker.Simulate()
            # print('Done Simulate')

        # break


if __name__ == "__main__":
    def mpiabort_excepthook(type, value, traceback):
        print('type: ', type)
        print('value: ', value)
        print('traceback: ', traceback)
        print('An exception occured. Aborting MPI')
        comm.Abort()
    sys.excepthook = mpiabort_excepthook
    main()