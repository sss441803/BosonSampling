import pickle
from qutip import squeeze, thermal_dm
from math import sinh
import numpy as np


def photon_number_distribution(n_input_states, squeeze_parameter, loss):
    am = (1 - loss) * np.exp(- 2 * squeeze_parameter) + loss
    ap = (1 - loss) * np.exp(2 * squeeze_parameter) + loss
    s = 1 / 4 * np.log(ap / am)
    n_th = 1 / 2 * (np.sqrt(am * ap) - 1)
    nn = 40
    single_dist = np.array(np.diag(squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()), dtype = float)
    prob_dist = np.array(np.diag(squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()), dtype = float)
    for _ in range(n_input_states - 1):
        prob_dist = np.convolve(prob_dist, single_dist)
    return prob_dist

def find_dimension_needed_to_preserve_probability(distribution, target_probability):
    probability = 0
    index = 0
    while probability < target_probability:
        probability += distribution[index]
        index += 1
    return index

def determine_hilbert_space_dimension_from_gaussian_state_parameters_and_target_probability(n_input_states, squeeze_parameter, loss, target_probability):
    distribution = photon_number_distribution(n_input_states, squeeze_parameter, loss)
    local_hilbert_space_dimension = find_dimension_needed_to_preserve_probability(distribution, target_probability)
    return local_hilbert_space_dimension


experiments = []
ds = []
ns = []
# for m in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50]:
#     for beta in [0.6, 0.8, 1.0, 1.2]:
#         for r in [0.48, 0.662, 0.88, 1.146, 1.44]:
#             n = 4 * m
#             ideal_ave_photons = m*sinh(r)**2
#             lossy_ave_photons = beta*sqrt(ideal_ave_photons)
#             loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
#             PS = int((1-loss)*m*sinh(r)**2)
#             PS += 1
#             d = PS + 1
#             init_chi = d**2
#             chi = int(max(32*2**PS, d**2, 512))
#             if  not (d > 10 or chi > 10000 or PS == 0):
#                 experiments.append({'n': n, 'm': m, 'beta': beta, 'r': r, 'PS': PS, 'd': d, 'status': 'incomplete'})
#                 ds.append(d)
#                 ns.append(n)
#                 if PS > 1:
#                     experiments.append({'n': n, 'm': m, 'beta': beta, 'r': r, 'PS': PS - 1, 'd': d - 1, 'status': 'incomplete'})
#                     ds.append(d - 1)
#                     ns.append(n)

target_probability = 0.99
for n_input_states in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]:
    for beta in [0.6, 0.8, 1.0, 1.2]:
        for squeeze_parameter in [0.88]:#, 1.146, 1.44]:
            n_modes = max(20, 4 * n_input_states)
            ideal_ave_photons = n_input_states*sinh(squeeze_parameter)**2
            lossy_ave_photons = beta*ideal_ave_photons**(1/2)
            loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
            local_hilbert_space_dimension = determine_hilbert_space_dimension_from_gaussian_state_parameters_and_target_probability(n_input_states, squeeze_parameter, loss, target_probability)
            init_chi = local_hilbert_space_dimension**2
            bond_dimension = int(max(8*2**lossy_ave_photons, local_hilbert_space_dimension**2, 512))
            parameters = {"beta": beta, "loss": loss, "r": squeeze_parameter}
            experiments.append({"input_state_type": 'Gaussian', "n_modes": n_modes, "n_input_states": n_input_states, "post_selected_photon_number": None, "local_hilbert_space_dimension": local_hilbert_space_dimension, "bond_dimension": bond_dimension, "parameters": parameters, 'status': 'incomplete'})
            ds.append(local_hilbert_space_dimension)
            ns.append(n_modes)

ds = np.array(ds)
ns = np.array(ns)
idx = np.lexsort([ns, ds])
experiments = np.array(experiments)[idx]
with open('experiments.pickle', 'wb') as file:
    pickle.dump(experiments, file)
print('New experiment tracking file created.')
# print(experiments)
print(len(experiments), 'experiments in total.')