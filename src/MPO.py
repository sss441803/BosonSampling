import numpy as np
from qutip import squeeze, thermal_dm

def Initialize(input_state_type, n_modes, n_input_states, post_selected_photon_number, local_hilbert_space_dimension, parameters):

    # Loading input state type specific experiment configuration parameters
    if input_state_type == 'Gaussian':
        # beta: output photon number multiplication factor
        loss, squeeze_parameter = parameters['loss'], parameters['r']
    elif input_state_type == 'Single photon':
        loss = parameters['loss']
    elif input_state_type == 'Thermal':
        raise NotImplementedError
    else:
        raise ValueError('input_state_type {} not supported.'.format(input_state_type))
    # Initializing single input state density matrix
    if input_state_type == 'Gaussian':
        am = (1 - loss) * np.exp(- 2 * squeeze_parameter) + loss
        ap = (1 - loss) * np.exp(2 * squeeze_parameter) + loss
        s = 1 / 4 * np.log(ap / am)
        n_th = 1 / 2 * (np.sqrt(am * ap) - 1)
        nn = 40
        single_input_density_matrix = (squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()).full()[:(local_hilbert_space_dimension + 1), :(local_hilbert_space_dimension + 1)]
    elif input_state_type == 'Single photon':
        single_input_density_matrix = np.zeros([local_hilbert_space_dimension, local_hilbert_space_dimension], dtype = 'complex64')
        single_input_density_matrix[0, 0] = loss
        single_input_density_matrix[1, 1] = 1 - loss
    elif input_state_type == 'Thermal':
        raise NotImplementedError
    else:
        raise ValueError('input_state_type {} not supported.'.format(input_state_type))

    initial_bond_dimension = local_hilbert_space_dimension ** 2
    Lambda = np.zeros([initial_bond_dimension, n_modes - 1], dtype = 'float32')
    Gamma = np.zeros([initial_bond_dimension, initial_bond_dimension, n_modes], dtype = 'complex64')  
    charge = local_hilbert_space_dimension * np.ones([initial_bond_dimension, n_modes + 1, 2], dtype = 'int32')
    charge[0] = 0
    
    # Filling MPO
    if post_selected_photon_number == None:
        for i in range(local_hilbert_space_dimension):
            charge[i, 0, 0] = i
            charge[i, 0, 1] = i
        updated_bonds = np.array([bond for bond in range(local_hilbert_space_dimension)])
    else:
        charge[0, 0, 0] = post_selected_photon_number
        charge[0, 0, 1] = post_selected_photon_number
        updated_bonds = np.array([0])

    for i in range(n_input_states - 1):
        bonds_updated = np.zeros(local_hilbert_space_dimension**2)
        for j in updated_bonds:
            if charge[j, i, 0] == local_hilbert_space_dimension:
                c1 = 0
            else:
                c1 = charge[j, i, 0]
            for ch_diff1 in range(c1, -1, -1):
                if charge[j, i, 1] == local_hilbert_space_dimension:
                    c2 = 0
                else:
                    c2 = charge[j, i, 1]
                for ch_diff2 in range(c2, -1, -1):
                    if np.abs(single_input_density_matrix[ch_diff1, ch_diff2]) <= 10 ** (-7):
                        continue
                    Gamma[j, (c1 - ch_diff1) * local_hilbert_space_dimension + c2 - ch_diff2, i] = single_input_density_matrix[ch_diff1, ch_diff2]
                    charge[(c1 - ch_diff1) * local_hilbert_space_dimension + c2 - ch_diff2, i + 1, 0] = c1 - ch_diff1
                    charge[(c1 - ch_diff1) * local_hilbert_space_dimension + c2 - ch_diff2, i + 1, 1] = c2 - ch_diff2
                    bonds_updated[(c1 - ch_diff1) * local_hilbert_space_dimension + c2 - ch_diff2] = 1
        updated_bonds = np.where(bonds_updated == 1)[0]
        Lambda[updated_bonds, i] = 1

    for j in updated_bonds:
        if charge[j, n_input_states - 1, 0] == local_hilbert_space_dimension:
            c0 = 0
        else:
            c0 = charge[j, n_input_states - 1, 0]
        if charge[j, n_input_states - 1, 1] == local_hilbert_space_dimension:
            c1 = 0
        else:
            c1 = charge[j, n_input_states - 1, 1]
        Gamma[j, 0, n_input_states - 1] = single_input_density_matrix[c0, c1]
    
    for i in range(n_input_states - 1, n_modes - 1):
        Lambda[0, i] = 1
        charge[0, i + 1, 0] = 0
        charge[0, i + 1, 1] = 0
    
    for i in range(n_input_states):
        Gamma[:, :, i] = np.multiply(Gamma[:, :, i], Lambda[:, i].reshape(1, -1))

    for i in range(n_input_states, n_modes):
        Gamma[0, 0, i] = 1

    Gamma = np.transpose(Gamma, (2, 0, 1))
    Lambda = np.transpose(Lambda, (1, 0))
    charge = np.transpose(charge, (1, 0, 2))

    # Sorting bonds based on bond charges
    for i in range(n_modes + 1):
        idx = np.lexsort((charge[i, :, 1], charge[i, :, 0]))
        charge[i] = charge[i, idx]
        if i > 0:
            Gamma[i - 1] = Gamma[i - 1][:, idx]
            if i < n_modes:
                Lambda[i - 1] = Lambda[i - 1, idx]
        if i < n_modes:
            Gamma[i] = Gamma[i, idx]
    
    Gamma = np.ascontiguousarray(Gamma)
    Lambda = np.ascontiguousarray(Lambda)
    charge = np.ascontiguousarray(charge)

    return Gamma, Lambda, charge