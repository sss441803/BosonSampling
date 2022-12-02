import numpy as np
import cupy as cp

from mpo_sort import Aligner
from cuda_kernels import Rand_U, update_MPO


def Canonicalize(n, d, chi, Gamma, Lambda, charge):
    for l in range(n - 1):
        MPOtwoqubitUpdate(n, d, chi, l, Gamma, Lambda, charge)


#MPO update after a two-qudit gate        
def MPOtwoqubitUpdate(n, d, chi, l, Gamma, Lambda, charge):

    # Initializing unitary matrix on GPU
    d_U_r, d_U_i = Rand_U(d, 0)

    LC = Lambda[l,:]
    # Determining the location of the two qubit gate
    left = "Left"
    center = "Center"
    right = "Right"
    if l == 0:
        location = left
        LR = Lambda[l+1,:]
    elif l == n - 2:
        location = right
        LR  = np.ones(chi, dtype = 'float32') # edge lambda (for first and last site) don't exists and are ones
    else:
        location = center
        LR = Lambda[l+1,:]
    
    Glc = Gamma[l,:]
    Gcr = Gamma[l+1,:]

    # charge of corresponding index (bond charge left/center/right)
    CL = charge[l]
    CC = charge[l+1]
    CR = charge[l+2]

    # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
    aligner = Aligner(d, CL, CC, CR)
    # Obtaining aligned charges
    cNewL_obj, cNewR_obj, change_charges_C, change_idx_C = aligner.cNewL, aligner.cNewR, aligner.change_charges_C, aligner.change_idx_C
    d_cNewL_obj, d_cNewR_obj = map(aligner.to_cupy, [cNewL_obj, cNewR_obj])
    d_change_charges_C, d_change_idx_C = cp.array(change_charges_C), cp.array(change_idx_C)
    # Obtaining aligned data
    LR_obj, Glc_obj, Gcr_obj = map(aligner.make_data_obj, ['LR','Glc','Gcr'], [True]*3, [LR, Glc, Gcr], [ [0],[0,0],[0,0] ])
    d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.to_cupy, [LR_obj, Glc_obj, Gcr_obj])
    d_LR_obj, d_Glc_obj, d_Gcr_obj = map(aligner.align_data, [d_LR_obj, d_Glc_obj, d_Gcr_obj])
    
    # Storage of generated data
    new_Gamma_L = []
    new_Gamma_R = []
    new_Lambda = np.array([], dtype='float32')
    new_charge_0 = np.array([], dtype='int32')
    new_charge_1 = np.array([], dtype='int32')
    tau_array = [0]

    valid_idx_l_0 = np.where(CL[:, 0] != d)[0]
    valid_idx_l_1 = np.where(CL[:, 1] != d)[0]
    if valid_idx_l_0.shape[0] > 0:
        largest_cl_0 = np.max(CL[valid_idx_l_0, 0])
    else:
        largest_cl_0 = 0
    if valid_idx_l_1.shape[0] > 0:
        largest_cl_1 = np.max(CL[valid_idx_l_1, 1])
    else:
        largest_cl_1 = 0
    smallest_cr_0 = np.min(CR[:, 0])
    smallest_cr_1 = np.min(CR[:, 1])
    for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
        for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):

            # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
            min_charge_l_0, max_charge_l_0, min_charge_c_0, max_charge_c_0, min_charge_r_0, max_charge_r_0 = charge_range(d, location, charge_c_0)
            min_charge_l_1, max_charge_l_1, min_charge_c_1, max_charge_c_1, min_charge_r_1, max_charge_r_1 = charge_range(d, location, charge_c_1)
            # Selecting data according to charge bounds
            d_cl_obj = aligner.select_data(d_cNewL_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1)
            d_cr_obj = aligner.select_data(d_cNewR_obj, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
            d_lr_obj = aligner.select_data(d_LR_obj, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
            d_glc_obj = aligner.select_data(d_Glc_obj, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1,
                                                        min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1)
            d_gcr_obj = aligner.select_data(d_Gcr_obj, min_charge_c_0, max_charge_c_0, min_charge_c_1, max_charge_c_1,
                                                        min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)

            # Skip if any selection must be empty
            if d_cl_obj.data.shape[0] * d_cr_obj.data.shape[0] * d_cl_obj.data.shape[1] * d_cr_obj.data.shape[1] == 0:
                tau_array.append(0)
                del d_cl_obj.data, d_cr_obj.data, d_lr_obj.data, d_glc_obj.data, d_gcr_obj.data
                continue

            d_C_obj = update_MPO(d, charge_c_0, charge_c_1, d_U_r, d_U_i, d_glc_obj, d_gcr_obj, d_cl_obj, d_cr_obj, d_change_charges_C, d_change_idx_C)
            d_T_obj = d_C_obj.clone()
            d_T_obj.data = cp.multiply(d_C_obj.data, d_lr_obj.data)
            d_C = aligner.compact_data(d_C_obj)
            d_T = aligner.compact_data(d_T_obj)
            
            # SVD
            d_V, d_Lambda, d_W = cp.linalg.svd(d_T, full_matrices = False)
            d_W = cp.matmul(cp.conj(d_V.T), d_C)
            Lambda = cp.asnumpy(d_Lambda)

            # Store new results
            new_Gamma_L = new_Gamma_L + [d_V[:, i] for i in range(len(Lambda))]
            new_Gamma_R = new_Gamma_R + [d_W[i, :] for i in range(len(Lambda))]
            
            new_Lambda = np.append(new_Lambda, Lambda)
            new_charge_0 = np.append(new_charge_0, np.repeat(np.array(charge_c_0, dtype='int32'), len(Lambda)))
            new_charge_1 = np.append(new_charge_1, np.repeat(np.array(charge_c_1, dtype='int32'), len(Lambda)))
            tau_array.append(len(Lambda))

    # Number of singular values to save
    num_lambda = int(min(new_Lambda.shape[0], chi))
    # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
    if num_lambda!= 0:
        idx_select = np.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
    else:
        idx_select = np.array([], dtype='int32')
    
    # Initialize selected and sorted Gamma outputs
    d_Gamma0Out = Aligner.make_data_obj('Glc', False, cp.zeros([chi, chi], dtype = 'complex64'), [0, 0])
    d_Gamma1Out = Aligner.make_data_obj('Gcr', False, cp.zeros([chi, chi], dtype = 'complex64'), [0, 0])

    # Indices of eigenvalues that mark the beginning of center charge tau
    cum_tau_array = np.cumsum(tau_array)
    
    tau = 0
    # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
    for charge_c_0 in range(smallest_cr_0, largest_cl_0 + 1):
        for charge_c_1 in range(smallest_cr_1, largest_cl_1 + 1):
            # Selecting gamma that will be modified. Modifying gamma will modify Gamma (because they are pointers).
            min_charge_l_0, max_charge_l_0, _, _, min_charge_r_0, max_charge_r_0 = charge_range(d, location, charge_c_0)
            min_charge_l_1, max_charge_l_1, _, _, min_charge_r_1, max_charge_r_1 = charge_range(d, location, charge_c_1)
            idx_gamma0_0, idx_gamma0_1 = aligner.get_select_index(d_Gamma0Out, min_charge_l_0, max_charge_l_0, min_charge_l_1, max_charge_l_1, 0, d, 0, d)
            idx_gamma1_0, idx_gamma1_1 = aligner.get_select_index(d_Gamma1Out, 0, d, 0, d, min_charge_r_0, max_charge_r_0, min_charge_r_1, max_charge_r_1)
            # Finding bond indices (tau_idx) that are in the largest num_lambda singular values and for center charge tau.
            # idx_select[indices] = tau_idx
            tau_idx, indices, _ = np.intersect1d(idx_select, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
            tau += 1 # This line MUST be before the continue statement

            if len(tau_idx) * idx_gamma0_0.shape[0] * idx_gamma0_1.shape[0] * idx_gamma1_0.shape[0] * idx_gamma1_1.shape[0] == 0:
                continue
            # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
            d_V = cp.array([new_Gamma_L[i] for i in tau_idx], dtype = 'complex64')
            d_W = cp.array([new_Gamma_R[i] for i in tau_idx], dtype = 'complex64')
            d_V = d_V.T
            
            # Calculating output gamma
            # Left
            d_Gamma0Out.data[idx_gamma0_0.reshape(-1,1), idx_gamma0_1[indices].reshape(1,-1)] = d_V
            # Right
            d_Gamma1Out.data[idx_gamma1_0[indices].reshape(-1,1), idx_gamma1_1.reshape(1,-1)] = d_W
            
    # Select charges that corresponds to the largest num_lambda singular values
    new_charge_0 = new_charge_0[idx_select]
    new_charge_1 = new_charge_1[idx_select]
    # Sort the new charges
    idx_sort = np.lexsort((new_charge_1, new_charge_0)) # Indices that will sort the new charges
    new_charge_0 = new_charge_0[idx_sort]
    new_charge_1 = new_charge_1[idx_sort]
    # Update charges (modifying CC modifies dcharge by pointer)
    CC[:num_lambda, 0] = new_charge_0
    CC[:num_lambda, 1] = new_charge_1
    CC[num_lambda:] = d # Charges beyond num_lambda are set to impossible values d
    
    # Selecting and sorting Lambda
    new_Lambda = new_Lambda[idx_select]
    new_Lambda = new_Lambda[idx_sort]

    LC[:num_lambda] = new_Lambda
    LC[num_lambda:] = 0

    # Sorting Gamma
    d_Gamma0Out.data[:, :num_lambda] = d_Gamma0Out.data[:, idx_sort]
    d_Gamma1Out.data[:num_lambda] = d_Gamma1Out.data[idx_sort]

    if location == right:
        Gamma[n - 2, :, :min(chi, d ** 2)] = cp.asnumpy(d_Gamma0Out.data[:, :min(chi, d ** 2)])
        Gamma[n - 1, :min(chi, d ** 2), 0] = cp.asnumpy(d_Gamma1Out.data[:min(chi, d ** 2), 0])
    else:
        Gamma[l, :, :] = cp.asnumpy(d_Gamma0Out.data)
        Gamma[l + 1, :, :] = cp.asnumpy(d_Gamma1Out.data)


# Gives the range of left, center and right hand side charge values when center charge is fixed to tau
def charge_range(d, location, tau):
    # Speficying allowed left and right charges
    if location == 'left':
        min_charge_l = max_charge_l = d - 1 # The leftmost site must have all photons to the right, hence charge can only be m
    else:
        min_charge_l, max_charge_l = tau, d - 1 # Left must have more or equal photons to its right than center
    # Possible center site charge
    min_charge_c, max_charge_c = 0, d - 1 # The center charge is summed so returns 0 and maximum possible charge.
    # Possible right site charge
    if location == 'right':
        min_charge_r = max_charge_r = 0 # The rightmost site must have all photons to the left, hence charge can only be 0
    else:    
        min_charge_r, max_charge_r = 0, tau # Left must have more or equal photons to its right than center
    
    return min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r