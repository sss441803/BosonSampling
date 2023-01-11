import numpy as np
np.set_printoptions(precision=3)
#import jax.numpy as jnp
import numpy as jnp
from qutip import *
from scipy.special import factorial, comb

def Canonicalize(n, d, Gamma, Lambda, charge):
    mpo = MPO(n, d, Gamma, Lambda, charge)
    return mpo.Gamma, mpo.Lambda, mpo.charge


def Rand_BS_MPS(seed, local_hilbert_space_dimension, reflectivity):
    random = np.random.RandomState(seed)
    t = np.sqrt(1 - reflectivity ** 2) * np.exp(1j * random.rand() * 2 * np.pi)
    reflectivity = reflectivity * np.exp(1j * random.rand() * 2 * np.pi)
    ct = np.conj(t); cr = np.conj(reflectivity)
    bs_coeff = lambda n, m, k, l: np.sqrt(factorial(l) * factorial(n + m - l) / factorial(n) / factorial(m)) * comb(n, k) * comb(m, l - k) * (t ** k) * (ct ** (m - l + k)) * (reflectivity ** (n - k)) * ((-cr) ** (l - k))
    Unitary = np.zeros([local_hilbert_space_dimension]*4, dtype = 'complex64')
    for n in range(local_hilbert_space_dimension): #photon number from 0 to d-1
        for m in range(local_hilbert_space_dimension):
            for l in range(max(0, n + m + 1 - local_hilbert_space_dimension), min(local_hilbert_space_dimension, n + m + 1)): #photon number on first output mode
                k = np.arange(max(0, l - m), min(l + 1, n + 1, local_hilbert_space_dimension))
                Unitary[n, m, l, n + m - l] = np.sum(bs_coeff(n, m, k, l))
    return Unitary
                

class MPO:
    def __init__(self, n_modes, local_hilbert_space_dimension, Gamma, Lambda, charge):
        self.n_modes = n_modes
        self.local_hilbert_space_dimension = local_hilbert_space_dimension
        self.bond_dimension = charge.shape[0]
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.edge_Lambda = np.ones(self.bond_dimension, dtype = 'float32')
        self.charge = charge
        Unitary = Rand_BS_MPS(0, self.local_hilbert_space_dimension, 0)
        self.Unitary = Unitary
        
        for mode in range(self.n_modes - 2, -1, -1):
            self.Update(mode)
        

    def Update(self, mode):

        LL = self.edge_Lambda if mode == 0 else self.Lambda[:, mode - 1]
        LC = self.Lambda[:, mode]
        LR = self.edge_Lambda if mode == self.n_modes - 2 else self.Lambda[:, mode + 1]
        Gamma_L_temp = []
        Gamma_R_temp = []
        Lambda_temp = np.array([])
        new_charge1 = np.array([])
        new_charge2 = np.array([])
        tau_array = [0]
        idx_L = np.empty([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "object")
        idx_R = np.empty([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "object")
        idx_C = np.empty([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "object")
        len_L = np.zeros([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "int32")
        len_R = np.zeros([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "int32")
        len_C = np.zeros([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "int32")

        l_bond_array = np.empty([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "object")
        r_bond_array = np.empty([self.local_hilbert_space_dimension, self.local_hilbert_space_dimension], dtype = "object")

        chi = self.bond_dimension
        d = self.local_hilbert_space_dimension
        
        for i in range(d):
            for j in range(d):
                idx_L[i, j] = np.intersect1d(np.nonzero(self.charge[:, mode, 0] == i), np.intersect1d(np.nonzero(self.charge[:, mode, 1] == j), np.nonzero(LL > 0)))
                len_L[i, j] = len(idx_L[i, j])
                idx_C[i, j] = np.intersect1d(np.nonzero(self.charge[:, mode + 1, 0] == i), np.intersect1d(np.nonzero(self.charge[:, mode + 1, 1] == j), np.nonzero(LC > 0)))
                len_C[i, j] = len(idx_C[i, j])
                idx_R[i, j] = np.intersect1d(np.nonzero(self.charge[:, mode + 2, 0] == i), np.intersect1d(np.nonzero(self.charge[:, mode + 2, 1] == j), np.nonzero(LR > 0)))
                len_R[i, j] = len(idx_R[i, j])
        
        CL = self.charge[:, mode]
        CR = self.charge[:, mode + 2]
        valid_idx_l_0 = np.where(CL[:, 0] != self.local_hilbert_space_dimension)[0]
        valid_idx_l_1 = np.where(CL[:, 1] != self.local_hilbert_space_dimension)[0]
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
        for ch_c_1 in range(smallest_cr_0, largest_cl_0 + 1):
            for ch_c_2 in range(smallest_cr_1, largest_cl_1 + 1):
                
                l_charge_1 = np.arange(ch_c_1, d)
                l_charge_2 = np.arange(ch_c_2, d)
                r_charge_1 = np.arange(ch_c_1 + 1)
                r_charge_2 = np.arange(ch_c_2 + 1)
                
                r_bond = []
                for i in r_charge_1:
                    for j in r_charge_2:
                        r_bond.extend(idx_R[i, j])

                l_bond = []
                for i in l_charge_1:
                    for j in l_charge_2:
                        l_bond.extend(idx_L[i, j])              

                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                

                C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                theta = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                L_stack = 0
                for ch_l_1 in l_charge_1:
                    for ch_l_2 in l_charge_2:
                        if len_L[ch_l_1, ch_l_2] == 0:
                            continue
                        L_stack += len_L[ch_l_1, ch_l_2]
                        R_stack = 0
                        for ch_r_1 in r_charge_1:
                            for ch_r_2 in r_charge_2:
                                if len_R[ch_r_1, ch_r_2] == 0:
                                    continue
                                R_stack += len_R[ch_r_1, ch_r_2]
                                c_bond = []; c_local_1 = []; c_local_2 = []
                                for i in np.arange(ch_r_1, ch_l_1 + 1):
                                    for j in np.arange(ch_r_2, ch_l_2 + 1):
                                        c_bond.extend(idx_C[i, j])
                                        c_local_1.extend([i] * len_C[i, j])
                                        c_local_2.extend([j] * len_C[i, j])
                                if len(c_bond) == 0:
                                    continue
                                c_bond = np.array(c_bond); c_local_1 = np.array(c_local_1); c_local_2 = np.array(c_local_2)
                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += np.matmul(self.Gamma[idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1), mode], np.matmul(np.diag(self.Unitary[ch_l_1 - ch_c_1, ch_c_1 - ch_r_1, ch_l_1 - c_local_1, c_local_1 - ch_r_1] * np.conj(self.Unitary[ch_l_2 - ch_c_2, ch_c_2 - ch_r_2, ch_l_2 - c_local_2, c_local_2 - ch_r_2])), self.Gamma[c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1), mode + 1]))

                theta = np.matmul(C, np.diag(LR[r_bond]))

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False)
                V = np.asarray(V)
                Lambda = np.asarray(Lambda)
                W = np.matmul(np.conj(V.T), C)

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))]
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))]          

                Lambda_temp = np.append(Lambda_temp, Lambda)

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)))
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)))
                tau_array.append(len(Lambda))
                    
        num_lambda = min(len(Lambda_temp), chi)
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi])
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[:, mode] = temp
        self.charge[:, mode + 1, :] = 0
        self.charge[:num_lambda, mode + 1, 0] = new_charge1[idx]
        self.charge[:num_lambda, mode + 1, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64')
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64')

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(smallest_cr_0, largest_cl_0 + 1):
            for j in range(smallest_cr_1, largest_cl_1 + 1):
                tau = i * d + j
                tau_idx, indices, _ = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

                if len(tau_idx) == 0 or len(l_bond_array[i, j]) == 0 or len(r_bond_array[i, j]) == 0:
                    continue
                V = np.array([Gamma_L_temp[i] for i in tau_idx], dtype = 'complex64')
                W = np.array([Gamma_R_temp[i] for i in tau_idx], dtype = 'complex64')
                V = V.T

                alpha = np.array(l_bond_array[i, j]).reshape(-1, 1)
                Gamma1Out[alpha, indices.reshape(1, -1)] = V
                alpha = np.array(r_bond_array[i, j]).reshape(1, -1)
                Gamma2Out[indices.reshape(-1, 1), alpha] = W
        
        
        self.Gamma[:, :, mode] = Gamma1Out; self.Gamma[:, :, mode + 1] = Gamma2Out