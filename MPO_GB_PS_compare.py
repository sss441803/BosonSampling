import numpy as np
np.set_printoptions(precision=3)
#import jax.numpy as jnp
import numpy as jnp
from qutip import *

from scipy.stats import rv_continuous
from scipy.special import factorial, comb

import time


def change_idx(x, d):
    return d * x[:, 0] + x[:, 1];

def Rand_BS_MPS(d, r):
    #t = 1 / np.sqrt(2); r = 1 / np.sqrt(2);
    t = np.sqrt(1 - r ** 2) * np.exp(1j * np.random.rand() * 2 * np.pi);
    r = r * np.exp(1j * np.random.rand() * 2 * np.pi);
    ct = np.conj(t); cr = np.conj(r);
    bs_coeff = lambda n, m, k, l: np.sqrt(factorial(l) * factorial(n + m - l) / factorial(n) / factorial(m)) * comb(n, k) * comb(m, l - k) * (t ** k) * (ct ** (m - l + k)) * (r ** (n - k)) * ((-cr) ** (l - k))
    BS = np.zeros([d, d, d, d], dtype = 'complex64');
    for n in range(d): #photon number from 0 to d-1
        for m in range(d):
            for l in range(max(0, n + m + 1 - d), min(d, n + m + 1)): #photon number on first output mode
                k = np.arange(max(0, l - m), min(l + 1, n + 1, d))
                BS[n, m, l, n + m - l] = np.sum(bs_coeff(n, m, k, l))
                
    Output = BS;
    
    return Output

class MPO:
    def __init__(self, n, m, d, r, loss, chi, errtol = 10 ** (-6), PS = None):
        self.n = n;
        self.m = m;
        self.d = d
        self.r = r;
        self.K = m;
        self.loss = loss;
        self.chi = chi;
        self.TotalProbPar = np.zeros([n], dtype = 'float32');
        self.SingleProbPar = np.zeros([n], dtype = 'float32');
        self.EEPar = np.zeros([n - 1, n], dtype = 'float32');        
        self.REPar = np.zeros([n - 1, n, 5], dtype = 'float32');
        self.PS = PS;
        self.U_time = 0
        self.svd_time = 0
        self.theta_time = 0

    def MPOInitialization(self):
        self.Lambda = np.zeros([self.chi, self.n - 1], dtype = 'float32');
        self.A = np.zeros([self.chi, self.chi, self.n], dtype = 'complex64');
        self.Gamma = np.zeros([self.chi, self.chi, self.n], dtype = 'complex64');        
        self.charge = np.zeros([self.chi, self.n + 1, 2], dtype = 'int32');

        #MPS Part
        chi = self.chi; d = self.d; K = self.K;
        
        am = (1 - self.loss) * np.exp(- 2 * self.r) + self.loss;
        ap = (1 - self.loss) * np.exp(2 * self.r) + self.loss;
        s = 1 / 4 * np.log(ap / am);
        n_th = 1 / 2 * (np.sqrt(am * ap) - 1);
        nn = 40;
        
        sq = (squeeze(nn, s) * thermal_dm(nn, n_th) * squeeze(nn, s).dag()).full()[:(d + 1), :(d + 1)];

        if self.PS == None:
            for i in range(d):
                self.charge[i, 0, 0] = i
                self.charge[i, 0, 1] = i
            #pre_chi = d
            updated_bonds = np.array([bond for bond in range(d)])
        else:
            self.charge[0, 0, 0] = self.PS
            self.charge[0, 0, 1] = self.PS
            # pre_chi = 1
            updated_bonds = np.array([0])

        for i in range(K - 1):
            print('Initializing mode ', i)
            #chi_ = 0
            #for j in range(pre_chi):
            bonds_updated = np.zeros(d**2)
            for j in updated_bonds:
                if self.charge[j, i, 0] == d:
                    c1 = 0
                else:
                    c1 = self.charge[j, i, 0]
                for ch_diff1 in range(c1, -1, -1):
                    if self.charge[j, i, 1] == d:
                        c2 = 0
                    else:
                        c2 = self.charge[j, i, 1]
                    for ch_diff2 in range(c2, -1, -1):
                        if np.abs(sq[ch_diff1, ch_diff2]) <= errtol:
                            continue
                        #self.Gamma_temp[j, chi_, i] = sq[ch_diff1, ch_diff2]
                        #self.charge[chi_, i + 1, 0] = c1 - ch_diff1
                        #self.charge[chi_, i + 1, 1] = c2 - ch_diff2
                        self.Gamma[j, (c1 - ch_diff1) * d + c2 - ch_diff2, i] = sq[ch_diff1, ch_diff2]
                        self.charge[(c1 - ch_diff1) * d + c2 - ch_diff2, i + 1, 0] = c1 - ch_diff1
                        self.charge[(c1 - ch_diff1) * d + c2 - ch_diff2, i + 1, 1] = c2 - ch_diff2
                        bonds_updated[(c1 - ch_diff1) * d + c2 - ch_diff2] = 1
                        #chi_ += 1
            # self.Lambda[:chi_, i] = 1
            #pre_chi = chi_
            updated_bonds = np.where(bonds_updated == 1)[0]
            self.Lambda[updated_bonds, i] = 1
            # print('Chi ', chi_)

        print('Computing Gamma')
        # for j in range(pre_chi):
        for j in updated_bonds:
            if self.charge[j, K - 1, 0] == d:
                c0 = 0
            else:
                c0 = self.charge[j, K - 1, 0]
            if self.charge[j, K - 1, 1] == d:
                c1 = 0
            else:
                c1 = self.charge[j, K - 1, 1]
            self.Gamma[j, 0, K - 1] = sq[c0, c1]
        
        for i in range(self.m - 1, self.n - 1):
            self.Lambda[0, i] = 1
            self.charge[0, i + 1, 0] = 0
            self.charge[0, i + 1, 1] = 0
        
        # for i in range(self.m):
        #     self.A[:, :, i] = self.Gamma[:, :, i] @ np.diag(self.Lambda[:, i]);
        for i in range(self.m):
            self.A[:, :, i] = np.multiply(self.Gamma[:, :, i], self.Lambda[:, i].reshape(1, -1))
        
        for i in range(self.m, self.n):
            self.A[0, 0, i] = 1;
        
        for l in range(self.n - 1):
            self.MPOtwoqubitUpdate(l, 0)
        
    #MPO update after a two-qudit gate

    def MPOtwoqubitUpdateLeft(self, UnitaryMPS):
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = np.array([]);
        new_charge1 = np.array([]);
        new_charge2 = np.array([]);
        tau_array = [0];
        idx_L = np.empty([self.d, self.d], dtype = "object")
        idx_R = np.empty([self.d, self.d], dtype = "object")
        idx_C = np.empty([self.d, self.d], dtype = "object")
        len_L = np.zeros([self.d, self.d], dtype = "int32")
        len_R = np.zeros([self.d, self.d], dtype = "int32")
        len_C = np.zeros([self.d, self.d], dtype = "int32")

        l_bond_array = np.empty([self.d, self.d], dtype = "object")
        r_bond_array = np.empty([self.d, self.d], dtype = "object")

        chi = self.chi;
        d = self.d;
        
        UnitaryMPO = np.kron(UnitaryMPS, np.conj(UnitaryMPS));
        
        for i in range(d):
            for j in range(d):
                idx_L[i, j] = np.intersect1d(np.nonzero(self.charge[:, 0, 0] == i), np.nonzero(self.charge[:, 0, 1] == j));
                len_L[i, j] = len(idx_L[i, j])
                idx_C[i, j] = np.intersect1d(np.nonzero(self.charge[:, 1, 0] == i), np.intersect1d(np.nonzero(self.charge[:, 1, 1] == j), np.nonzero(self.Lambda[:, 0] > 0)))
                len_C[i, j] = len(idx_C[i, j])
                idx_R[i, j] = np.intersect1d(np.nonzero(self.charge[:, 2, 0] == i), np.intersect1d(np.nonzero(self.charge[:, 2, 1] == j), np.nonzero(self.Lambda[:, 0 + 1] > 0)))
                len_R[i, j] = len(idx_R[i, j])
        
        CL = self.charge[:, 0]
        CR = self.charge[:, 2]
        valid_idx_l_0 = np.where(CL[:, 0] != self.d)[0]
        valid_idx_l_1 = np.where(CL[:, 1] != self.d)[0]
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
                ch_c = ch_c_1 * d + ch_c_2
                l_charge_1 = np.arange(ch_c_1, d);
                l_charge_2 = np.arange(ch_c_2, d);
                r_charge_1 = np.arange(ch_c_1 + 1);
                r_charge_2 = np.arange(ch_c_2 + 1);
                
                r_bond = [];
                for i in r_charge_1:
                    for j in r_charge_2:
                        r_bond.extend(idx_R[i, j]);

                l_bond = [];
                for i in l_charge_1:
                    for j in l_charge_2:
                        l_bond.extend(idx_L[i, j]);                

                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                

                start = time.time()

                C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                theta = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                L_stack = 0;
                for ch_l_1 in l_charge_1:
                    for ch_l_2 in l_charge_2:
                        if len_L[ch_l_1, ch_l_2] == 0:
                            continue
                        ch_l = ch_l_1 * d + ch_l_2
                        L_stack += len_L[ch_l_1, ch_l_2];
                        R_stack = 0;
                        for ch_r_1 in r_charge_1:
                            for ch_r_2 in r_charge_2:
                                if len_R[ch_r_1, ch_r_2] == 0:
                                    continue
                                ch_r = ch_r_1 * d + ch_r_2
                                R_stack += len_R[ch_r_1, ch_r_2];
                                c_bond = []; c_local = [];
                                for i in np.arange(ch_r_1, ch_l_1 + 1):
                                    for j in np.arange(ch_r_2, ch_l_2 + 1):
                                        c_bond.extend(idx_C[i, j])
                                        #c_local.extend(np.repeat(i * d + j, len_C[i, j]))
                                        c_local.extend([i * d + j] * len_C[i, j])
                                if len(c_bond) == 0:
                                    continue
                                c_bond = np.array(c_bond); c_local = np.array(c_local)
                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += np.matmul(self.A[idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1), 0], np.matmul(np.diag(UnitaryMPO[ch_l - ch_c, ch_c - ch_r, ch_l - c_local, c_local - ch_r]), self.A[c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1), 1]))

                theta = np.matmul(C, np.diag(self.Lambda[r_bond, 1]));
                self.theta_time += time.time() - start
                # print(theta)

                start = time.time()

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False);
                W = np.matmul(np.conj(V.T), C);

                self.svd_time += time.time() - start

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];            

                Lambda_temp = np.append(Lambda_temp, Lambda);

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)));
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)));
                tau_array.append(len(Lambda))
                    
        num_lambda = min(len(Lambda_temp), chi, )
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[:, 0] = temp
        # if num_lambda == 0:
        #     print(0)
        # else:
        #     print(np.max(temp))
        self.charge[:, 1, :] = 0;
        self.charge[:num_lambda, 1, 0] = new_charge1[idx]
        self.charge[:num_lambda, 1, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64');
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(smallest_cr_0, largest_cl_0 + 1):
            for j in range(smallest_cr_1, largest_cl_1 + 1):
                tau = i * d + j;
                tau_idx, indices, trash = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

                if len(tau_idx) == 0 or len(l_bond_array[i, j]) == 0 or len(r_bond_array[i, j]) == 0:
                    continue
                V = np.array([Gamma_L_temp[i] for i in tau_idx], dtype = 'complex64')
                W = np.array([Gamma_R_temp[i] for i in tau_idx], dtype = 'complex64')

                V = V.T;

                alpha = np.array(l_bond_array[i, j]).reshape(-1, 1);
                beta = np.arange(len(tau_idx)).reshape(1, -1);

                Gamma1Out[alpha, indices.reshape(1, -1)] = V;

                alpha = np.array(r_bond_array[i, j]).reshape(1, -1);
                beta = np.arange(len(tau_idx)).reshape(-1, 1);

                Gamma2Out[indices.reshape(-1, 1), alpha] = W;

                
        self.A[:, :, 0] = Gamma1Out[:, :]; self.A[:, :, 1] = Gamma2Out[:, :];
        
                
    def MPOtwoqubitUpdateRight(self, UnitaryMPS):
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = np.array([]);
        new_charge1 = np.array([]);
        new_charge2 = np.array([]);
        tau_array = [0];
        idx_L = np.empty([self.d, self.d], dtype = "object")
        idx_R = np.empty([self.d, self.d], dtype = "object")
        idx_C = np.empty([self.d, self.d], dtype = "object")
        len_L = np.zeros([self.d, self.d], dtype = "int32")
        len_R = np.zeros([self.d, self.d], dtype = "int32")
        len_C = np.zeros([self.d, self.d], dtype = "int32")

        l_bond_array = np.empty([self.d, self.d], dtype = "object")
        r_bond_array = np.empty([self.d, self.d], dtype = "object")

        chi = self.chi;
        d = self.d;
        
        UnitaryMPO = np.kron(UnitaryMPS, np.conj(UnitaryMPS));
        
        for i in range(d):
            for j in range(d):
                idx_L[i, j] = np.intersect1d(np.nonzero(self.charge[:, self.n - 2, 0] == i), np.intersect1d(np.nonzero(self.charge[:, self.n - 2, 1] == j), np.nonzero(self.Lambda[:, self.n - 3] > 0)))
                len_L[i, j] = len(idx_L[i, j])
                idx_C[i, j] = np.intersect1d(np.nonzero(self.charge[:, self.n - 1, 0] == i), np.intersect1d(np.nonzero(self.charge[:, self.n - 1, 1] == j), np.nonzero(self.Lambda[:, self.n - 2] > 0)))
                len_C[i, j] = len(idx_C[i, j])
                idx_R[i, j] = np.intersect1d(np.nonzero(self.charge[:, self.n, 0] == i), np.nonzero(self.charge[:, self.n, 1] == j))
                len_R[i, j] = len(idx_R[i, j])
        
        CL = self.charge[:, self.n - 2]
        CR = self.charge[:, self.n]
        valid_idx_l_0 = np.where(CL[:, 0] != self.d)[0]
        valid_idx_l_1 = np.where(CL[:, 1] != self.d)[0]
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
                ch_c = ch_c_1 * d + ch_c_2
                l_charge_1 = np.arange(ch_c_1, d);
                l_charge_2 = np.arange(ch_c_2, d);
                r_charge_1 = np.array([0]);
                r_charge_2 = np.array([0]);
                
                r_bond = [];
                for i in r_charge_1:
                    for j in r_charge_2:
                        r_bond.extend(idx_R[i, j]);

                l_bond = [];
                for i in l_charge_1:
                    for j in l_charge_2:
                        l_bond.extend(idx_L[i, j]);                

                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                

                start = time.time()

                C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                theta = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                L_stack = 0;
                for ch_l_1 in l_charge_1:
                    for ch_l_2 in l_charge_2:
                        if len_L[ch_l_1, ch_l_2] == 0:
                            continue
                        ch_l = ch_l_1 * d + ch_l_2
                        L_stack += len_L[ch_l_1, ch_l_2];
                        R_stack = 0;
                        for ch_r_1 in r_charge_1:
                            for ch_r_2 in r_charge_2:
                                if len_R[ch_r_1, ch_r_2] == 0:
                                    continue
                                ch_r = ch_r_1 * d + ch_r_2
                                R_stack += len_R[ch_r_1, ch_r_2];
                                c_bond = []; c_local = [];
                                for i in np.arange(ch_r_1, ch_l_1 + 1):
                                    for j in np.arange(ch_r_2, ch_l_2 + 1):
                                        c_bond.extend(idx_C[i, j])
                                        #c_local.extend(np.repeat(i * d + j, len_C[i, j]))
                                        c_local.extend([i * d + j] * len_C[i, j])
                                if len(c_bond) == 0:
                                    continue
                                c_bond = np.array(c_bond); c_local = np.array(c_local)
                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += np.matmul(self.A[idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1), self.n - 2], np.matmul(np.diag(UnitaryMPO[ch_l - ch_c, ch_c - ch_r, ch_l - c_local, c_local - ch_r]), self.A[c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1), self.n - 1]))

                theta = C;
                self.theta_time += time.time() - start
                # print(theta)

                start = time.time()

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False);
                W = np.matmul(np.conj(V.T), C);
                self.svd_time += time.time() - start

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];            

                Lambda_temp = np.append(Lambda_temp, Lambda);

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)));
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)));
                tau_array.append(len(Lambda))
                    
        num_lambda = min(len(Lambda_temp), chi, self.d ** 2)
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[:, self.n - 2] = temp
        # if num_lambda == 0:
        #     print(0)
        # else:
        #     print(np.max(temp))
        self.charge[:, self.n - 1, :] = 0;
        self.charge[:num_lambda, self.n - 1, 0] = new_charge1[idx]
        self.charge[:num_lambda, self.n - 1, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64');
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(smallest_cr_0, largest_cl_0 + 1):
            for j in range(smallest_cr_1, largest_cl_1 + 1):
                tau = i * d + j;
                tau_idx, indices, trash = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

                if len(tau_idx) == 0 or len(l_bond_array[i, j]) == 0 or len(r_bond_array[i, j]) == 0:
                    continue
                V = np.array([Gamma_L_temp[i] for i in tau_idx], dtype = 'complex64')
                W = np.array([Gamma_R_temp[i] for i in tau_idx], dtype = 'complex64')

                V = V.T;

                alpha = np.array(l_bond_array[i, j]).reshape(-1, 1);
                beta = np.arange(len(tau_idx)).reshape(1, -1);

                Gamma1Out[alpha, indices.reshape(1, -1)] = V;

                alpha = np.array(r_bond_array[i, j]).reshape(1, -1);
                beta = np.arange(len(tau_idx)).reshape(-1, 1);

                Gamma2Out[indices.reshape(-1, 1), alpha] = W;
        
        self.A[:, :min(chi, d ** 2), self.n - 2] = Gamma1Out[:, :min(chi, d ** 2)]; self.A[:min(chi, d ** 2), 0, self.n - 1] = Gamma2Out[:min(chi, d ** 2), 0];
        
        
            
    def MPOtwoqubitUpdateBulk(self, l, UnitaryMPS):
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = np.array([]);
        new_charge1 = np.array([]);
        new_charge2 = np.array([]);
        tau_array = [0];
        idx_L = np.empty([self.d, self.d], dtype = "object")
        idx_R = np.empty([self.d, self.d], dtype = "object")
        idx_C = np.empty([self.d, self.d], dtype = "object")
        len_L = np.zeros([self.d, self.d], dtype = "int32")
        len_R = np.zeros([self.d, self.d], dtype = "int32")
        len_C = np.zeros([self.d, self.d], dtype = "int32")

        l_bond_array = np.empty([self.d, self.d], dtype = "object")
        r_bond_array = np.empty([self.d, self.d], dtype = "object")

        chi = self.chi;
        d = self.d;
        
        UnitaryMPO = np.kron(UnitaryMPS, np.conj(UnitaryMPS));
        
        for i in range(d):
            for j in range(d):
                idx_L[i, j] = np.intersect1d(np.nonzero(self.charge[:, l, 0] == i), np.intersect1d(np.nonzero(self.charge[:, l, 1] == j), np.nonzero(self.Lambda[:, l - 1] > 0)))
                len_L[i, j] = len(idx_L[i, j])
                idx_C[i, j] = np.intersect1d(np.nonzero(self.charge[:, l + 1, 0] == i), np.intersect1d(np.nonzero(self.charge[:, l + 1, 1] == j), np.nonzero(self.Lambda[:, l] > 0)))
                len_C[i, j] = len(idx_C[i, j])
                idx_R[i, j] = np.intersect1d(np.nonzero(self.charge[:, l + 2, 0] == i), np.intersect1d(np.nonzero(self.charge[:, l + 2, 1] == j), np.nonzero(self.Lambda[:, l + 1] > 0)))
                len_R[i, j] = len(idx_R[i, j])
        
        CL = self.charge[:, l]
        CR = self.charge[:, l + 2]
        valid_idx_l_0 = np.where(CL[:, 0] != self.d)[0]
        valid_idx_l_1 = np.where(CL[:, 1] != self.d)[0]
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
                ch_c = ch_c_1 * d + ch_c_2
                
                l_charge_1 = np.arange(ch_c_1, d);
                l_charge_2 = np.arange(ch_c_2, d);
                r_charge_1 = np.arange(ch_c_1 + 1);
                r_charge_2 = np.arange(ch_c_2 + 1);
                
                r_bond = [];
                for i in r_charge_1:
                    for j in r_charge_2:
                        r_bond.extend(idx_R[i, j]);

                l_bond = [];
                for i in l_charge_1:
                    for j in l_charge_2:
                        l_bond.extend(idx_L[i, j]);                

                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                

                start = time.time()

                C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                theta = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64')
                L_stack = 0;
                for ch_l_1 in l_charge_1:
                    for ch_l_2 in l_charge_2:
                        if len_L[ch_l_1, ch_l_2] == 0:
                            continue
                        ch_l = ch_l_1 * d + ch_l_2
                        L_stack += len_L[ch_l_1, ch_l_2];
                        R_stack = 0;
                        for ch_r_1 in r_charge_1:
                            for ch_r_2 in r_charge_2:
                                if len_R[ch_r_1, ch_r_2] == 0:
                                    continue
                                ch_r = ch_r_1 * d + ch_r_2
                                R_stack += len_R[ch_r_1, ch_r_2];
                                c_bond = []; c_local = [];
                                for i in np.arange(ch_r_1, ch_l_1 + 1):
                                    for j in np.arange(ch_r_2, ch_l_2 + 1):
                                        c_bond.extend(idx_C[i, j])
                                        c_local.extend([i * d + j] * len_C[i, j])
                                if len(c_bond) == 0:
                                    continue
                                c_bond = np.array(c_bond); c_local = np.array(c_local)
                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += np.matmul(self.A[idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1), l], np.matmul(np.diag(UnitaryMPO[ch_l - ch_c, ch_c - ch_r, ch_l - c_local, c_local - ch_r]), self.A[c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1), l + 1]))

                theta = np.matmul(C, np.diag(self.Lambda[r_bond, l + 1]));
                self.theta_time += time.time() - start
                # print(theta)

                start = time.time()

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False);
                V = np.asarray(V)
                Lambda = np.asarray(Lambda)
                W = np.matmul(np.conj(V.T), C);
                self.svd_time += time.time() - start

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];            

                Lambda_temp = np.append(Lambda_temp, Lambda);

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)));
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)));
                tau_array.append(len(Lambda))
                    
        num_lambda = min(len(Lambda_temp), chi)
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[:, l] = temp
        # if num_lambda == 0:
        #     print(0)
        # else:
        #     print(np.max(temp))
        self.charge[:, l + 1, :] = 0;
        self.charge[:num_lambda, l + 1, 0] = new_charge1[idx]
        self.charge[:num_lambda, l + 1, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64');
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(smallest_cr_0, largest_cl_0 + 1):
            for j in range(smallest_cr_1, largest_cl_1 + 1):
                tau = i * d + j;
                tau_idx, indices, trash = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

                if len(tau_idx) == 0 or len(l_bond_array[i, j]) == 0 or len(r_bond_array[i, j]) == 0:
                    continue
                V = np.array([Gamma_L_temp[i] for i in tau_idx], dtype = 'complex64')
                W = np.array([Gamma_R_temp[i] for i in tau_idx], dtype = 'complex64')

                V = V.T;

                alpha = np.array(l_bond_array[i, j]).reshape(-1, 1);
                beta = np.arange(len(tau_idx)).reshape(1, -1);

                Gamma1Out[alpha, indices.reshape(1, -1)] = V;

                alpha = np.array(r_bond_array[i, j]).reshape(1, -1);
                beta = np.arange(len(tau_idx)).reshape(-1, 1);

                Gamma2Out[indices.reshape(-1, 1), alpha] = W;
        
        
        self.A[:, :, l] = Gamma1Out; self.A[:, :, l + 1] = Gamma2Out;

    def MPOtwoqubitUpdate(self, l, r):
        seed = np.random.randint(0, 13579)
        #print(seed)
        self.MPOtwoqubitCombined(l, r, seed)
            
    def MPOtwoqubitCombined(self, l, r, seed):
        np.random.seed(seed)
        start = time.time()
        UnitaryMPS = Rand_BS_MPS(self.d, r)
        self.U_time += time.time() - start
        if l == 0:
            self.MPOtwoqubitUpdateLeft(UnitaryMPS);
        elif l == n - 2:
            self.MPOtwoqubitUpdateRight(UnitaryMPS);
        else:
            self.MPOtwoqubitUpdateBulk(l, UnitaryMPS);

    def RCS1DOneCycleUpdate(self, k):
        if k < self.n / 2:
            temp1 = 2 * k + 1;
            temp2 = 2;
            l = 2 * k;
            while l >= 0:
                if temp1 > 0:
                    T = my_cv.rvs(2 * k + 2, temp1)
                    temp1 -= 2;
                else:
                    T = my_cv.rvs(2 * k + 2, temp2)
                    temp2 += 2;
                self.MPOtwoqubitUpdate(l, np.sqrt(1 - T));
                l -= 1;
        else:
            temp1 = 2 * self.n - (2 * k + 3);
            temp2 = 2;
            l = self.n - 2;
            for i in range(2 * self.n - 2 * k - 2):
                if temp1 >= 0:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp1)
                    temp1 -= 2;
                else:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp2);
                    temp2 += 2;
                self.MPOtwoqubitUpdate(l, np.sqrt(1 - T));
                l -= 1;        
        
    def RCS1DMultiCycle(self):

        start = time.time()

        self.MPOInitialization();        
        self.TotalProbPar[0] = self.TotalProbFromMPO();
        self.EEPar[:, 0] = self.MPOEntanglementEntropy();
        
        alpha_array = [0.5, 0.6, 0.7, 0.8, 0.9];
        
        for i in range(5):
            self.REPar[:, 0, i] = self.MPORenyiEntropy(alpha_array[i]);
        for k in range(self.n - 1):
            if k % 20 == 0:
                print(k)                
            self.RCS1DOneCycleUpdate(k);
            self.TotalProbPar[k + 1] = self.TotalProbFromMPO();
            self.EEPar[:, k + 1] = self.MPOEntanglementEntropy();
            for i in range(5):
                self.REPar[:, k + 1, i] = self.MPORenyiEntropy(alpha_array[i]);
            print("m: {:.2f}. Total time (unreliable): {:.2f}. U time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}.".format(self.m, time.time() - start, self.U_time, self.theta_time, self.svd_time))
            
            #self.REPar[:, k + 1] = self.MPORenyiEntropy();
        
        return self.TotalProbPar, self.EEPar, self.REPar
    
    def TotalProbFromMPO(self):
        R = self.A[:, 0, self.n - 1];
        RTemp = np.copy(R);
        for k in range(self.n - 2):
            idx = np.array([], dtype = 'int32');
            for ch in range(self.d):
                idx = np.append(idx,np.intersect1d(np.nonzero(self.charge[:, self.n - 1 - k, 0] == ch), np.intersect1d(np.nonzero(self.charge[:, self.n - 1 - k, 1] == ch), np.nonzero(self.Lambda[:, self.n - 1 - k - 1] > 0))))
            R = np.matmul(self.A[:, idx, self.n - 1 - k - 1], RTemp[idx].reshape(-1));
            RTemp = np.copy(R);
        idx = np.array([], dtype = 'int32');
        for ch in range(self.d):
            idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[:, 1, 0] == ch), np.intersect1d(np.nonzero(self.charge[:, 1, 1] == ch), np.nonzero(self.Lambda[:, 0] > 0))))
        res = np.matmul(self.A[:, idx, 0], RTemp[idx].reshape(-1))
        return np.sum(res)
        
    
    def MPOEntanglementEntropy(self):      
        Output = np.zeros([self.n - 1]);
        sq_lambda = np.copy(self.Lambda ** 2);
        for i in range(self.n - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[:, i]));
        
        
        #Output = Output * (1 - self.loss ** self.m)
        return Output

    def MPORenyiEntropy(self, alpha = 0.5):
        Output = np.zeros([self.n - 1]);
        sq_lambda = np.copy(self.Lambda ** 2);
        for i in range(self.n - 1):
            Output[i] += RenyiFromColumn(ColumnSumToOne(sq_lambda[:, i]), alpha);
        
        #Output = Output * (1 - self.loss ** self.m)
        return Output
    
    def getProb(self, outcome):
        tot_ch = np.sum(outcome);
        charge = [tot_ch];
        for i in range(len(outcome) - 1):
            charge.append(tot_ch - outcome[i]);
            tot_ch = tot_ch - outcome[i];

        R = self.A[:, 0, self.n - 1];
        RTemp = np.copy(R);            
            
        for k in range(self.n - 1):
            idx = np.array([], dtype = 'int32');
            idx = np.append(idx,np.intersect1d(np.nonzero(self.charge[:, self.n - 1 - k, 0] == charge[self.n - 1 - k]), np.intersect1d(np.nonzero(self.charge[:, self.n - 1 - k, 1] == charge[self.n - 1 - k]), np.nonzero(self.Lambda[:, self.n - 1 - k - 1] > 0))))
            R = np.matmul(self.A[:, idx, self.n - 1 - k - 1], RTemp[idx].reshape(-1));
            RTemp = np.copy(R);
        idx = np.array([], dtype = 'int32');
        idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[:, 0, 0] == np.sum(outcome)), np.nonzero(self.charge[:, 0, 1] == np.sum(outcome))))
        return np.abs(np.sum(RTemp[idx]))

class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1);
my_cv = my_pdf(a = 0, b = 1, name='my_pdf');

def EntropyFromColumn(InputColumn):
    Output = -np.nansum(InputColumn * np.log2(InputColumn))
    return Output;

def RenyiFromColumn(InputColumn, alpha):
    Output = np.log2(np.nansum(InputColumn ** alpha)) / (1 - alpha)
    return Output;

def ColumnSumToOne(InputColumn):
    return InputColumn / np.sum(InputColumn);

def multi_f(args):
    i, n, m, loss, chi = args
    boson = MPO(n, m, loss, chi)
    return boson.RCS1DMultiCycle();

def RCS1DMultiCycleAvg(NumSample, n, m, d, r, loss, chi, errtol = 10 ** (-6), PS = None):
    TotalProbAvg = np.zeros([n]);
    EEAvg = np.zeros([n - 1, n]);
    REAvg = np.zeros([n - 1, n, 5]);

    TotalProbTot = np.zeros([n]);
    EETot = np.zeros([n - 1, n]);
    RETot = np.zeros([n - 1, n, 5]);


    
    for i in range(NumSample):
        print("Sample number", i)
        boson = MPO(n, m, d, r, loss, chi, errtol, PS);
        Totprob, EE, RE = boson.RCS1DMultiCycle();
        TotalProbTot += Totprob;#TotalProbPar[:,i];
        EETot += EE;#EEPar[:,:,i];
        RETot += RE;#EEPar[:,:,i];

    
    TotalProbAvg = TotalProbTot / NumSample;
    EEAvg = EETot / NumSample;
    REAvg = RETot / NumSample;

    return TotalProbAvg,  EEAvg, REAvg

np.random.seed(1)
if __name__ == "__main__":
    t0 = time.time() 
    np.random.seed(1)
    NumSample = 1; n = 10; m = 4; loss = 0.5; chi = 2000; r = 0.2; d = 5; errtol = 10 ** (-7);
    PS = 4;
    Totprob, EE, RE = RCS1DMultiCycleAvg(NumSample, n, m, d, r, loss, chi, errtol, PS);
    print(Totprob)
    print(EE)
    print("Time cost", time.time() - t0)
    #np.save("MPO_EE%d_%d_%d_%d" %(NumSample, n, m, int(100 * loss)), EE);
    #np.save("MPO_RE%d_%d_%d_%d" %(NumSample, n, m, int(100 * loss)), RE);