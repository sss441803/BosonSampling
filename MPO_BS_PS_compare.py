import numpy as np
import numpy as jnp

np.random.seed(1)
np.set_printoptions(precision=3)

from scipy.stats import rv_continuous
from scipy.special import factorial, comb

from itertools import combinations
from filelock import FileLock
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

def change_idx(x, d):
    return d * x[:, 0] + x[:, 1]

def Rand_BS_MPS(d, r):
    # #t = 1 / np.sqrt(2); r = 1 / np.sqrt(2);
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
    # Output = np.array(np.random.rand(d,d,d,d) + 1j*np.random.rand(d,d,d,d), dtype='complex64')
    
    return Output

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

def f(m, k, *arg):
    temp = np.zeros([m], dtype = 'int');
    temp[::k] = 1;
    temp[np.array(*arg)] = 0
    return tuple(temp)

def random_photon_state(n_ph, lost_ph, loss):
    c = np.zeros([2] * n_ph);
    if lost_ph == 0:
        c[tuple([1] * n_ph)] = (1 - loss) ** n_ph;
        return c
    
    if lost_ph == n_ph:
        c[tuple([0] * n_ph)] = loss ** n_ph;
        return c
    
    photon_array = list(combinations(np.arange(n_ph), lost_ph));

    for photon in photon_array:
        c[f(n_ph, 1, np.array(photon))] = loss ** lost_ph * (1 - loss) ** (n_ph - lost_ph); 
    return c

def canonicalize(chi, n_ph, loss, PS):
    #c = np.copy(state)
    A = np.zeros([n_ph, chi, chi], dtype = 'complex64')
    Lambda = np.zeros([n_ph - 1, chi], dtype = 'float32')
    charge = np.zeros([n_ph + 1, chi], dtype = 'int32')
    
    if n_ph == 1:
        c = np.zeros((2, 2));
        if PS == None:
            for lost_ph in range(n_ph + 1):
                c[lost_ph, :] = random_photon_state(n_ph, lost_ph, loss)
                charge[0, lost_ph] = n_ph - lost_ph
        else:
            lost_ph = n_ph - PS
            c[0, :] = random_photon_state(n_ph, lost_ph, loss)
            charge[0, 0] = n_ph - lost_ph
        A[n_ph - 1, 0, 0] = c[0, 1]
        A[n_ph - 1, 1, 0] = c[1, 0]
        return A, Lambda, charge

    
    
    size_ = np.array([n_ph + 1], dtype = 'int32')
    size_ = np.append(size_, [2] * n_ph)
    c = np.zeros(size_);
    if PS == None:
        for lost_ph in range(n_ph + 1):
            c[lost_ph, :, :] = random_photon_state(n_ph, lost_ph, loss)
            charge[0, lost_ph] = n_ph - lost_ph
    else:
        lost_ph = n_ph - PS
        c[0, :, :] = random_photon_state(n_ph, lost_ph, loss)
        charge[0, 0] = n_ph - lost_ph

    c = c.reshape(n_ph + 1, 2, -1)

    pre_tot = n_ph + 1;
    for l in range(n_ph - 1):
        tot = 0; check = 0;
        for tau in range(n_ph + 1):
            l_bond_0 = np.nonzero(charge[l, :pre_tot] - tau == 0)[0][:min(chi, 2 ** (n_ph - l + 1))];
            l_bond_1 = np.nonzero(charge[l, :pre_tot] - tau == 1)[0][:min(chi, 2 ** (n_ph - l + 1))];
            l_bond = np.union1d(l_bond_0, l_bond_1)[:min(chi, 2 ** (n_ph - l))]
            if len(l_bond) == 0:
                continue
            c_temp = np.vstack((c[l_bond_0, 0, :], c[l_bond_1, 1, :]))
            u, v, w = jnp.linalg.svd(c_temp, full_matrices = False);
            
            len_ = int(np.sum(v > 10 ** (-10)));
            if len_ == 0:
                continue
            tot += len_;
            
            charge[l + 1, tot - len_:tot] = tau;
            if check == 0:
                temp_w = w[:len_, :];
                check = 1;
            else:
                temp_w = np.vstack((temp_w, w[:len_, :]))
            
            Lambda[l, tot - len_:tot] = v[:len_]

            if len(l_bond_0) > 0:
                u0 = u[:len(l_bond_0), :len_]
                A[l, l_bond_0, tot - len_:tot] = u0;
            if len(l_bond_1) > 0:
                u1 = u[len(l_bond_0):len(l_bond_0) + len(l_bond_1), :len_]
                A[l, l_bond_1, tot - len_:tot] = u1;
        if l == n_ph - 2 :
            continue
        c = np.matmul(np.diag(Lambda[l, :tot]), temp_w).reshape(tot, 2, -1);
        pre_tot = tot;
    
    c = np.matmul(np.diag(Lambda[l, :tot]), temp_w).reshape(tot, 2);
    if tot == 1:
        #print("here")
        A[n_ph - 1, 0, 0] = np.sum(c);
    elif charge[0, n_ph] == 0:
        A[n_ph - 1, 0, 0] = c[0, 0]
        A[n_ph - 1, 1, 0] = c[1, 1]
    else:
        #print("warning")
        A[n_ph - 1, 1, 0] = c[0, 0]
        A[n_ph - 1, 0, 0] = c[1, 1]
        
    return A, Lambda, charge


class MPO:
    def __init__(self, n, m, d, loss, init_chi, chi, errtol = 10 ** (-6), PS=None):
        self.n = n;
        self.m = m;
        self.d = d
        self.loss = loss;
        if PS != None:
            self.init_chi = init_chi
        else:
            self.init_chi = chi
        self.chi = chi;
        self.errtol = errtol
        self.PS = PS
        self.TotalProbPar = np.zeros([n], dtype = 'float32');
        self.SingleProbPar = np.zeros([n], dtype = 'float32');
        self.EEPar = np.zeros([n - 1, n], dtype = 'float32');
        self.u_time = 0
        self.start_time = 0
        self.theta_time = 0
        self.svd_time = 0
        self.post_time = 0
        self.cycle_time = 0

    def MPOInitialization(self):
        self.Lambda = np.zeros([self.n - 1, self.chi], dtype = 'float32');
        self.A = np.zeros([self.n, self.chi, self.chi], dtype = 'complex64');
        self.charge = np.zeros([self.n + 1, self.chi, 2], dtype = 'int32');
                
        chi = self.chi; d = self.d; errtol =  self.errtol;
        
        
        A, Lambda, charge = canonicalize(chi, self.m, self.loss, self.PS);
        
        self.A[:self.m, :, :] = A;
        self.Lambda[:self.m - 1, :] = Lambda;
        self.charge[:self.m + 1, :, 0] = charge;
        self.charge[:self.m + 1, :, 1] = charge;
        
        for i in range(self.m, self.n):
            self.A[i, 0, 0] = 1;

        for i in range(self.m - 1, self.n - 1):
            self.Lambda[i, 0] = 1;

        self.normalization = np.real(self.TotalProbFromMPO())
        print('Total probability normalization factor: ', self.normalization)

        print('Canonicalization update')
        for l in range(self.n - 1):
            self.MPOtwoqubitCombined(l, 0)

    def MPOInitialization1(self):
        
        self.Lambda = np.zeros([self.chi, self.n - 1], dtype = 'float32');
        self.A = np.zeros([self.chi, self.chi, self.n], dtype = 'complex64');
        self.Gamma = np.zeros([self.chi, self.chi, self.n], dtype = 'complex64');        
        self.charge = np.zeros([self.chi, self.n + 1, 2], dtype = 'int32');

        #MPS Part
        chi = self.chi; d = self.d; K = self.m; errtol =  self.errtol; loss = self.loss;
        
        rho = np.zeros([d, d], dtype = 'complex64')
        rho[0, 0] = loss
        rho[1, 1] = 1 - loss

        if self.PS == None:
            for i in range(d):
                self.charge[i, 0, 0] = i;
                self.charge[i, 0, 1] = i;
            pre_chi = d;
        else:
            self.charge[0, 0, 0] = self.PS;
            self.charge[0, 0, 1] = self.PS;
            pre_chi = 1;

        for i in range(K - 1):
            chi_ = 0;
            for j in range(pre_chi):
                for ch_diff1 in range(self.charge[j, i, 0] + 1):
                    for ch_diff2 in range(self.charge[j, i, 1] + 1):
                        if np.abs(rho[ch_diff1, ch_diff2]) <= errtol:
                            continue
                        self.Gamma[j, chi_, i] = rho[ch_diff1, ch_diff2];
                        self.charge[chi_, i + 1, 0] = self.charge[j, i, 0] - ch_diff1;
                        self.charge[chi_, i + 1, 1] = self.charge[j, i, 1] - ch_diff2;                
                        chi_ += 1;
            self.Lambda[:chi_, i] = 1;
            pre_chi = chi_;

        for j in range(pre_chi):
            self.Gamma[j, 0, K - 1] = rho[self.charge[j, K - 1, 0], self.charge[j, K - 1, 1]]
        
        for i in range(self.m - 1, self.n - 1):
            self.Lambda[0, i] = 1;
        
        for i in range(self.m):
            self.A[:, :, i] = self.Gamma[:, :, i] @ np.diag(self.Lambda[:, i]);
        
        for i in range(self.m, self.n):
            self.A[0, 0, i] = 1;

        print('Array transposition')
        self.A = np.transpose(self.A, (2, 0, 1))
        self.Lambda = np.transpose(self.Lambda, (1, 0))
        self.charge = np.transpose(self.charge, (1, 0, 2))

        self.normalization = np.real(self.TotalProbFromMPO())
        print('Total probability normalization factor: ', self.normalization)
        # print(self.A, self.Lambda, self.charge)

        print('Canonicalization update')
        for l in range(self.n - 1):
            self.MPOtwoqubitCombined(l, 0)
        
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
                idx_L[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[0, :, 0] == i)[0]), set(np.nonzero(self.charge[0, :, 1] == j)[0]))), dtype = "int32")
                len_L[i, j] = len(idx_L[i, j])
                idx_C[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[1, :, 0] == i)[0]), set(np.nonzero(self.charge[1, :, 1] == j)[0]), set(np.nonzero(self.Lambda[0, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_C[i, j] = len(idx_C[i, j])
                idx_R[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[2, :, 0] == i)[0]), set(np.nonzero(self.charge[2, :, 1] == j)[0]), set(np.nonzero(self.Lambda[0 + 1, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_R[i, j] = len(idx_R[i, j])
        
        
        c_bond_list = []; c_local_list = [];
        for ch_l_1 in range(d):
            for ch_l_2 in range(d):
                for ch_r_1 in range(d):
                    for ch_r_2 in range(d):
                        temp0 = idx_C[ch_r_1: ch_l_1 + 1, ch_r_2: ch_l_2 + 1].ravel()
                        if len(temp0) == 0:
                            c_bond_list.append([])
                            c_local_list.append(0)
                        else:
                            temp = np.hstack(temp0)
                            c_bond_list.append(temp)
                            c_local_list.append(np.repeat((np.arange(ch_r_1, ch_l_1 + 1).reshape(-1, 1) * d + np.arange(ch_r_2, ch_l_2 + 1).reshape(1, -1)).reshape(-1), len_C[ch_r_1: ch_l_1 + 1, ch_r_2: ch_l_2 + 1].reshape(-1)))

        
        for ch_c_1 in range(d):
            for ch_c_2 in range(d):
                ch_c = ch_c_1 * d + ch_c_2
                l_charge_1 = np.arange(ch_c_1, d);
                l_charge_2 = np.arange(ch_c_2, d);
                r_charge_1 = np.arange(ch_c_1 + 1);
                r_charge_2 = np.arange(ch_c_2 + 1);
                
                r_bond = np.hstack(idx_R[r_charge_1.reshape(-1, 1), r_charge_2.reshape(1, -1)].flatten())
                l_bond = np.hstack(idx_L[l_charge_1.reshape(-1, 1), l_charge_2.reshape(1, -1)].flatten())

                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                

                start = time.time()

                #print('glc: {}, gcr: {}.'.format(self.A[0, l_bond.reshape(-1, 1)], self.A[1, :, r_bond.reshape(1, -1)]))

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
                                c_bond = c_bond_list[d ** 3 * ch_l_1 + d ** 2 * ch_l_2 + d * ch_r_1 + ch_r_2]
                                if len(c_bond) == 0:
                                    continue
                                c_local = c_local_list[d ** 3 * ch_l_1 + d ** 2 * ch_l_2 + d * ch_r_1 + ch_r_2]

                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += self.A[0, idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1)] @ np.multiply(UnitaryMPO[ch_l - ch_c, ch_c - ch_r, ch_l - c_local, c_local - ch_r].reshape(-1, 1), self.A[1, c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1)])

                #print(C)

                theta = np.multiply(C, self.Lambda[1, r_bond].reshape(1, -1));
                
                self.theta_time += time.time() - start
                
                start = time.time()

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False);
                #print('V: ', V)

                self.svd_time += time.time() - start

                W = np.matmul(np.conj(V.T), C);

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];            

                Lambda_temp = np.append(Lambda_temp, Lambda);

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)));
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)));
                tau_array.append(len(Lambda))

        start = time.time()

        num_lambda = min(len(Lambda_temp), chi, )
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[0, :] = temp
        self.charge[1, :, :] = 0;
        self.charge[1, :num_lambda, 0] = new_charge1[idx]
        self.charge[1, :num_lambda, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64');
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(d):
            for j in range(d):
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

                
        self.A[0, :, :] = Gamma1Out[:, :]; self.A[1, :, :] = Gamma2Out[:, :];
        #print('Gamma: ', Gamma1Out, Gamma2Out)

        self.post_time += time.time() - start
        
                
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
                idx_L[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[self.n - 2, :, 0] == i)[0]), set(np.nonzero(self.charge[self.n - 2, :, 1] == j)[0]), set(np.nonzero(self.Lambda[self.n - 3, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_L[i, j] = len(idx_L[i, j])
                idx_C[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[self.n - 1, :, 0] == i)[0]), set(np.nonzero(self.charge[self.n - 1, :, 1] == j)[0]), set(np.nonzero(self.Lambda[self.n - 2, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_C[i, j] = len(idx_C[i, j])
                idx_R[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[self.n, :, 0] == i)[0]), set(np.nonzero(self.charge[self.n, :, 1] == j)[0]))), dtype = "int32")
                len_R[i, j] = len(idx_R[i, j])
        
        c_bond_list = []; c_local_list = [];
        for ch_l_1 in range(d):
            for ch_l_2 in range(d):
                for ch_r_1 in range(d):
                    for ch_r_2 in range(d):
                        temp0 = idx_C[ch_r_1: ch_l_1 + 1, ch_r_2: ch_l_2 + 1].ravel()
                        if len(temp0) == 0:
                            c_bond_list.append([])
                            c_local_list.append(0)
                        else:
                            temp = np.hstack(temp0)
                            c_bond_list.append(temp)
                            c_local_list.append(np.repeat((np.arange(ch_r_1, ch_l_1 + 1).reshape(-1, 1) * d + np.arange(ch_r_2, ch_l_2 + 1).reshape(1, -1)).reshape(-1), len_C[ch_r_1: ch_l_1 + 1, ch_r_2: ch_l_2 + 1].reshape(-1)))

        
        
        
        for ch_c_1 in range(d):
            for ch_c_2 in range(d):
                ch_c = ch_c_1 * d + ch_c_2
                l_charge_1 = np.arange(ch_c_1, d);
                l_charge_2 = np.arange(ch_c_2, d);
                r_charge_1 = np.array([0]);
                r_charge_2 = np.array([0]);
                
                r_bond = np.hstack(idx_R[r_charge_1.reshape(-1, 1), r_charge_2.reshape(1, -1)].flatten())
                l_bond = np.hstack(idx_L[l_charge_1.reshape(-1, 1), l_charge_2.reshape(1, -1)].flatten())

                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                

                start = time.time()

                #print('glc: {}, gcr: {}.'.format(self.A[self.n-2, l_bond.reshape(-1, 1)], self.A[self.n-1, :, r_bond.reshape(1, -1)]))

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
                                c_bond = c_bond_list[d ** 3 * ch_l_1 + d ** 2 * ch_l_2 + d * ch_r_1 + ch_r_2]
                                if len(c_bond) == 0:
                                    continue
                                c_local = c_local_list[d ** 3 * ch_l_1 + d ** 2 * ch_l_2 + d * ch_r_1 + ch_r_2]
                                
                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += self.A[self.n - 2, idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1)] @ np.multiply(UnitaryMPO[ch_l - ch_c, ch_c - ch_r, ch_l - c_local, c_local - ch_r].reshape(-1, 1), self.A[self.n - 1, c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1)])

                #print(C)

                theta = C
                #print('T: ', theta)
                
                self.theta_time += time.time() - start
                
                start = time.time()

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False);

                self.svd_time += time.time() - start

                W = np.matmul(np.conj(V.T), C);

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];            

                Lambda_temp = np.append(Lambda_temp, Lambda);

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)));
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)));
                tau_array.append(len(Lambda))
                    
        start = time.time()

        num_lambda = min(len(Lambda_temp), chi, self.d ** 2)
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[self.n - 2 :] = temp
        self.charge[self.n - 1, :, :] = 0;
        self.charge[self.n - 1, :num_lambda, 0] = new_charge1[idx]
        self.charge[self.n - 1, :num_lambda, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64');
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(d):
            for j in range(d):
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


        #print('Gamma: ', Gamma1Out, Gamma2Out)
        self.A[self.n - 2, :, :min(chi, d ** 2)] = Gamma1Out[:, :min(chi, d ** 2)]; self.A[self.n - 1, :min(chi, d ** 2), 0] = Gamma2Out[:min(chi, d ** 2), 0];
        
        self.post_time += time.time() - start
        
            
    def MPOtwoqubitUpdateBulk(self, l, UnitaryMPS):
        start = time.time()

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
                idx_L[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[l, :, 0] == i)[0]), set(np.nonzero(self.charge[l, :, 1] == j)[0]), set(np.nonzero(self.Lambda[l - 1, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_L[i, j] = len(idx_L[i, j])
                idx_R[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[l + 2, :, 0] == i)[0]), set(np.nonzero(self.charge[l + 2, :, 1] == j)[0]), set(np.nonzero(self.Lambda[l + 1, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_R[i, j] = len(idx_R[i, j])
                idx_C[i, j] = np.array(list(set.intersection(set(np.nonzero(self.charge[l + 1, :, 0] == i)[0]), set(np.nonzero(self.charge[l + 1, :, 1] == j)[0]), set(np.nonzero(self.Lambda[l, :] > 10 ** (-10))[0]))), dtype = "int32")
                len_C[i, j] = len(idx_C[i, j])
                
        c_bond_list = []; c_local_list = [];
        for ch_l_1 in range(d):
            for ch_l_2 in range(d):
                for ch_r_1 in range(d):
                    for ch_r_2 in range(d):
                        temp0 = idx_C[ch_r_1: ch_l_1 + 1, ch_r_2: ch_l_2 + 1].ravel()
                        if len(temp0) == 0:
                            c_bond_list.append([])
                            c_local_list.append(0)
                        else:
                            temp = np.hstack(temp0)
                            c_bond_list.append(temp)
                            c_local_list.append(np.repeat((np.arange(ch_r_1, ch_l_1 + 1).reshape(-1, 1) * d + np.arange(ch_r_2, ch_l_2 + 1).reshape(1, -1)).reshape(-1), len_C[ch_r_1: ch_l_1 + 1, ch_r_2: ch_l_2 + 1].reshape(-1)))
        
        self.start_time += time.time() - start

        for ch_c_1 in range(d):
            for ch_c_2 in range(d):
                ch_c = ch_c_1 * d + ch_c_2
                
                l_charge_1 = np.arange(ch_c_1, d);
                l_charge_2 = np.arange(ch_c_2, d);
                r_charge_1 = np.arange(ch_c_1 + 1);
                r_charge_2 = np.arange(ch_c_2 + 1);
                
                r_bond = np.hstack(idx_R[r_charge_1.reshape(-1, 1), r_charge_2.reshape(1, -1)].flatten())
                l_bond = np.hstack(idx_L[l_charge_1.reshape(-1, 1), l_charge_2.reshape(1, -1)].flatten())
                
                        
                l_bond_array[ch_c_1, ch_c_2] = l_bond
                r_bond_array[ch_c_1, ch_c_2] = r_bond


                if len(l_bond) == 0 or len(r_bond) == 0:
                    tau_array.append(0)
                    continue                
                
                start = time.time()

                #print('glc: {}, gcr: {}.'.format(self.A[l, l_bond.reshape(-1, 1)], self.A[l+1, :, r_bond.reshape(1, -1)]))

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
                                c_bond = c_bond_list[d ** 3 * ch_l_1 + d ** 2 * ch_l_2 + d * ch_r_1 + ch_r_2]
                                if len(c_bond) == 0:
                                    continue
                                c_local = c_local_list[d ** 3 * ch_l_1 + d ** 2 * ch_l_2 + d * ch_r_1 + ch_r_2]
                                C[L_stack - len_L[ch_l_1, ch_l_2]:L_stack, R_stack - len_R[ch_r_1, ch_r_2]:R_stack] += self.A[l, idx_L[ch_l_1, ch_l_2].reshape(-1, 1), c_bond.reshape(1, -1)] @ np.multiply(UnitaryMPO[ch_l - ch_c, ch_c - ch_r, ch_l - c_local, c_local - ch_r].reshape(-1, 1), self.A[l + 1, c_bond.reshape(-1, 1), idx_R[ch_r_1, ch_r_2].reshape(1, -1)])

                #print(C)

                theta = np.multiply(C, self.Lambda[l + 1, r_bond].reshape(1, -1));
                #print('T: ', theta)
                
                self.theta_time += time.time() - start
                
                start = time.time()

                V, Lambda, W = jnp.linalg.svd(theta, full_matrices = False);

                self.svd_time += time.time() - start

                V = np.asarray(V)
                Lambda = np.asarray(Lambda)
                W = np.matmul(np.conj(V.T), C);

                Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
                Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];            

                Lambda_temp = np.append(Lambda_temp, Lambda);

                new_charge1 = np.append(new_charge1, np.repeat(ch_c_1, len(Lambda)));
                new_charge2 = np.append(new_charge2, np.repeat(ch_c_2, len(Lambda)));
                tau_array.append(len(Lambda))
                    
        start = time.time()

        num_lambda = min(len(Lambda_temp), chi)
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        self.Lambda[l, :] = temp
        self.charge[l + 1, :, :] = 0;
        self.charge[l + 1, :num_lambda, 0] = new_charge1[idx]
        self.charge[l + 1, :num_lambda, 1] = new_charge2[idx]
        
        Gamma1Out = np.zeros([chi, chi], dtype = 'complex64');
        Gamma2Out = np.zeros([chi, chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
                    
        for i in range(d):
            for j in range(d):
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
                    
                #self.A[alpha, indices.reshape(1, -1), l] = V;
                
                alpha = np.array(r_bond_array[i, j]).reshape(1, -1);
                beta = np.arange(len(tau_idx)).reshape(-1, 1);

                #self.A[indices.reshape(-1, 1), alpha, l + 1] = W
                
                Gamma2Out[indices.reshape(-1, 1), alpha] = W;
        
        #print('Gamma: ', Gamma1Out, Gamma2Out)
        self.A[l, :, :] = Gamma1Out; self.A[l + 1, :, :] = Gamma2Out;

        self.post_time += time.time() - start
            
    def MPOtwoqubitCombined(self, l, r):
        seed = np.random.randint(0, 13579)
        np.random.seed(seed)
        start = time.time()
        UnitaryMPS = Rand_BS_MPS(self.d, r)
        self.u_time += time.time() - start
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
                self.MPOtwoqubitCombined(l, np.sqrt(1 - T));
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
                self.MPOtwoqubitCombined(l, np.sqrt(1 - T));
                l -= 1;        
        
    def RCS1DMultiCycle(self):     
        self.TotalProbPar[0] = self.TotalProbFromMPO();
        self.EEPar[:, 0] = self.MPOEntanglementEntropy();
        
        for k in range(self.n - 1):
            self.RCS1DOneCycleUpdate(k);
            print('u, start, theta, svd, post', self.u_time, self.start_time, self.theta_time, self.svd_time, self.post_time)
            self.TotalProbPar[k + 1] = self.TotalProbFromMPO();
            self.EEPar[:, k + 1] = self.MPOEntanglementEntropy();
        
        return self.TotalProbPar, self.EEPar
    
    def TotalProbFromMPO(self):
        R = self.A[self.n - 1, :, 0];
        RTemp = np.copy(R);
        for k in range(self.n - 2):
            idx = np.array([], dtype = 'int32');
            for ch in range(self.d):
                idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 0] == ch), np.intersect1d(np.nonzero(self.charge[self.n - 1 - k, :, 1] == ch), np.nonzero(self.Lambda[self.n - 1 - k - 1, :] > 0))))
            R = np.matmul(self.A[self.n - 1 - k - 1, :, idx].T, RTemp[idx].reshape(-1));
            RTemp = np.copy(R);
        idx = np.array([], dtype = 'int32');
        for ch in range(self.d):
            idx = np.append(idx, np.intersect1d(np.nonzero(self.charge[1, :, 0] == ch), np.intersect1d(np.nonzero(self.charge[1, :, 1] == ch), np.nonzero(self.Lambda[0, :] > 0))))
        res = np.matmul(self.A[0, :, idx].T, RTemp[idx].reshape(-1))
        return np.sum(res)
        
    
    def MPOEntanglementEntropy(self):      
        Output = np.zeros([self.n - 1]);
        sq_lambda = np.copy(self.Lambda ** 2);
        for i in range(self.n - 1):
            Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i, :]));
        
        
        #Output = Output * (1 - self.loss ** self.m)
        return Output


def RCS1DMultiCycleAvg(n, m, d, loss, chi, PS):
    init_chi = d ** 2
    boson = MPO(n, m, d, loss, init_chi, chi, PS=PS)
    boson.MPOInitialization();
    if boson.normalization > 0:
        return boson.RCS1DMultiCycle()
    else:
        print('Failed')
        quit()


if __name__ == "__main__":
    exp_idx_beginning = 0
    while True:
    # Loop until all experiments are over
        time.sleep(np.random.rand(1).item())
        with FileLock("./experiment.pickle.lock"):
            # work with the file as it is now locked
            print("Lock acquired.")
            with open("./experiment.pickle", 'rb') as experiment_file:
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
                quit()
            else:
                n, m, beta, loss, PS = experiment['n'], experiment['m'], experiment['beta'], experiment['loss'], experiment['PS']
                d = PS + 1
                chi = int(max(8*2**d, 256))
                begin_dir = './SPBSPSResults/n_{}_m_{}_beta_{}_loss_{}_PS_{}_'.format(n, m, beta, loss, PS)
                # begin_dir = './SPBSPSNoneResults/n_{}_m_{}_beta_{}_loss_{}_'.format(n, m, beta, loss)
                if os.path.isfile(begin_dir + 'chi.npy'):
                    chi_array = np.load(begin_dir + 'chi.npy')
                    chi = int(np.max(chi_array))
                    prob = np.load(begin_dir + 'chi_{}_Totprob.npy'.format(chi))
                    prob = prob[np.where(prob > 0)[0]]
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

                print('Loss: {}. Chi: {}'.format(loss, chi))
                
                if status == 'run':
                    if chi > 5000:
                        print('Required bond-dimension chi too large. Moving on to next experiment.')
                        status = 'skip'
                    # elif n > 100:
                    #     print('Too many modes. Moving on to next experiment.')
                    #     status = 'skip'
                    else:
                        # Will run the first found incomplete experiment, set status to in progress
                        experiments[exp_idx]['status'] = 'in progress'
                        # Update experiment track file
                        with open('./experiment.pickle', 'wb') as file:
                            pickle.dump(experiments, file)
                        status = 'run'

        if status == 'skip':
            continue

        t0 = time.time()
        # errtol = 10 ** (-7)   
        # n, m, beta, errtol = 20, 5, 1.0, 10**(-7)
        # ideal_ave_photons = beta * m
        # lossy_ave_photons = 0.5 * ideal_ave_photons
        # loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
        # PS = int((1-loss)*m)
        # PS += 1
        # d = PS + 1
        # init_chi = d**2
        # chi = int(max(32*2**PS, d**2, 512))
        # print(n, m, beta, loss, PS, d, chi)

        if not os.path.isfile(begin_dir + 'EE.npy'):
        # if True:
            t0 = time.time()
            try:
                # PS = None
                # d = m+1
                # chi *= 4
                Totprob, EE = RCS1DMultiCycleAvg(n, m, d, loss, chi, PS)
                print(Totprob)
                # print(EE)
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
            except ValueError:
                print('Bad initialization. Next experiment.')
        else:
            print("Simulation already ran.")
        # break