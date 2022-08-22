'''Full simulation code containing all three methods: Origianl (numpy, left bulk right update), Host (numpy, unified update) with explicit loop of update_MPS_np, and Device (cupy, unified update)'''

import numpy as np
import numpy as jnp
import cupy as cp
import itertools

from scipy.stats import rv_continuous
from scipy.special import factorial, comb

import time

from cuda_kernels import update_MPS, Rand_U
from test_cp import update_MPS_np, Rand_BS_MPS_np

np.random.seed(1)

def Rand_BS_MPS(d, r):
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


class MPS:
    def __init__(self, n, m, chi, errtol):
        self.n = n;
        self.m = m;
        self.d = m + 1;
        self.chi = chi;
        self.errtol = errtol;
        self.TotalProbPar = np.zeros([n]);
        self.SingleProbPar = np.zeros([n]);
        self.EEPar = np.zeros([n - 1,n]);
        self.REPar = np.zeros([n - 1, n, 1], dtype = 'float32');
        self.unitary_count = 0
        self.update_count = 0
        self.U_time = 0
        self.svd_time = 0
        self.theta_time = 0
        self.largest_C = 0
        self.largest_T = 0

    def MPSInitialization(self):
        self.Gamma = np.zeros([self.n, self.chi, self.chi], dtype = 'complex64'); # alpha, alpha, I, modes
        self.Lambda = np.zeros([self.n - 1, self.chi]); # alpha, modes - 1
        self.Lambda_edge = np.ones(self.chi, dtype=float) # edge lambda don't exists and are ones
        self.charge = self.d * np.ones([self.n + 1, self.chi]);
        self.charge[:, 0] = 0
        
        for i in range(self.n):
            if i < self.m:
                self.Gamma[i, 0, 0] = 1;
            else:
                self.Gamma[i, 0, 0] = 1;

        for i in range(self.n - 1):
            self.Lambda[i, 0] = 1;
    
        for i in range(self.m):
            self.charge[i, 0] = self.m - i

    def MPStwoqubitUpdate(self, l, r):
        
        seed = np.random.randint(0, 13579)
        self.MPStwoqubitUpdateOriginal(l, r, seed)

    def MPStwoqubitUpdateLeft(self, UnitaryMPS):
        idx_L = []; idx_R = []; idx_C = [];
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = np.array([]);
        new_charge = np.array([]);
        l_bond_array = []; r_bond_array = [];
        tau_array = [0];
        
        for j in range(self.d):
            idx_L.append(np.nonzero(self.charge[0, :] == j)[0])
            idx_C.append(np.nonzero(self.charge[1, :] == j)[0])
            idx_R.append(np.nonzero(self.charge[2, :] == j)[0])
                
        for tau in range(self.d):

            start = time.time()

            l_charge = np.array([self.m]);
            r_charge = np.arange(tau + 1)
            l_bond = np.array(list(itertools.chain(*[idx_L[i] for i in l_charge]))) # bond index corresponding to some charges
            r_bond = np.array(list(itertools.chain(*[idx_R[i] for i in r_charge])))
            l_bond_array.append(l_bond)
            r_bond_array.append(r_bond)
            if len(l_bond) * len(r_bond) == 0:
                tau_array.append(0)
                continue
            
            self.largest_C = max(max(len(l_bond), len(r_bond)), self.largest_C)
            C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64');
            temp_l = 0;
            for ch_l in l_charge:
                if len(idx_L[ch_l]) == 0:
                    continue
                temp_l += len(idx_L[ch_l]);
                temp_r = 0;
                for ch_r in r_charge:
                    if len(idx_R[ch_r]) == 0:
                        continue
                    temp_r += len(idx_R[ch_r]);
                    for ch_c in range(ch_r, ch_l + 1):
                        C[temp_l - len(idx_L[ch_l]):temp_l, temp_r - len(idx_R[ch_r]):temp_r] += UnitaryMPS[ch_l - tau, tau - ch_r, ch_l - ch_c, ch_c - ch_r] * np.multiply(np.matmul(np.multiply(self.Gamma[0, idx_L[ch_l].reshape(-1, 1), idx_C[ch_c].reshape(1, -1)], self.Lambda[0, idx_C[ch_c]].reshape(1, -1)), self.Gamma[1, idx_C[ch_c].reshape(-1, 1), idx_R[ch_r].reshape(1, -1)]), self.Lambda[1, idx_R[ch_r]].reshape(1, -1))

           #print(C)
                            
            dt = time.time() - start
            self.largest_T = max(dt, self.largest_T)
            self.theta_time += dt

            start = time.time()
            V, Lambda, W = jnp.linalg.svd(C, full_matrices = False);
            self.svd_time += time.time() - start
           #print('SVD out: ', V, W)

            Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
            Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];

            Lambda_temp = np.append(Lambda_temp, Lambda);
            new_charge = np.append(new_charge, np.repeat(tau, len(Lambda)));    
            tau_array.append(len(Lambda))
        
        num_lambda = int(min((Lambda_temp > errtol).sum(), self.chi))
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([self.chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        print(np.sort(Lambda_temp[idx]))
        self.Lambda[0, :] = temp
        self.charge[1, :num_lambda] = new_charge[idx]
        
        Gamma1Out = np.zeros([self.chi, self.chi], dtype = 'complex64');
        Gamma2Out = np.zeros([self.chi, self.chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
        self.Lambda[1][np.where(self.Lambda[1] == 0)[0]] = 999999999999
        
        for tau in range(self.d): # charge at center
            tau_idx, indices, trash = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
            
            if len(tau_idx) == 0 or len(r_bond_array[tau]) == 0 or len(l_bond_array[tau]) == 0:
                continue
            V = np.array([Gamma_L_temp[i] for i in tau_idx])
            W = np.array([Gamma_R_temp[i] for i in tau_idx])
            
            V = V.T
 
            alpha = l_bond_array[tau].reshape(-1, 1);
            beta = np.arange(len(tau_idx)).reshape(1, -1);
            Gamma1Out[alpha, indices.reshape(1, -1)] = V[np.arange(len(l_bond_array[tau])).reshape(-1, 1), beta];
            
            beta = beta.reshape(-1, 1);
            alpha = r_bond_array[tau].reshape(1, -1);

            Gamma2Out[indices.reshape(-1, 1), alpha] = W[beta, np.arange(len(r_bond_array[tau])).reshape(1, -1)] / self.Lambda[1, alpha];
       #print('Gamma out: ', Gamma1Out, Gamma2Out)
                
        self.Gamma[0, 0, :] = Gamma1Out[0, :]; self.Gamma[1, :, :] = Gamma2Out;

        return temp

    def MPStwoqubitUpdateRight(self, UnitaryMPS):
        idx_L = []; idx_R = []; idx_C = [];
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = np.array([]);
        new_charge = np.array([]);
        l_bond_array = []; r_bond_array = [];
        tau_array = [0];
        
        for j in range(self.d):
            idx_L.append(np.nonzero(self.charge[self.n - 2, :] == j)[0])
            idx_R.append(np.nonzero(self.charge[self.n, :] == j)[0])
            idx_C.append(np.nonzero(self.charge[self.n - 1, :] == j)[0])
                
        for tau in range(self.d):
            
            start = time.time()

            l_charge = np.arange(tau, self.d);
            r_charge = np.array([0])
            l_bond = np.array(list(itertools.chain(*[idx_L[i] for i in l_charge]))) # bond index corresponding to some charges
            r_bond = np.array(list(itertools.chain(*[idx_R[i] for i in np.arange(1)])))
            l_bond_array.append(l_bond)
            r_bond_array.append(r_bond)
            if len(l_bond) * len(r_bond) == 0:
                tau_array.append(0)
                continue
            
            self.largest_C = max(max(len(l_bond), len(r_bond)), self.largest_C)
            C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64');
            temp_l = 0;
            for ch_l in l_charge:
                if len(idx_L[ch_l]) == 0:
                    continue
                temp_l += len(idx_L[ch_l]);
                temp_r = 0;
                for ch_r in r_charge:
                    if len(idx_R[ch_r]) == 0:
                        continue
                    temp_r += len(idx_R[ch_r]);
                    for ch_c in range(ch_r, ch_l + 1):
                        C[temp_l - len(idx_L[ch_l]):temp_l, temp_r - len(idx_R[ch_r]):temp_r] += UnitaryMPS[ch_l - tau, tau - ch_r, ch_l - ch_c, ch_c - ch_r] * np.matmul(np.multiply(np.multiply(self.Lambda[self.n - 3, idx_L[ch_l]].reshape(-1, 1), self.Gamma[self.n - 2, idx_L[ch_l].reshape(-1, 1), idx_C[ch_c].reshape(1, -1)]), self.Lambda[self.n - 2, idx_C[ch_c]].reshape(1, -1)), self.Gamma[self.n - 1, idx_C[ch_c].reshape(-1, 1), idx_R[ch_r].reshape(1, -1)])
                            
           #print(C)

            dt = time.time() - start
            self.largest_T = max(dt, self.largest_T)
            self.theta_time += dt

            start = time.time()
            V, Lambda, W = jnp.linalg.svd(C, full_matrices = False);
            self.svd_time += time.time() - start
           #print('SVD out: ', V, W)

            Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
            Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];

            Lambda_temp = np.append(Lambda_temp, Lambda);
            new_charge = np.append(new_charge, np.repeat(tau, len(Lambda)));    
            tau_array.append(len(Lambda))
        
        num_lambda = int(min((Lambda_temp > errtol).sum(), self.chi))
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([self.chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        print(np.sort(Lambda_temp[idx]))
        self.Lambda[self.n - 2, :] = temp
        self.charge[self.n - 1, :num_lambda] = new_charge[idx]
        
        Gamma1Out = np.zeros([self.chi, self.chi], dtype = 'complex64');
        Gamma2Out = np.zeros([self.chi, self.chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
        self.Lambda[self.n - 3][np.where(self.Lambda[self.n - 3] == 0)[0]] = 999999999999
        
        for tau in range(self.d): # charge at center
            tau_idx, indices, trash = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
            
            if len(tau_idx) == 0 or len(r_bond_array[tau]) == 0 or len(l_bond_array[tau]) == 0:
                continue
            V = np.array([Gamma_L_temp[i] for i in tau_idx])
            W = np.array([Gamma_R_temp[i] for i in tau_idx])
            
            V = V.T
            
            i = np.arange(self.d).reshape(1, 1, -1);
 
            alpha = l_bond_array[tau].reshape(-1, 1);
            beta = np.arange(len(tau_idx)).reshape(1, -1);
            Gamma1Out[alpha, indices.reshape(1, -1)] = V[np.arange(len(l_bond_array[tau])).reshape(-1, 1), beta] / self.Lambda[self.n - 3, alpha];
            
            beta = beta.reshape(-1, 1);
            alpha = r_bond_array[tau].reshape(1, -1);

            Gamma2Out[indices.reshape(-1, 1), alpha] = W[beta, np.arange(len(r_bond_array[tau])).reshape(1, -1)];
       #print('Gamma out: ', Gamma1Out, Gamma2Out)
                
        self.Gamma[self.n - 2, :, :] = Gamma1Out; self.Gamma[self.n - 1, :, 0] = Gamma2Out[:, 0];

        return temp

    def MPStwoqubitUpdateBulk(self,l,UnitaryMPS): 
        idx_L = []; idx_R = []; idx_C = [];
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = np.array([]);
        new_charge = np.array([]);
        l_bond_array = []; r_bond_array = [];
        tau_array = [0];
        
        for j in range(self.d):
            idx_L.append(np.nonzero(self.charge[l, :] == j)[0])
            idx_C.append(np.nonzero(self.charge[l + 1, :] == j)[0])
            idx_R.append(np.nonzero(self.charge[l + 2, :] == j)[0])

        mat_list = []
                
        for tau in range(self.d):

            start = time.time()

            l_charge = np.arange(tau, self.d);
            r_charge = np.arange(tau + 1)
            l_bond = np.array(list(itertools.chain(*[idx_L[i] for i in l_charge]))) # bond index corresponding to some charges
            r_bond = np.array(list(itertools.chain(*[idx_R[i] for i in r_charge])))
            l_bond_array.append(l_bond)
            r_bond_array.append(r_bond)
            if len(l_bond) * len(r_bond) == 0:
                tau_array.append(0)
                continue

           #print('inputs: ', self.Gamma[l-1:l+1], self.Lambda[l-1:l+2])
            #time.sleep(0.1)
            
            self.largest_C = max(max(len(l_bond), len(r_bond)), self.largest_C)
            C = np.zeros([len(l_bond), len(r_bond)], dtype = 'complex64');
            temp_l = 0;
            for ch_l in l_charge:
                if len(idx_L[ch_l]) == 0:
                    continue
                temp_l += len(idx_L[ch_l]);
                temp_r = 0;
                for ch_r in r_charge:
                    if len(idx_R[ch_r]) == 0:
                        continue
                    temp_r += len(idx_R[ch_r]);
                    for ch_c in range(ch_r, ch_l + 1):
                        C[temp_l - len(idx_L[ch_l]):temp_l, temp_r - len(idx_R[ch_r]):temp_r] += UnitaryMPS[ch_l - tau, tau - ch_r, ch_l - ch_c, ch_c - ch_r] * np.multiply(np.matmul(np.multiply(np.multiply(self.Lambda[l - 1, idx_L[ch_l]].reshape(-1, 1), self.Gamma[l, idx_L[ch_l].reshape(-1, 1), idx_C[ch_c].reshape(1, -1)]), self.Lambda[l, idx_C[ch_c]].reshape(1, -1)), self.Gamma[l + 1, idx_C[ch_c].reshape(-1, 1), idx_R[ch_r].reshape(1, -1)]), self.Lambda[l + 1, idx_R[ch_r]].reshape(1, -1))
                        #C[temp_l - len(idx_L[ch_l]):temp_l, temp_r - len(idx_R[ch_r]):temp_r] += UnitaryMPS[ch_l - tau, tau - ch_r, ch_l - ch_c, ch_c - ch_r] * np.matmul(np.matmul(np.matmul(np.matmul(np.diag(self.Lambda[idx_L[ch_l], l - 1]), self.Gamma[idx_L[ch_l].reshape(-1, 1), idx_C[ch_c].reshape(1, -1), l]), np.diag(self.Lambda[idx_C[ch_c], l])), self.Gamma[idx_C[ch_c].reshape(-1, 1), idx_R[ch_r].reshape(1, -1), l + 1]), np.diag(self.Lambda[idx_R[ch_r], l + 1]))

           #print(C)

            dt = time.time() - start
            if dt > self.largest_T:
                self.largest_T = dt
            self.theta_time += dt

            start = time.time()
            V, Lambda, W = jnp.linalg.svd(C, full_matrices = False);
            self.svd_time += time.time() - start
           #print('SVD out: ', V, W)

            Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))]
            Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))]

            Lambda_temp = np.append(Lambda_temp, Lambda);
            new_charge = np.append(new_charge, np.repeat(tau, len(Lambda)));    
            tau_array.append(len(Lambda))

            mat_list.append(C)
                #print("appended matrix for tau={}".format(tau))
        
        num_lambda = int(min((Lambda_temp > errtol).sum(), self.chi))
        idx = np.argpartition(Lambda_temp, -num_lambda)[-num_lambda:] # Largest chi singular values
        temp = np.zeros([self.chi]);
        temp[:num_lambda] = Lambda_temp[idx]
        print(np.sort(Lambda_temp[idx]))
        self.Lambda[l, :] = temp
        self.charge[l + 1, :num_lambda] = new_charge[idx]
        
        Gamma1Out = np.zeros([self.chi, self.chi], dtype = 'complex64');
        Gamma2Out = np.zeros([self.chi, self.chi], dtype = 'complex64');

        cum_tau_array = np.cumsum(tau_array)
        self.Lambda[l - 1][np.where(self.Lambda[l - 1] == 0)[0]] = 999999999999
        self.Lambda[l + 1][np.where(self.Lambda[l + 1] == 0)[0]] = 999999999999
        
        for tau in range(self.d): # charge at center
            tau_idx, indices, trash = np.intersect1d(idx, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
            
            if len(tau_idx) == 0 or len(r_bond_array[tau]) == 0 or len(l_bond_array[tau]) == 0:
                continue
            V = np.array([Gamma_L_temp[i] for i in tau_idx])
            W = np.array([Gamma_R_temp[i] for i in tau_idx])
            
            V = V.T
 
            alpha = l_bond_array[tau].reshape(-1, 1);
            beta = np.arange(len(tau_idx)).reshape(1, -1);
            Gamma1Out[alpha, indices.reshape(1, -1)] = V[np.arange(len(l_bond_array[tau])).reshape(-1, 1), beta] / self.Lambda[l - 1, alpha];
            
            beta = beta.reshape(-1, 1);
            alpha = r_bond_array[tau].reshape(1, -1);

            Gamma2Out[indices.reshape(-1, 1), alpha] = W[beta, np.arange(len(r_bond_array[tau])).reshape(1, -1)] / self.Lambda[l + 1, alpha];
       #print('Gamma out: ', Gamma1Out, Gamma2Out)
                
        self.Gamma[l, :, :] = Gamma1Out; self.Gamma[l + 1, :, :] = Gamma2Out;

        return temp

    def MPStwoqubitUpdateOriginal(self, l, r, seed):
        np.random.seed(seed)
        start = time.time()
        UnitaryMPS = Rand_BS_MPS(self.d, r);
        self.U_time += time.time() - start
        if l == 0:
            Lambda = self.MPStwoqubitUpdateLeft(UnitaryMPS);
        elif l == self.n - 2:
            Lambda = self.MPStwoqubitUpdateRight(UnitaryMPS);
        else:
            Lambda = self.MPStwoqubitUpdateBulk(l, UnitaryMPS);
        
        return Lambda



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
                self.MPStwoqubitUpdate(l, np.sqrt(1 - T));
                l -= 1;
        else:
            temp1 = 2 * self.n - (2 * k + 3);
            temp2 = 2;
            l = n - 2;
            for i in range(2 * self.n - 2 * k - 2):
                if temp1 >= 0:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp1)
                    temp1 -= 2;
                else:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp2);
                    temp2 += 2;
                self.MPStwoqubitUpdate(l, np.sqrt(1 - T));
                l -= 1;   
        
    def RCS1DMultiCycle(self):
        self.MPSInitialization();
        #self.TotalProbPar[0] = self.TotalProbFromMPS();
        
        #self.EEPar[:, 0] = self.MPSEntanglementEntropy();
        
        for k in range(self.n - 1):
            #print('k=', k)
            self.RCS1DOneCycleUpdate(k);

            #self.TotalProbPar[k + 1] = self.TotalProbFromMPS();
            #self.EEPar[:, k + 1] = self.MPSEntanglementEntropy();
    
        
        return self.TotalProbPar, self.EEPar#, self.REPar
    
    def TotalProbFromMPS(self):
        # Gamma (chi, chi, d, n) # Gamma1 first component is blank
        # Lambda3 (chi, n)
        
        sq_lambda = np.sum(self.Lambda ** 2, axis = 1);
        return np.average(sq_lambda)
    
    
    def MPSEntanglementEntropy(self):
        sq_lambda = np.copy(self.Lambda ** 2);
        Output = np.zeros([self.n - 1]);

        for i in range(self.n - 1):
            Output[i] = EntropyFromColumn(ColumnSumToOne(sq_lambda[i, :]));

        return Output


def EntropyFromRow(InputRow):
    Output = 0;

    for i in range(len(InputRow)):
        if InputRow[0,i] == 0:
            Output = Output;
        else:
            Output = Output - InputRow[0,i] * np.log2(InputRow[0,i]);

    return Output;

def EntropyFromColumn(InputColumn):
    Output = np.nansum(-InputColumn * np.log2(InputColumn))
    
    return Output;

def ColumnSumToOne(InputColumn):
    return InputColumn / np.sum(InputColumn);

def multi_f(args):
    i, n, m, d, chi, errtol = args
    boson = MPS(n, m, d, chi, errtol)

    return boson.RCS1DMultiCycle()

def RCS1DMultiCycleAvg(NumSample, n, m, chi, errtol):
    TotalProbAvg = np.zeros([n], dtype = "float32")
    EEAvg = np.zeros([n - 1, n], dtype = "float32")
    REAvg = np.zeros([n - 1, n, 1], dtype = "float32")
    
    TotalProbTot = np.zeros([n], dtype = "float32")
    EETot = np.zeros([n - 1, n], dtype = "float32")
    RETot = np.zeros([n - 1, n, 1], dtype = "float32")


    #TotalProbPar = np.zeros([n, NumSample], dtype = "float32");
    #SingleProbPar = np.zeros([n, NumSample], dtype = "float32");
    #EEPar = np.zeros([n - 1, n, NumSample], dtype = "float32");

    
    for i in range(NumSample):
        boson = MPS(n, m, chi, errtol)
        Totprob, EE = boson.RCS1DMultiCycle()
        TotalProbTot = TotalProbTot + Totprob;#TotalProbPar[:,i];
        #SingleProbTot = SingleProbTot + outcome[i][1];#SingleProbPar[:,i];
        EETot = EETot + EE;#EEPar[:,:,i];    
        #RETot = RETot + RE;#EEPar[:,:,i];    
        
    
    TotalProbAvg = TotalProbTot / NumSample
    #SingleProbAvg = SingleProbTot / NumSample;
    EEAvg = EETot / NumSample
    #REAvg = RETot / NumSample;
    

    return TotalProbAvg, EEAvg, REAvg

def RenyiFromColumn(InputColumn, alpha):
    Output = np.log2(np.nansum(InputColumn ** alpha)) / (1 - alpha)
    return Output;

class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1)
my_cv = my_pdf(a = 0, b = 1, name='my_pdf')

if __name__ == "__main__":
    m_array = [6]#, 4, 6 ,8, 10, 12, 14, 16, 18, 20]
    #m_array = [12]
    EE_tot = []
    for m in m_array:
        start = time.time()
        EE_temp = np.zeros((31, 32))
        for i in range(1):
            NumSample = 1; n = 32; chi = 2 ** m; errtol = 10 ** (-5)
            boson = MPS(n, m, chi, errtol)
            Totprob, EE = boson.RCS1DMultiCycle()
            EE_temp += EE
        EE_tot.append(EE_temp / 100)
        #print("Time taken: ", time.time() - start)
        print("m: {:.2f}. Total time: {:.2f}. U time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. Largest array dimension: {:.2f}. Longest time for single matrix: {:.2f}".format(m, time.time()-start, boson.U_time, boson.theta_time, boson.svd_time, boson.largest_C, boson.largest_T))