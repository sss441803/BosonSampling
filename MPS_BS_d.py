'''Full simulation code containing the Device method (cupy, unified update)'''

import numpy as np
import numpy as jnp
import cupy as cp
import itertools

from scipy.stats import rv_continuous

import time

from cuda_kernels import update_MPS, Rand_U


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
        
    def MPSInitialization(self):
        self.dGamma = cp.zeros([self.n, self.chi, self.chi], dtype=complex); # alpha, alpha, I, modes
        self.dLambda = cp.zeros([self.n - 1, self.chi], dtype=float); # alpha, modes - 1
        self.dLambda_edge = cp.ones(self.chi, dtype=float) # edge lambda don't exists and are ones
        self.dcharge = cp.zeros([self.n + 1, self.chi], dtype=int);
        
        for i in range(self.n):
            if i < self.m:
                self.dGamma[i, 0, 0] = 1;
            else:
                self.dGamma[i, 0, 0] = 1;

        for i in range(self.n - 1):
            self.dLambda[i, 0] = 1;
    
        for i in range(self.m):
            self.dcharge[i, :] = self.m - i

    def MPStwoqubitUpdate(self, l, r):
        seed = np.random.randint(0, 13579)
        self.MPStwoqubitUpdateDevice(l, r, seed)

    #MPO update after a two-qudit gate        
    def MPStwoqubitUpdateDevice(self, l, r, seed):
        
        np.random.seed(seed)
        BS = cp.zeros(self.d**3, dtype=complex);
        Rand_U(self.d, r, BS);
        self.unitary_count += 1

        # Determining the location of the two qubit gate
        left = "Left"
        center = "Center"
        right = "Right"
        if l == 0:
            location = left
            Lambda_l = self.dLambda_edge[:]
            Lambda_c = self.dLambda[l,:]
            Lambda_r = self.dLambda[l+1,:]
        elif l == self.n - 2:
            location = right
            Lambda_l = self.dLambda[l-1,:]
            Lambda_c = self.dLambda[l,:]
            Lambda_r = self.dLambda_edge[:]
        else:
            location = center
            Lambda_l = self.dLambda[l-1,:]
            Lambda_c = self.dLambda[l,:]
            Lambda_r = self.dLambda[l+1,:]
        
        Gamma_lc = self.dGamma[l,:]
        Gamma_cr = self.dGamma[l+1,:]
        # charge of corresponding index (bond charge left/center/right)
        bc_l = self.dcharge[l,:]
        bc_c = self.dcharge[l+1,:]
        bc_r = self.dcharge[l+2,:]

        idx_L = []; idx_R = []; idx_C = [];
        Gamma_L_temp = [];
        Gamma_R_temp = [];
        Lambda_temp = cp.array([], dtype=float);
        new_charge = cp.array([], dtype=int);
        l_bond_array = []; r_bond_array = [];
        tau_array = [0];
        
        for j in range(self.d):
            idx_L.append(np.array(list(set.intersection(set(cp.asnumpy(cp.nonzero(bc_l == j)[0])), set(cp.asnumpy(cp.nonzero(Lambda_l > self.errtol)[0])))), dtype=int)) #left hand side charge
            idx_C.append(np.array(list(set.intersection(set(cp.asnumpy(cp.nonzero(bc_c == j)[0])), set(cp.asnumpy(cp.nonzero(Lambda_c > self.errtol)[0])))), dtype=int)) #left hand side charge
            idx_R.append(np.array(list(set.intersection(set(cp.asnumpy(cp.nonzero(bc_r == j)[0])), set(cp.asnumpy(cp.nonzero(Lambda_r > self.errtol)[0])))), dtype=int)) #left hand side charge

        for tau in range(self.d):
            if location == left:
                l_charge = np.array([self.m], dtype=int);
            else:
                l_charge = np.arange(tau, self.d, dtype=int);
            # Possible center site charge
            c_charge = np.arange(self.d, dtype=int)
            # Possible right site charge
            if location == right:
                r_charge = np.array([0], dtype=int)
            else:    
                r_charge = np.arange(tau + 1, dtype=int)
            l_bond = cp.array(list(itertools.chain(*[idx_L[i] for i in l_charge])), dtype=int) # bond index corresponding to some charges
            c_bond = cp.array(list(itertools.chain(*[idx_C[i] for i in c_charge])), dtype=int)
            r_bond = cp.array(list(itertools.chain(*[idx_R[i] for i in r_charge])), dtype=int)
            l_bond_array.append(l_bond)
            r_bond_array.append(r_bond)
            len_l, len_r = len(l_bond), len(r_bond)
            if len_l*len_r == 0:
                tau_array.append(0)
                continue

            C = cp.zeros(len_l*len_r, dtype=complex)
            update_MPS(tau, self.d, chi, l_bond, c_bond, r_bond, bc_l, bc_c, bc_r, C, BS, Lambda_l, Gamma_lc, Lambda_c, Gamma_cr, Lambda_r)
            self.update_count += 1
            
            V, Lambda, W = cp.linalg.svd(C.reshape([len_l, len_r]), full_matrices = False);

            Gamma_L_temp = Gamma_L_temp + [V[:, i] for i in range(len(Lambda))];
            Gamma_R_temp = Gamma_R_temp + [W[i, :] for i in range(len(Lambda))];

            Lambda_temp = cp.append(Lambda_temp, Lambda);
            new_charge = cp.append(new_charge, cp.repeat(cp.array(tau, dtype=int), len(Lambda)));    
            tau_array.append(len(Lambda))
        
        num_lambda = int(min(len(Lambda_temp), self.chi))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx = cp.argpartition(Lambda_temp, -num_lambda)[-num_lambda:]# Indices of the largest chi singular values
        #else:
        #    idx = cp.array([], dtype=int)

        temp = cp.zeros([self.chi], dtype=float);
        temp[:num_lambda] = Lambda_temp[idx]
        #print('gpu: ', temp, Lambda_temp[idx], temp[:num_lambda])
        self.dLambda[l,:] = temp
        self.dcharge[l+1,:num_lambda] = new_charge[idx]
        
        Gamma1Out = cp.zeros([self.chi, self.chi], dtype = complex);
        Gamma2Out = cp.zeros([self.chi, self.chi], dtype = complex);

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        
        for tau in range(self.d): # charge at center
            # Find indices that corresponds to center charge tau that have been selected as the chi largest values
            # The returlen(l_bond_host)ed value "indices" are the indices for idx
            tau_idx, indices, _ = np.intersect1d(cp.asnumpy(idx), np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)
            
            if len(tau_idx) == 0 or len(r_bond_array[tau]) == 0 or len(l_bond_array[tau]) == 0:
                continue
            V = cp.array([Gamma_L_temp[i] for i in tau_idx], dtype=complex)
            W = cp.array([Gamma_R_temp[i] for i in tau_idx], dtype=complex)
            
            V = V.T
 
            alpha = l_bond_array[tau].reshape(-1, 1);
            beta = cp.arange(len(tau_idx), dtype=int).reshape(1, -1);
            Gamma1Out[alpha, indices.reshape(1, -1)] = V[np.arange(len(l_bond_array[tau]), dtype=int).reshape(-1, 1), beta] / (Lambda_l[alpha] if location != left else 1);
            
            beta = beta.reshape(-1, 1);
            alpha = r_bond_array[tau].reshape(1, -1);

            Gamma2Out[indices.reshape(-1, 1), alpha] = W[beta, np.arange(len(r_bond_array[tau]), dtype=int).reshape(1, -1)] / (Lambda_r[alpha] if location != right else 1);

        if location == left:
            self.dGamma[0, 0, :] = Gamma1Out[0, :]; self.dGamma[1, :, :] = Gamma2Out;
        elif location == right:
            self.dGamma[self.n - 2, :, :] = Gamma1Out; self.dGamma[self.n - 1, :, 0] = Gamma2Out[:, 0];
        else:
            self.dGamma[l, :, :] = Gamma1Out; self.dGamma[l + 1, :, :] = Gamma2Out;
        
        return temp


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
        self.TotalProbPar[0] = self.TotalProbFromMPS();
        
        self.EEPar[:, 0] = self.MPSEntanglementEntropy();
        
        for k in range(self.n - 1):
            self.RCS1DOneCycleUpdate(k);
            self.TotalProbPar[k + 1] = self.TotalProbFromMPS();
            self.EEPar[:, k + 1] = self.MPSEntanglementEntropy();
    
        
        return self.TotalProbPar, self.EEPar#, self.REPar
    
    def TotalProbFromMPS(self):
        # Gamma (chi, chi, d, n) # Gamma1 first component is blank
        # Lambda3 (chi, n)
        
        sq_lambda = np.sum(self.dLambda ** 2, axis = 1);
        return np.average(sq_lambda)
    
    
    def MPSEntanglementEntropy(self):
        sq_lambda = np.copy(self.dLambda ** 2);
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

    return boson.RCS1DMultiCycle();

def RCS1DMultiCycleAvg(NumSample, n, m, chi, errtol):
    TotalProbAvg = np.zeros([n], dtype = "float32");
    EEAvg = np.zeros([n - 1, n], dtype = "float32");
    REAvg = np.zeros([n - 1, n, 1], dtype = "float32");
    
    TotalProbTot = np.zeros([n], dtype = "float32");
    EETot = np.zeros([n - 1, n], dtype = "float32");
    RETot = np.zeros([n - 1, n, 1], dtype = "float32");


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
        
    
    TotalProbAvg = TotalProbTot / NumSample;
    #SingleProbAvg = SingleProbTot / NumSample;
    EEAvg = EETot / NumSample;
    #REAvg = RETot / NumSample;
    

    return TotalProbAvg, EEAvg, REAvg

def RenyiFromColumn(InputColumn, alpha):
    Output = np.log2(np.nansum(InputColumn ** alpha)) / (1 - alpha)
    return Output;

class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1);
my_cv = my_pdf(a = 0, b = 1, name='my_pdf');

if __name__ == "__main__":
    m_array = np.arange(10) + 1;
    EE_tot = []
    for m in m_array:
        start = time.time()
        EE_temp = np.zeros((31, 32));
        for i in range(1):
            NumSample = 1; n = 32; chi = 2 ** m; errtol = 10 ** (-10);
            boson = MPS(n, m, chi, errtol);
            Totprob, EE = boson.RCS1DMultiCycle()
            EE_temp += EE
        EE_tot.append(EE_temp / 100)
        print("Time taken: ", time.time() - start)
        print("Unitary and update count: {}, {}".format(boson.unitary_count, boson.update_count))