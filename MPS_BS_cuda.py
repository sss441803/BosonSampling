'''Full simulation code containing the Device method (cupy, unified update)'''
import numpy as np
import cupy as cp

from scipy.stats import rv_continuous

import time

from cuda_kernels import update_MPS, Rand_U
from sort import Aligner

np.random.seed(1)





# Checking if GPU computed results agree with the CPU results
def loop(d, tau, U, CL, CC, CR, LL, Glc, LC, Gcr, LR):

    U, CL, CC, CR, LL, Glc, LC, Gcr, LR = map(cp.asnumpy, [U, CL, CC, CR, LL, Glc, LC, Gcr, LR])

    m, n, k = CL.shape[0], CR.shape[0], CC.shape[0]
    T = np.zeros([m, n], dtype = data_type)

    for i in range(m):

        cl = CL[i]
        ll = LL[i]

        for j in range(n):
            cr = CR[j]
            lr = LR[j]
            result = 0

            for p in range(k):
                cc = CC[p]
                
                if (cl >= cc) and (cc >= cr):
                    u = U[cl - tau, tau - cr, cl - cc]
                    glc = Glc[i, p]
                    gcr = Gcr[p, j]
                    lc = LC[p]
                    result += u * glc * gcr * lc
                    if np.isnan(result):
                        print(Glc, Gcr)
                        print(i,j,p,u, glc, gcr, lc)
                        quit()

            result *= ll * lr
            T[i, j] = result

    return T


data_type = cp.float32
float_type = np.float32


class MPS:
    def __init__(self, n, m, chi, errtol):
        self.n = n # Number of modes
        self.m = m # Number of photons
        self.d = int(m + 1) # Local Hilbert space dimension
        self.chi = chi # Maximum bond dimension
        self.errtol = errtol # Error tolerance. Singular values smaller than which are eliminated
        self.TotalProbPar = np.zeros([n])
        self.SingleProbPar = np.zeros([n])
        self.EEPar = np.zeros([n - 1,n])
        self.REPar = np.zeros([n - 1, n, 1], dtype = 'float32')
        
    def MPSInitialization(self):
        self.dGamma = cp.zeros([self.n, self.chi, self.chi], dtype=data_type); # modes, alpha, alpha
        self.dLambda = cp.zeros([self.n - 1, self.chi], dtype=float_type); # modes - 1, alpha
        self.dLambda_edge = cp.ones(self.chi, dtype=float_type) # edge lambda (for first and last site) don't exists and are ones
        self.dcharge = self.d * cp.ones([self.n + 1, self.chi], dtype=np.int32) # Initialize initial charges for all bonds to the impossible charge d.
        
        # Initialize the first bond
        # At the beginning, only the first bond is initialized to non-trivial information
        for i in range(self.n):
            self.dGamma[i, 0, 0] = 1 
        
        for i in range(self.n - 1):
            self.dLambda[i, 0] = 1
        
        self.dcharge[:, 0] = 0 # First bond has charge zero
        for i in range(self.m):
            self.dcharge[i, 0] = self.m - i # Since each mode has a photon for the first m modes, the charge number (photons to the right) increases by one per site

    # Gives the range of left, center and right hand side charge values when center charge is fixed to tau
    def charge_range(self, location, tau):
        # Speficying allowed left and right charges
        if location == 'left':
            min_charge_l = max_charge_l = self.m # The leftmost site must have all photons to the right, hence charge can only be m
        else:
            min_charge_l, max_charge_l = tau, self.d - 1 # Left must have more or equal photons to its right than center
        # Possible center site charge
        min_charge_c, max_charge_c = 0, self.d - 1 # The center charge is summed so returns 0 and maximum possible charge.
        # Possible right site charge
        if location == 'right':
            min_charge_r = max_charge_r = 0 # The rightmost site must have all photons to the left, hence charge can only be 0
        else:    
            min_charge_r, max_charge_r = 0, tau # Left must have more or equal photons to its right than center
        
        return min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r

    def MPStwoqubitUpdate(self, l, r):
        seed = np.random.randint(0, 13579)
        self.MPStwoqubitUpdateDevice(l, r, seed)

    #MPO update after a two-qudit gate        
    def MPStwoqubitUpdateDevice(self, l, r, seed):
        
        cp.random.seed(seed)
        # U = cp.zeros(self.d**3, dtype=data_type)
        # Rand_U(self.d, r, U)
        # U = U.reshape(self.d, self.d, self.d)
        U = cp.random.rand(self.d, self.d, self.d, dtype=cp.float32)

        # Determining the location of the two qubit gate
        left = "Left"
        center = "Center"
        right = "Right"
        if l == 0:
            location = left
            LL = self.dLambda_edge[:]
            LC = self.dLambda[l,:]
            LR = self.dLambda[l+1,:]
        elif l == self.n - 2:
            location = right
            LL = self.dLambda[l-1,:]
            LC = self.dLambda[l,:]
            LR = self.dLambda_edge[:]
        else:
            location = center
            LL = self.dLambda[l-1,:]
            LC = self.dLambda[l,:]
            LR = self.dLambda[l+1,:]
        
        Glc = self.dGamma[l,:]
        Gcr = self.dGamma[l+1,:]

        # charge of corresponding index (bond charge left/center/right)
        CL = self.dcharge[l,:]
        CC = self.dcharge[l+1,:]
        CR = self.dcharge[l+2,:]

        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.d, self.chi, CL, CC, CR)
        # Obtaining aligned charges
        cNewL, cNewR, incC = map(cp.array, [aligner.cNewL, aligner.cNewR, aligner.incC])
        # Obtaining aligned data
        LL, LR, Glc, Gcr = map(aligner.align_data, ['LL','LR','Glc','Gcr'], [LL,LR,Glc,Gcr])

        # Storage of generated data
        new_Gamma_L = []
        new_Gamma_R = []
        new_Lambda = cp.array([], dtype=float)
        new_charge = cp.array([], dtype=int)
        tau_array = [0]

        for tau in range(self.d):

            # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
            min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r = self.charge_range(location, tau)
            # Selecting data according to charge bounds
            cl, cc, cr, ll, lc, lr = map(aligner.select_data, [True]*6, ['cl','cc','cr','ll','lc','lr'], [cNewL,CC,cNewR,LL,LC,LR], [min_charge_l, min_charge_c, min_charge_r]*2, [max_charge_l, max_charge_c, max_charge_r]*2)
            glc = aligner.select_data(True, 'glc', Glc, min_charge_l, max_charge_l, min_charge_c, max_charge_c)
            gcr = aligner.select_data(True, 'gcr', Gcr, min_charge_c, max_charge_c, min_charge_r, max_charge_r)

            # Skip if any selection must be empty
            len_l, len_r = cl.shape[0], cr.shape[0]
            if len_l*len_r == 0:
                tau_array.append(0)
                continue

            # Computes Theta matrix
            #cpuT = loop(self, tau, U, cl, cc, cr, ll, glc, lc, gcr, lr)
            #T = cp.array(cpuT)
            T = update_MPS(self.d, tau, U, glc, gcr, ll, lc, lr, cl, cc, cr, incC)
            #print('cpuT: ', cpuT)
            # De-align (compact) T
            T = aligner.compact_data(True, 'T', T, min_charge_l, max_charge_l, min_charge_r, max_charge_r)
            
            # SVD
            V, Lambda, W = np.linalg.svd(T, full_matrices = False)
            T, V, W = map(cp.array, [T, V, W])

            # Store new results
            new_Gamma_L = new_Gamma_L + [V[:, i] for i in range(len(Lambda))]
            new_Gamma_R = new_Gamma_R + [W[i, :] for i in range(len(Lambda))]
            new_Lambda = cp.append(new_Lambda, Lambda)
            new_charge = cp.append(new_charge, cp.repeat(cp.array(tau, dtype=int), len(Lambda)))
            tau_array.append(len(Lambda))
        
        # Number of singular values to save
        num_lambda = int(min((new_Lambda > errtol).sum(), self.chi))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx_select = cp.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
        else:
            idx_select = cp.array([], dtype=int)
        
        # Initialize selected and sorted Gamma outputs
        Gamma1Out = cp.zeros([self.chi, self.chi], dtype = data_type)
        Gamma2Out = cp.zeros([self.chi, self.chi], dtype = data_type)

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        # Since division by Lambda, if Lambda is 0, output should be 0.
        LL[cp.where(LL == 0)[0]] = 999999999999
        LR[cp.where(LR == 0)[0]] = 999999999999
        
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for tau in range(self.d): # charge at center
            # Selecting gamma that will be modified. Modifying gamma will modify Gamma (because they are pointers).
            min_charge_l, max_charge_l, _, _, min_charge_r, max_charge_r = self.charge_range(location, tau)
            gamma1out = aligner.select_data(False, 'Glc', Gamma1Out, min_charge_l, max_charge_l, 0, self.d)
            gamma2out = aligner.select_data(False, 'Gcr', Gamma2Out, 0, self.d, min_charge_r, max_charge_r)

            # Finding bond indices (tau_idx) that are in the largest num_lambda singular values and for center charge tau.
            # idx_select[indices] = tau_idx
            tau_idx, indices, _ = np.intersect1d(cp.asnumpy(idx_select), np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

            if len(tau_idx) * gamma1out.shape[0] * gamma2out.shape[1] == 0:
                continue

            # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
            V = cp.array([new_Gamma_L[i] for i in tau_idx], dtype=data_type)
            W = cp.array([new_Gamma_R[i] for i in tau_idx], dtype=data_type)
            V = V.T

            # Calculating output gamma
            # Left
            ll = aligner.compact_data(False, 'LL', LL, min_charge_l, max_charge_l)[:, np.newaxis] # Selecting lambda
            gamma1out[:, indices] = V / ll
            # Right
            lr = aligner.compact_data(False, 'LR', LR, min_charge_r, max_charge_r)
            gamma2out[indices] = W / lr

        # Select charges that corresponds to the largest num_lambda singular values
        new_charge = new_charge[idx_select]
        # Sort the new charges
        idx_sort = cp.argsort(new_charge) # Indices that will sort the new charges
        new_charge = new_charge[idx_sort]
        # Update charges (modifying CC modifies self.dcharge by pointer)
        CC[:num_lambda] = new_charge
        CC[num_lambda:] = self.d # Charges beyond num_lambda are set to impossible values d
        
        # Selecting and sorting Lambda
        new_Lambda = new_Lambda[idx_select]
        new_Lambda = new_Lambda[idx_sort]
        print(np.sort(new_Lambda))
        LC[:num_lambda] = new_Lambda
        LC[num_lambda:] = 0

         # Sorting Gamma
        Gamma1Out[:, :num_lambda] = Gamma1Out[:, idx_sort]
        Gamma2Out[:num_lambda] = Gamma2Out[idx_sort]
        if location == left:
            self.dGamma[0, 0, :] = Gamma1Out[0, :]; self.dGamma[1, :, :] = Gamma2Out
        elif location == right:
            self.dGamma[self.n - 2, :, :] = Gamma1Out; self.dGamma[self.n - 1, :, 0] = Gamma2Out[:, 0]
        else:
            self.dGamma[l, :, :] = Gamma1Out; self.dGamma[l + 1, :, :] = Gamma2Out
        

    def RCS1DOneCycleUpdate(self, k):
        if k < self.n / 2:
            temp1 = 2 * k + 1
            temp2 = 2
            l = 2 * k
            while l >= 0:
                if temp1 > 0:
                    T = my_cv.rvs(2 * k + 2, temp1)
                    temp1 -= 2
                else:
                    T = my_cv.rvs(2 * k + 2, temp2)
                    temp2 += 2
                self.MPStwoqubitUpdate(l, np.sqrt(1 - T))
                l -= 1
        else:
            temp1 = 2 * self.n - (2 * k + 3)
            temp2 = 2
            l = n - 2
            for i in range(2 * self.n - 2 * k - 2):
                if temp1 >= 0:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp1)
                    temp1 -= 2
                else:
                    T = my_cv.rvs(2 * self.n - (2 * k + 1), temp2)
                    temp2 += 2
                self.MPStwoqubitUpdate(l, np.sqrt(1 - T))
                l -= 1;   
        
    def RCS1DMultiCycle(self):
        self.MPSInitialization()
        self.TotalProbPar[0] = self.TotalProbFromMPS()
        
        self.EEPar[:, 0] = self.MPSEntanglementEntropy()
        
        for k in range(self.n - 1):
            self.RCS1DOneCycleUpdate(k)
            self.TotalProbPar[k + 1] = self.TotalProbFromMPS()
            self.EEPar[:, k + 1] = self.MPSEntanglementEntropy()
    
        
        return self.TotalProbPar, self.EEPar#, self.REPar
    
    def TotalProbFromMPS(self):
        # Gamma (chi, chi, d, n) # Gamma1 first component is blank
        # Lambda3 (chi, n)
        
        sq_lambda = np.sum(self.dLambda ** 2, axis = 1)
        return np.average(sq_lambda)
    
    
    def MPSEntanglementEntropy(self):
        sq_lambda = np.copy(self.dLambda ** 2)
        Output = np.zeros([self.n - 1])

        for i in range(self.n - 1):
            Output[i] = EntropyFromColumn(ColumnSumToOne(sq_lambda[i, :]))

        return Output


def EntropyFromRow(InputRow):
    Output = 0

    for i in range(len(InputRow)):
        if InputRow[0,i] == 0:
            Output = Output
        else:
            Output = Output - InputRow[0,i] * np.log2(InputRow[0,i])

    return Output

def EntropyFromColumn(InputColumn):
    Output = np.nansum(-InputColumn * np.log2(InputColumn))
    
    return Output

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
    return Output

class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1)
my_cv = my_pdf(a = 0, b = 1, name='my_pdf')

if __name__ == "__main__":
    m_array = [6]#, 4, 6 ,8, 10, 12, 14, 16, 18, 20];
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
        print("m: {}, Time taken: {}.".format(m, time.time() - start))