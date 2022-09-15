'''
Full simulation code containing the Device method (cupy, unified update)
To run:
nvidia-cuda-mps-control -d
ps -ef | grep mps
python -u MPS_BS_cupy.py > results1.out & python -u MPS_BS_cupy.py > results2.out & python -u MPS_BS_cupy.py > results3.out & python -u MPS_BS_cupy.py > results4.out & python -u MPS_BS_cupy.py > results5.out & python -u MPS_BS_cupy.py > results6.out & python -u MPS_BS_cupy.py > results7.out & python -u MPS_BS_cupy.py > results8.out
echo quit | nvidia-cuda-mps-control
'''

import numpy as np
import cupy as cp

from scipy.stats import rv_continuous
from scipy.special import factorial, comb

import time

from cuda_kernels import update_MPS, Rand_U
from sort import Aligner

#np.random.seed(1)
np.set_printoptions(precision=3)
s = cp.cuda.Stream()


def Rand_BS_MPS(d, r):
    t = np.sqrt(1 - r ** 2) * np.exp(1j * np.random.rand() * 2 * np.pi);
    r = r * np.exp(1j * np.random.rand() * 2 * np.pi);
    ct = np.conj(t); cr = np.conj(r);
    bs_coeff = lambda n, m, k, l: np.sqrt(factorial(l) * factorial(n + m - l) / factorial(n) / factorial(m)) * comb(n, k) * comb(m, l - k) * (t ** k) * (ct ** (m - l + k)) * (r ** (n - k)) * ((-cr) ** (l - k))
    BS = np.zeros([d, d, d], dtype = 'complex64');
    for n in range(d): #photon number from 0 to d-1
        for m in range(d):
            for l in range(max(0, n + m + 1 - d), min(d, n + m + 1)): #photon number on first output mode
                k = np.arange(max(0, l - m), min(l + 1, n + 1, d))
                BS[n, m, l] = np.sum(bs_coeff(n, m, k, l))
                
    Output = BS
    
    return cp.array(Output)


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


data_type = np.complex64
float_type = np.float32
int_type = np.int32


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
        self.REPar = np.zeros([n - 1, n, 1], dtype=np.float32)
        self.update_time = 0
        self.U_time = 0
        self.svd_time = 0
        self.theta_time = 0
        self.align_init_time = 0
        self.align_info_time = 0
        self.index_time = 0
        self.copy_time = 0
        self.align_time = 0
        self.other_time = 0
        self.largest_C = 0
        self.largest_T = 0
        
    def MPSInitialization(self):
        self.Gamma = np.zeros([self.n, self.chi, self.chi], dtype=data_type); # modes, alpha, alpha
        self.Lambda = np.zeros([self.n - 1, self.chi], dtype=float_type); # modes - 1, alpha
        self.Lambda_edge = np.ones(self.chi, dtype=float_type) # edge lambda (for first and last site) don't exists and are ones
        self.charge = self.d * np.ones([self.n + 1, self.chi], dtype=int_type) # Initialize initial charges for all bonds to the impossible charge d.
        
        # Initialize the first bond
        # At the beginning, only the first bond is initialized to non-trivial information
        for i in range(self.n):
            self.Gamma[i, 0, 0] = 1 
        
        for i in range(self.n - 1):
            self.Lambda[i, 0] = 1
        
        self.charge[:, 0] = 0 # First bond has charge zero
        for i in range(self.m):
            self.charge[i, 0] = self.m - i # Since each mode has a photon for the first m modes, the charge number (photons to the right) increases by one per site

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
        
        # Initializing unitary matrix on GPU
        np.random.seed(seed)
        start = time.time()
        d_U_r, d_U_i = Rand_U(self.d, r)
        s.synchronize()
        #print(U_r[0,0,0])
        self.U_time += time.time() - start

        # Determining the location of the two qubit gate
        left = "Left"
        center = "Center"
        right = "Right"
        if l == 0:
            location = left
            LL = self.Lambda_edge[:]
            LC = self.Lambda[l,:]
            LR = self.Lambda[l+1,:]
        elif l == self.n - 2:
            location = right
            LL = self.Lambda[l-1,:]
            LC = self.Lambda[l,:]
            LR = self.Lambda_edge[:]
        else:
            location = center
            LL = self.Lambda[l-1,:]
            LC = self.Lambda[l,:]
            LR = self.Lambda[l+1,:]
        
        Glc = self.Gamma[l,:]
        Gcr = self.Gamma[l+1,:]

        # charge of corresponding index (bond charge left/center/right)
        CL = self.charge[l,:]
        CC = self.charge[l+1,:]
        CR = self.charge[l+2,:]

        start = time.time()
        # Creating aligner according to left and right charges. Will be used for algning, de-aligning (compacting), selecting data, etc.
        aligner = Aligner(self.d, self.chi, CL, CC, CR)
        self.align_info_time += aligner.align_info_time
        self.index_time += aligner.index_time
        self.align_init_time += time.time() - start
        start = time.time()
        # Obtaining aligned charges
        d_cNewL, d_cNewR, d_incC = map(cp.array, [aligner.cNewL, aligner.cNewR, aligner.incC])
        self.copy_time = time.time() - start
        start = time.time()
        # Obtaining aligned data
        LL, LR, Glc, Gcr = map(aligner.align_data, ['LL','LR','Glc','Gcr'], [LL,LR,Glc,Gcr])
        d_CC, d_LL, d_LC, d_LR, d_Glc, d_Gcr = map(cp.array, [CC, LL, LC, LR, Glc, Gcr])
        self.align_time += time.time() - start

        # Storage of generated data
        d_new_Gamma_L = []
        d_new_Gamma_R = []
        new_Lambda = np.array([], dtype=float_type)
        new_charge = np.array([], dtype=int_type)
        tau_array = [0]

        for tau in range(self.d):

            start = time.time()

            # Bounds for data selection. Given tau (center charge), find the range of possible charges for left, center and right.
            min_charge_l, max_charge_l, min_charge_c, max_charge_c, min_charge_r, max_charge_r = self.charge_range(location, tau)
            # Selecting data according to charge bounds
            d_cl, d_cc, d_cr, d_ll, d_lc, d_lr = map(aligner.select_data, [True]*6, ['cl','cc','cr','ll','lc','lr'], [d_cNewL,d_CC,d_cNewR,d_LL,d_LC,d_LR], [min_charge_l, min_charge_c, min_charge_r]*2, [max_charge_l, max_charge_c, max_charge_r]*2)
            d_glc = aligner.select_data(True, 'glc', d_Glc, min_charge_l, max_charge_l, min_charge_c, max_charge_c)
            d_gcr = aligner.select_data(True, 'gcr', d_Gcr, min_charge_c, max_charge_c, min_charge_r, max_charge_r)

            # Skip if any selection must be empty
            len_l, len_r = d_cl.shape[0], d_cr.shape[0]
            self.largest_C = max(max(len_l, len_r), self.largest_C)
            if len_l*len_r == 0:
                tau_array.append(0)
                continue

            self.align_time += time.time() - start

            start = time.time()
            #print('U_r: {}, U_i: {}, glc: {}, gcr: {}, ll: {}, lc: {}, lr: {}, cl: {}, cc: {}, cr: {}, incC: {}.'.format(*map(type, [d_U_r, d_U_i, d_glc, d_gcr, d_ll, d_lc, d_lr, d_cl, d_cc, d_cr, d_incC])))
            d_T = update_MPS(self.d, tau, d_U_r, d_U_i, d_glc, d_gcr, d_ll, d_lc, d_lr, d_cl, d_cc, d_cr, d_incC)
            d_T = aligner.compact_data(True, 'T', d_T, min_charge_l, max_charge_l, min_charge_r, max_charge_r)
            s.synchronize()
            dt = time.time() - start
            self.largest_T = max(dt, self.largest_T)
            self.theta_time += dt
            
            # SVD
            start = time.time()
            d_V, d_Lambda, d_W = cp.linalg.svd(d_T, full_matrices = False)
            Lambda = cp.asnumpy(d_Lambda)
            s.synchronize()
            #V, W = map(cp.array, [V, W])
            self.svd_time += time.time() - start

            # Store new results
            d_new_Gamma_L = d_new_Gamma_L + [d_V[:, i] for i in range(len(Lambda))]
            d_new_Gamma_R = d_new_Gamma_R + [d_W[i, :] for i in range(len(Lambda))]
            new_Lambda = np.append(new_Lambda, Lambda)
            new_charge = np.append(new_charge, np.repeat(np.array(tau, dtype=int_type), len(Lambda)))
            tau_array.append(len(Lambda))
        
        start = time.time()

        # Number of singular values to save
        num_lambda = int(min((new_Lambda > errtol).sum(), self.chi))
        # cupy behavior differs from numpy, the case of 0 length cupy array must be separately taken care of
        if num_lambda!= 0:
            idx_select = np.argpartition(new_Lambda, -num_lambda)[-num_lambda:] # Indices of the largest num_lambda singular values
        else:
            idx_select = np.array([], dtype=int_type)
        
        # Initialize selected and sorted Gamma outputs
        Gamma1Out = np.zeros([self.chi, self.chi], dtype = data_type)
        Gamma2Out = np.zeros([self.chi, self.chi], dtype = data_type)

        # Indices of eigenvalues that mark the beginning of center charge tau
        cum_tau_array = np.cumsum(tau_array)
        # Since division by Lambda, if Lambda is 0, output should be 0.
        LL[np.where(LL == 0)[0]] = 999999999999
        LR[np.where(LR == 0)[0]] = 999999999999
        
        # Need to loop through center charges to select (bonds corresponds to the largest singular values) saved Gammas to output gammas
        for tau in range(self.d): # charge at center
            # Selecting gamma that will be modified. Modifying gamma will modify Gamma (because they are pointers).
            min_charge_l, max_charge_l, _, _, min_charge_r, max_charge_r = self.charge_range(location, tau)
            gamma1out = aligner.select_data(False, 'Glc', Gamma1Out, min_charge_l, max_charge_l, 0, self.d)
            gamma2out = aligner.select_data(False, 'Gcr', Gamma2Out, 0, self.d, min_charge_r, max_charge_r)

            # Finding bond indices (tau_idx) that are in the largest num_lambda singular values and for center charge tau.
            # idx_select[indices] = tau_idx
            tau_idx, indices, _ = np.intersect1d(idx_select, np.arange(cum_tau_array[tau], cum_tau_array[tau + 1]), return_indices = True)

            if len(tau_idx) * gamma1out.shape[0] * gamma2out.shape[1] == 0:
                continue

            # Left and right singular vectors that corresponds to the largest num_lambda singular values and center charge tau
            d_V = cp.array([d_new_Gamma_L[i] for i in tau_idx], dtype=data_type)
            d_W = cp.array([d_new_Gamma_R[i] for i in tau_idx], dtype=data_type)
            d_V = d_V.T
            V, W = map(cp.asnumpy, [d_V, d_W])

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
        idx_sort = np.argsort(new_charge) # Indices that will sort the new charges
        new_charge = new_charge[idx_sort]
        # Update charges (modifying CC modifies self.dcharge by pointer)
        CC[:num_lambda] = new_charge
        CC[num_lambda:] = self.d # Charges beyond num_lambda are set to impossible values d
        
        # Selecting and sorting Lambda
        new_Lambda = new_Lambda[idx_select]
        new_Lambda = new_Lambda[idx_sort]

        if new_Lambda.shape[0] == 0:
            print(0)
        else:
            print(np.max(new_Lambda))

        LC[:num_lambda] = new_Lambda
        LC[num_lambda:] = 0

         # Sorting Gamma
        Gamma1Out[:, :num_lambda] = Gamma1Out[:, idx_sort]
        Gamma2Out[:num_lambda] = Gamma2Out[idx_sort]
        #print('Gamma: ', Gamma1Out, Gamma2Out)
        if location == left:
            self.Gamma[0, 0, :] = Gamma1Out[0, :]; self.Gamma[1, :, :] = Gamma2Out
        elif location == right:
            self.Gamma[self.n - 2, :, :] = Gamma1Out; self.Gamma[self.n - 1, :, 0] = Gamma2Out[:, 0]
        else:
            self.Gamma[l, :, :] = Gamma1Out; self.Gamma[l + 1, :, :] = Gamma2Out
        
        #print(self.dGamma[0,0,0])
        self.other_time += time.time() - start


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
                start = time.time()
                self.MPStwoqubitUpdate(l, np.sqrt(1 - T))
                self.update_time += time.time() - start
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
                start = time.time()
                self.MPStwoqubitUpdate(l, np.sqrt(1 - T))
                self.update_time += time.time() - start
                l -= 1;   
        print('One cycle')
        
    def RCS1DMultiCycle(self):
        self.MPSInitialization()
        #self.TotalProbPar[0] = self.TotalProbFromMPS()
        
        #self.EEPar[:, 0] = self.MPSEntanglementEntropy()
        
        for k in range(self.n - 1):
            self.RCS1DOneCycleUpdate(k)
            #self.TotalProbPar[k + 1] = self.TotalProbFromMPS()
            #self.EEPar[:, k + 1] = self.MPSEntanglementEntropy()
    
        
        return self.TotalProbPar, self.EEPar#, self.REPar
    
    def TotalProbFromMPS(self):
        # Gamma (chi, chi, d, n) # Gamma1 first component is blank
        # Lambda3 (chi, n)
        
        sq_lambda = np.sum(self.Lambda ** 2, axis = 1)
        return np.average(sq_lambda)
    
    
    def MPSEntanglementEntropy(self):
        sq_lambda = np.copy(self.Lambda ** 2)
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
    m_array = [12]#, 4, 6 ,8, 10, 12, 14, 16, 18, 20];
    EE_tot = []
    for m in m_array:
        start = time.time()
        EE_temp = np.zeros((31, 32))
        for i in range(1):
            NumSample = 1; n = 32; chi = 2**m; errtol = 10 ** (-5)
            boson = MPS(n, m, chi, errtol)
            Totprob, EE = boson.RCS1DMultiCycle()
            EE_temp += EE
        EE_tot.append(EE_temp / 100)
        print("m: {:.2f}. Total time: {:.2f}. Update time: {:.2f}. U time: {:.2f}. Theta time: {:.2f}. SVD time: {:.2f}. Align init time: {:.2f}. Align info time: {:.2f}. Index time: {:.2f}. Copy time: {:.2f}. Align time: {:.2f}. Other_time: {:.2f}. Largest array dimension: {:.2f}. Longest time for single matrix: {:.8f}".format(m, time.time()-start, boson.update_time, boson.U_time, boson.theta_time, boson.svd_time, boson.align_init_time, boson.align_info_time, boson.index_time, boson.copy_time, boson.align_time, boson.other_time, boson.largest_C, boson.largest_T))