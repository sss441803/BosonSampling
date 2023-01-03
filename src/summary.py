import numpy as np

def EntanglementEntropy(Lambda):
    n_modes = Lambda.shape[0] + 1
    Output = np.zeros([n_modes - 1])
    sq_lambda = np.copy(Lambda ** 2)
    for i in range(n_modes - 1):
        Output[i] += EntropyFromColumn(ColumnSumToOne(sq_lambda[i]))
    return Output

def Probability(n, d, Gamma, Lambda, charge):
    R = Gamma[n - 1, :, 0]
    RTemp = np.copy(R)
    for k in range(n - 2):
        idx = np.array([], dtype = 'int32')
        for ch in range(d):
            idx = np.append(idx, np.intersect1d(np.nonzero(charge[n - 1 - k, :, 0] == ch), np.intersect1d(np.nonzero(charge[n - 1 - k, :, 1] == ch), np.nonzero(Lambda[n - 1 - k - 1] > 0))))
        R = np.matmul(Gamma[n - 1 - k - 1, :, idx].T, RTemp[idx].reshape(-1))
        RTemp = np.copy(R)
    idx = np.array([], dtype = 'int32')
    for ch in range(d):
        idx = np.append(idx, np.intersect1d(np.nonzero(charge[1, :, 0] == ch), np.intersect1d(np.nonzero(charge[1, :, 1] == ch), np.nonzero(Lambda[0, :] > 0))))
    res = np.matmul(Gamma[0, :, idx].T, RTemp[idx].reshape(-1))
    tot_prob = np.sum(res)
    print('Probability: ', np.real(tot_prob))
    return tot_prob

def EntropyFromColumn(InputColumn):
    Output = -np.nansum(InputColumn * np.log2(InputColumn))
    return Output

def RenyiFromColumn(InputColumn, alpha):
    Output = np.log2(np.nansum(InputColumn ** alpha)) / (1 - alpha)
    return Output

def ColumnSumToOne(InputColumn):
    return InputColumn / np.sum(InputColumn)