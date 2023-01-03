import numpy as np
from scipy.stats import rv_continuous

class my_pdf(rv_continuous):
    def _pdf(self, x, k, idx):
        return (k - idx) * (1 - x) ** (k - idx - 1)
my_cv = my_pdf(a = 0, b = 1, name='my_pdf')

def ReflectivityAndSeeds(n_modes):
    np.random.seed(1)
    reflectivity = np.empty([n_modes, n_modes // 2])
    seeds = np.empty([n_modes, n_modes // 2], dtype=int)
    for k in range(n_modes - 1):
        # print('k, ', k)
        if k < n_modes / 2:
            temp1 = 2 * k + 1
            temp2 = 2
            l = 2 * k
            i = 0
            while l >= 0:
                if temp1 > 0:
                    T = my_cv.rvs(2 * k + 2, temp1)
                    temp1 -= 2
                else:
                    T = my_cv.rvs(2 * k + 2, temp2)
                    temp2 += 2
                reflectivity[i, k-(i+1)//2] = np.sqrt(1 - T)
                seed = np.random.randint(0, 13579)
                seeds[i, k-(i+1)//2] = seed
                # np.random.seed(seed)
                # print(seed)
                l -= 1
                i += 1
        else:
            temp1 = 2 * n_modes - (2 * k + 3)
            temp2 = 2
            l = n_modes - 2
            first_layer = 2 * k - n_modes + 2
            for i in range(2 * n_modes - 2 * k - 2):
                if temp1 >= 0:
                    T = my_cv.rvs(2 * n_modes - (2 * k + 1), temp1)
                    temp1 -= 2
                else:
                    T = my_cv.rvs(2 * n_modes - (2 * k + 1), temp2)
                    temp2 += 2
                reflectivity[first_layer + i, n_modes//2-1-(i+1)//2] = np.sqrt(1 - T)
                seed = np.random.randint(0, 13579)
                seeds[first_layer + i, n_modes//2-1-(i+1)//2] = seed
                # np.random.seed(seed)
                # print(seed)
                l -= 1  

    return reflectivity, seeds