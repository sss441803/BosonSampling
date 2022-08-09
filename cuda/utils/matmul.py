import numpy as np
import time

m = 4*1024
n = 2*1024
k = 4*1024

a = np.random.rand(m,k)
b = np.random.rand(k,n)

start = time.time()
for i in range(10):
    c = np.matmul(a,b)
stop = time.time()
print((stop - start)*100)