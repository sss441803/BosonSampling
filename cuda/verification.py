import numpy as np
import cupy as cp
import time

C = np.load('C.npy')
A = np.load('A.npy')
B = np.load('B.npy')
LL = np.load('LL.npy')
LC = np.load('LC.npy')
LR = np.load('LR.npy')

A = np.array(A, dtype = np.float32)
B = np.array(B, dtype = np.float32)
LL = np.array(LL, dtype = np.float32)
LC = np.array(LC, dtype = np.float32)
LR = np.array(LR, dtype = np.float32)

start = time.time()
cpu_C = np.multiply(LL.reshape(-1,1),A)
cpu_C = np.multiply(LC, cpu_C)
cpu_C = np.matmul(cpu_C, B)
cpu_C = np.multiply(LR, cpu_C)
stop = time.time()
print('CPU time: ', (stop - start) * 1000)

A = cp.array(A, dtype = np.float32)
B = cp.array(B, dtype = np.float32)
LL = cp.array(LL, dtype = np.float32)
LC = cp.array(LC, dtype = np.float32)
LR = cp.array(LR, dtype = np.float32)

start = time.time()
gpu_C = cp.multiply(LL.reshape(-1,1),A)
gpu_C = cp.multiply(LC, gpu_C)
gpu_C = cp.matmul(gpu_C, B)
gpu_C = cp.multiply(LR, gpu_C)
print(gpu_C[0][0].item())
stop = time.time()
print('Cupy time: ', (stop - start) * 1000)

start = time.time()
gpu_C = cp.einsum('a,ab,b,bc,c->ac', LL, A, LC, B, LR)
print(gpu_C[0][0].item())
stop = time.time()
print('Cupy einsum time: ', (stop - start) * 1000)

print(np.allclose(C, cpu_C, rtol=1e-05, atol=1e-08, equal_nan=False))
print(np.max(np.abs(C- cpu_C)))
print('GPU results: ', C)
print('Python results: ', cpu_C)