import cupy as cp
import time

a = cp.random.rand(10000,5000)
idx = cp.arange(0, 4999, dtype=int)
idxs = []
answer = []
reordered = []
for i in range(100):
    cp.random.shuffle(idx)
    idxs.append(cp.copy(idx))
for i in range(100):
    answer.append(a[idxs[i][2003], 4821].item())
start = time.time()
for i in range(100):
    #print(idxs[i])
    b = cp.ascontiguousarray(a[idxs[i]])
    reordered.append(b[2003,4821].item())
print(time.time() - start)
print(cp.allclose(answer, reordered))