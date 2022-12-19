import os
import pickle
from math import sqrt, sinh
import numpy as np

if not os.path.exists("experiment.pickle"):
    experiments = []
    ds = []
    ns = []
    for m in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50]:
        for beta in [0.6, 0.8, 1.0, 1.2]:
            for r in [0.48, 0.662, 0.88, 1.146, 1.44]:
                n = 4 * m
                ideal_ave_photons = m*sinh(r)**2
                lossy_ave_photons = beta*sqrt(ideal_ave_photons)
                loss = round(1000*(1 - lossy_ave_photons/ideal_ave_photons))/1000
                PS = int((1-loss)*m*sinh(r)**2)
                PS += 1
                d = PS + 1
                init_chi = d**2
                chi = int(max(32*2**PS, d**2, 512))
                if  not (d > 10 or chi > 10000 or PS == 0):
                    experiments.append({'n': n, 'm': m, 'beta': beta, 'r': r, 'PS': PS, 'd': d, 'status': 'incomplete'})
                    ds.append(d)
                    ns.append(n)
                    if PS > 1:
                        experiments.append({'n': n, 'm': m, 'beta': beta, 'r': r, 'PS': PS - 1, 'd': d - 1, 'status': 'incomplete'})
                        ds.append(d - 1)
                        ns.append(n)
    ds = np.array(ds)
    ns = np.array(ns)
    idx = np.lexsort([-ns, -ds])
    experiments = np.array(experiments)[idx]
    with open('experiment.pickle', 'wb') as file:
        pickle.dump(experiments, file)
    print('New experiment tracking file created.')
    print(experiments)