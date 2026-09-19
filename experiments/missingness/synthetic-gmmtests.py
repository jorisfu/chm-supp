import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from commontools import add_nans_uniform_everywhere

import time

import runs
import scipy

from data.synthetic.get import get_dataset

synth_mat, true_fcluster, _ = get_dataset()

MISSINGNISS_RATIOS = [i/100 for i in [10, 15, 20, 25, 30, 35, 40, 45, 50]]
N_REPLICATES = 50


print("Missingness,Replicate,n_components")
for p in MISSINGNISS_RATIOS:
    for replicate in range(N_REPLICATES):
        d_miss = add_nans_uniform_everywhere(synth_mat)
        d_miss_cols = d_miss.transpose()
        gmm_cols = get_best_gmm(1, 4, 200, "BIC", d_miss_cols)
        print(str(p) + "," + str(replicate) + "," + str(gmm_cols.n_components), flush=True)
