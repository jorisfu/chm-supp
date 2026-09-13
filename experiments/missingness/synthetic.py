# TODO: Clean this up


import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import cophenet
from scipy import stats

from clusteredheatmap.chm import ClusteredHeatMap
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder
from clusteredheatmap.types import Vector
from clusteredheatmap.algos import distance, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from clusteredheatmap.algos.modelselection import get_best_gmm

from commontools import add_nans_uniform_everywhere, add_nans_uniform_partial, coph_corr, add_nans_uniform_specific_samples, add_nans_uniform_only_lowerhalf

import time

import sys

## MVN
DATA_SHAPE = (80, 80)
RNG = np.random.default_rng(5)

MEAN_UP = 1
MEAN_DOWN = -1
VAR_UP = 0.5
VAR_DOWN = 0.5
VAR_TOTAL = 4

# We sample the means from a 1D GMM with 2 components because why not.
mean_down = RNG.normal(MEAN_DOWN, VAR_DOWN, size=DATA_SHAPE[0] // 2)
mean_up = RNG.normal(MEAN_UP, VAR_UP, size=DATA_SHAPE[0] // 2)
mean_du = np.concatenate([mean_down, mean_up])
mean_ud = np.concatenate([mean_up, mean_down])

cov = np.eye(DATA_SHAPE[0], DATA_SHAPE[1]) * VAR_TOTAL
d_du = RNG.multivariate_normal(mean_du, cov, DATA_SHAPE[0] // 2)
d_ud = RNG.multivariate_normal(mean_ud, cov, DATA_SHAPE[0] // 2)
d = np.concatenate([d_du, d_ud])

df = pd.DataFrame(d)
col_gm = {"Column": {str(i): "C_DU" for i in range(DATA_SHAPE[0]//2)} | {str(i): "C_UD" for i in range(DATA_SHAPE[0]//2, DATA_SHAPE[0])}}
row_gm = {"Row": {str(i): "R_UD" for i in range(DATA_SHAPE[1]//2)} | {str(i): "R_DU" for i in range(DATA_SHAPE[0]//2, DATA_SHAPE[1])}}

# Required for testing against distance estimated with missingness
truedist_cols_sq = scipy.spatial.distance.pdist(d.transpose(), "sqeuclidean")
truedist_cols = scipy.spatial.distance.pdist(d.transpose(), "euclidean")


LINKAGE = "complete"
MISSINGNISS_RATIOS = [i/100 for i in [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]]
N_REPLICATES = 50

# (fnname, distance_args, reference_distmat)
DISTANCES = [
    ("sqeuclidean", truedist_cols_sq, {}),
    ("dixon_pds_sqeuclidean", truedist_cols_sq, {}),
    ("nandist_euclidean", truedist_cols_sq, {}),
    ("eirola_esd_mvn", truedist_cols_sq, {"max_iter": 200}),
    ("eirola_esd_gmm", truedist_cols_sq, {}),
    ("mesquita_eed", truedist_cols_sq, {}),
]
DISTNAMES = [d[0] for d in DISTANCES]

run_results = {}

def run(data, missfun):
    results = {d: {int(i*100): [] for i in MISSINGNISS_RATIOS} for d in DISTNAMES}
    for p in MISSINGNISS_RATIOS:
        
        # Only 1 replicate for no missingness
        n_replicates = N_REPLICATES
        if p == 0.0:
            n_replicates = 1


        for replicate in range(n_replicates):
            while True:
                try:
                    d_miss = missfun(data, p)

                    # Pre-fit for ESD/GMM and EED
                    gmm = get_best_gmm(1, 4, 200, "BIC", d_miss)

                    for dist, ref, args in DISTANCES:
                        print(dist, p, replicate)

                        distance_args = args
                        if dist in ["eirola_esd_gmm", "mesquita_eed"]:
                            distance_args |= {"gmm": gmm}
                        
                        c = ClusteredHeatMap(
                            pd.DataFrame(d_miss),
                            distance=dist,
                            linkage=LINKAGE,
                            use_completecase_analysis=True,
                            column_group_mappings=col_gm,
                            row_group_mappings=row_gm,
                            cluster_rows=False,
                            distance_args=distance_args,
                        )

                        results[dist][int(p*100)].append(c)

                    break
                # Failure should only be due to CC criterion, draw new replicate if so
                except Exception as e:
                    print(e)
                    continue

    return results


# Prerequisite for aRand
gmmap = {
    "C_DU": 1,
    "C_UD": 2,
}

mapping = col_gm["Column"]
indices = [None for _ in range(DATA_SHAPE[0])]
for idx, val in mapping.items():
    indices[int(idx)] = gmmap[val]
true_major_clusters_cols = indices


rng = np.random.default_rng(123)

input_sc = sys.argv[1]

if "1" in input_sc:
    print(">>> SCENARIO 1")
    # Scenario 1: All observations
    missfun = lambda data, p: add_nans_uniform_everywhere(data, p, rng)
    run_results["uniform_everywhere"] = run(d, missfun)

if "2" in input_sc:
    print(">>> SCENARIO 2")
    # Scenario 2: Random part of observations (30%)
    missfun = lambda data, p: add_nans_uniform_partial(data, p, 0.3, rng)
    run_results["uniform_randomrows"] = run(d, missfun)

if "3" in input_sc:
    print(">>> SCENARIO 3")
    # Scenario 3: Only to 30% of vectors with lowest sum of intensities
    m = int(0.3 * DATA_SHAPE[0]) # amount of vectors with lowest sum of intensities affected
    sums = d.sum(axis=1)
    indices = np.argpartition(sums, m-1)[:m]
    missfun = lambda data, p: add_nans_uniform_specific_samples(d, indices, p, rng)
    run_results["uniform_lowestrowsumsonly"] = run(d, missfun)

if "4" in input_sc:
    print(">>> SCENARIO 4")
    # Scenario 4: Only to lower half of scalars
    rng = np.random.default_rng(123)
    missfun = lambda data, p: add_nans_uniform_only_lowerhalf(data, p, rng)
    run_results["uniform_halflowestscalars"] = run(d, missfun)



fullres_df = pd.DataFrame(columns=["Distance", "Run", "Missingness", "Replicate", "CCC", "Rand", "aRand"])

for run, result in run_results.items():
    for idx, (dist, comp, _args) in enumerate(DISTANCES):
        res_for_distance = result[dist]
        reference_distmat = comp

        for missingness in res_for_distance.keys():
            for jdx, replicate in enumerate(result[dist][missingness]):
                # Cophenetic Correlation Coefficient
                ccc_cols = coph_corr(replicate.linkage_matrix_cols, reference_distmat)

                flatclusters = fcluster(replicate.linkage_matrix_cols, t=2, criterion="maxclust") 

                # (a)Rand (2 clusters)
                rand_cols = rand_score(true_major_clusters_cols, flatclusters)
                arand_cols = adjusted_rand_score(true_major_clusters_cols, flatclusters)

                fullres_df.loc[-1] = [dist, run, missingness, jdx, ccc_cols, rand_cols, arand_cols]
                fullres_df.index = fullres_df.index + 1
                fullres_df = fullres_df.sort_index()

ts = str(int(time.time()))
fullres_df.to_csv(f"./results-syntheticdata-sc{str(input_sc)}-{ts}.csv")
