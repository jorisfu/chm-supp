from commontools import add_nans_uniform_everywhere, add_nans_uniform_partial, coph_corr, add_nans_uniform_specific_samples, add_nans_uniform_only_lowerhalf

import numpy as np
import pandas as pd

def perform_runs(
    data, # Dataset, columns are clustered
    scenarios,
    truedist_cols_sq, # Condensted sqeuclidean distmat
    truedist_cols, # Condensed euclidean distmat
    true_flatclusters, # List of integers (encoded cluster assignments for col indices)
):

    ##
    ## CONFIG
    ##

    LINKAGE = "complete"
    MISSINGNISS_RATIOS = [i/100 for i in [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]]
    N_REPLICATES = 50

    # (fnname, distance_args, reference_distmat)
    DISTANCES = [
        ("sqeuclidean", truedist_cols_sq, {}),
        ("dixon_pds_sqeuclidean", truedist_cols_sq, {}),
        ("nandist_euclidean", truedist_cols, {}),
        ("eirola_esd_mvn", truedist_cols_sq, {"max_iter": 200}),
        ("eirola_esd_gmm", truedist_cols_sq, {}),
        ("mesquita_eed", truedist_cols, {}),
    ]
    DISTNAMES = [d[0] for d in DISTANCES]

    ##
    ## Single run
    ##

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

    ##
    ## Scenarios
    ##

    rng = np.random.default_rng(123)
    run_results = {}

    # Scenario 1: All observations
    if "1" in scenarios:
        print(">>> SCENARIO 1")
        missfun = lambda data, p: add_nans_uniform_everywhere(data, p, rng)
        run_results["uniform_everywhere"] = run(data, missfun)

    # Scenario 2: Random part of observations (30%)
    if "2" in scenarios:
        print(">>> SCENARIO 2")
        missfun = lambda data, p: add_nans_uniform_partial(data, p, 0.3, rng)
        run_results["uniform_randomrows"] = run(data, missfun)

    # Scenario 3: Only to 30% of vectors with lowest sum of intensities
    if "3" in scenarios:
        print(">>> SCENARIO 3")
        m = int(0.3 * DATA_SHAPE[0]) # amount of vectors with lowest sum of intensities affected
        sums = d.sum(axis=1)
        indices = np.argpartition(sums, m-1)[:m]
        missfun = lambda data, p: add_nans_uniform_specific_samples(d, indices, p, rng)
        run_results["uniform_lowestrowsumsonly"] = run(data, missfun)

    # Scenario 4: Only to lower half of scalars
    if "4" in scenarios:
        print(">>> SCENARIO 4")
        rng = np.random.default_rng(123)
        missfun = lambda data, p: add_nans_uniform_only_lowerhalf(data, p, rng)
        run_results["uniform_halflowestscalars"] = run(data, missfun)


    ##
    ## ANALYSIS
    ##

    n_flatclusters = len(set(true_flatclusters))

    fullres_df = pd.DataFrame(columns=["Distance", "Run", "Missingness", "Replicate", "CCC", "Rand", "aRand"])

    for run, result in run_results.items():
        for idx, (dist, comp, _args) in enumerate(DISTANCES):
            res_for_distance = result[dist]
            reference_distmat = comp

            for missingness in res_for_distance.keys():
                for jdx, replicate in enumerate(result[dist][missingness]):
                    # Cophenetic Correlation Coefficient
                    ccc_cols = coph_corr(replicate.linkage_matrix_cols, reference_distmat)

                    flatclusters = fcluster(replicate.linkage_matrix_cols, t=n_flatclusters, criterion="maxclust") 

                    # (a)Rand (2 clusters)
                    # TODO BEFORE RUN: THIS IS FISHY!!!!
                    rand_cols = rand_score(true_flatclusters, flatclusters)
                    arand_cols = adjusted_rand_score(true_flatclusters, flatclusters)

                    fullres_df.loc[-1] = [dist, run, missingness, jdx, ccc_cols, rand_cols, arand_cols]
                    fullres_df.index = fullres_df.index + 1
                    fullres_df = fullres_df.sort_index()

    return fullres_df
