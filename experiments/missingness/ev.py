import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time

import runs

from data.ev.get import get_dataset

ev_df = get_dataset()
ev_mat = ev_df.to_numpy()

# TODO: Get flat clusters

results = runs.perform_runs(
    ev_mat, 
    "1234",
    scipy.spatial.distance.pdist(ev_mat, "sqeuclidean"),
    scipy.spatial.distance.pdist(ev_mat, "euclidean"),
)

ts = str(int(time.time()))
fullres_df.to_csv(f"./results/runs-ev-{ts}.csv")
