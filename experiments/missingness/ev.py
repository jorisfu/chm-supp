import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time

import runs
import scipy

from data.ev.get import get_dataset

ev_df, true_fcluster = get_dataset()
ev_mat = ev_df.to_numpy()

results = runs.perform_runs(
    ev_mat, 
    "1234",
    true_fcluster,
    5,
)

ts = str(int(time.time()))
results.to_csv(f"./results/runs-ev-{ts}.csv")
