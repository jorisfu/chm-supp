import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time

import runs
import scipy

from data.synthetic.get import get_dataset

synth_mat, true_fcluster, _ = get_dataset()

results = runs.perform_runs(
    synth_mat, 
    "1234",
    true_fcluster,
    50,
)

ts = str(int(time.time()))
results.to_csv(f"runs-synth-{ts}.csv")
results.to_csv(f"./results/runs-synth-{ts}.csv")
