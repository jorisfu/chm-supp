# Sanity checks for data import and missingness addition

import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import scipy
import pandas as pd
import numpy as np

import clusteredheatmap as chm
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

from data.synthetic.get import get_dataset

from experiments.missingness.commontools import add_nans_uniform_partial, add_nans_uniform_everywhere

mat, true_fcluster, col_gm = get_dataset()

rng = np.random.default_rng(2)
# missing_mat = add_nans_uniform_partial(mat, 0.5, 0.9, rng)
missing_mat = add_nans_uniform_everywhere(mat, 0.4, rng)

c = chm.chm.ClusteredHeatMap(
    pd.DataFrame(missing_mat),
    distance="sqeuclidean",
    linkage="complete",
    column_group_mappings=col_gm,
    use_completecase_analysis=True,
)

b = PlotlyVisuBuilder(c)
b.autobuild().show()
