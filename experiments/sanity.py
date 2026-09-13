# Sanity checks for data import and missingness addition

import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import runs
import scipy
import clusteredheatmap as chm
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

from data.synthetic.get import get_dataset

synth_mat, true_fcluster = get_dataset()

c = chm.chm.ClusteredHeatMap(
    pd.DataFrame(synth_mat),
    distance="euclidean",
    linkage="complete"
)

b = PlotlyVisuBuilder(c)
c.autobuild().get_figure().show()
