import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

from clusteredheatmap.chm import ClusteredHeatMap
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

rng = np.random.default_rng(1)
mat = np.full((100, 100), 0.0)

red_range_x = list(range(0, 85))
red_range_y = list(range(0, 85))

mat[np.ix_(red_range_y, red_range_x)] += np.full((85, 85), 1.0)

ylw_range_x = list(range(25, 100))
ylw_range_y = list(range(25, 100))

mat[np.ix_(ylw_range_y, ylw_range_x)] += np.full((75, 75), 3.0)

noise = rng.normal(size=(100, 100), scale=0.3)
mat += noise

rng.shuffle(mat)
mat = mat.transpose()
rng.shuffle(mat)
mat = mat.transpose()

df = pd.DataFrame(mat)
c = ClusteredHeatMap(
    df,
    distance="euclidean",
    optimal_leaf_ordering=False,
    linkage="single"
)

b = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh")
b.add_heatmap(colorscale=[[0.0, "#FFFFFF"], [0.25, "#b30137"], [0.5, "#fce300"], [0.75, "#f7ab00"], [1, "#df6201"]])
b.add_col_dendrogram()
b.add_row_dendrogram()
fig = b.get_figure()
fig.update_layout(
    autosize=True,
    width=600,
    height=600,
    title="HPI"
)
fig.show()