import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.ev.get import get_dataset, get_groupmap

from clusteredheatmap.chm import ClusteredHeatMap
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

df = tuple(get_dataset())[0]
sample_gm = get_groupmap()

dist = sys.argv[1]

c = ClusteredHeatMap(
    df, 
    distance=dist,
    use_completecase_analysis=True,
    linkage="complete", 
    column_group_mappings=sample_gm,
)

b = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh")
b.add_heatmap()
b.add_col_dendrogram()
b.add_row_dendrogram()
b.add_col_group_markers()
fig = b.get_figure()
fig.show()
