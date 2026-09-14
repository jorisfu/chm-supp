import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.aml.get import get_dataset, get_groupmap

from clusteredheatmap.chm import ClusteredHeatMap
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

df = get_dataset()
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
b.add_heatmap(
    _zmin=-3.5,
    _zmid=0.0,
    _zmax=2.5,
    colorscale=[[0.0, "#0000FF"], [0.5, "#FFFFFF"], [1.0, "#FF0000"]],
    nan_color="#000000"
)
b.add_col_dendrogram()
b.add_row_dendrogram()
b.add_col_group_markers(_color_overrides={"Group": {"REL-FREE": "#09c901", "RELAPSE": "#760696"}})
fig = b.get_figure()
fig.update_layout(
    autosize=True,
    width=600,
    height=800,
    title="AML"
)
fig.show()
