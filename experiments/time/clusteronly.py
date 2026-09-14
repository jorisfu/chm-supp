import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.aml.get import get_dataset

from clusteredheatmap.chm import ClusteredHeatMap

df = get_dataset()

dist = sys.argv[1]

c = ClusteredHeatMap(
    df, 
    distance=dist,
    use_completecase_analysis=True,
    linkage="complete", 
)
