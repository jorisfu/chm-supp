# chm-supp
Supplementary material for my bachelor thesis / the clusteredheatmap library.
Shortened overview of files:

```
.
├── analysis                                # Jupyter notebooks for analysis of results
<...>
├── data
│   ├── aml                                 # Preprocessed AML dataset, see https://doi.org/10.3390/cancers12030709 (supp. mat. 1)
<...>
│   ├── ev                                  # Preprocessed EV dataset, see https://doi.org/10.1038/s42003-019-0570-8 (supp. dataset 1)
<...>
│   └── synthetic                           # Generator for synthetic 100x100 dataset
<...>
├── experiments
│   ├── missingness
│   │   ├── commontools.py                  # Tools that introduce missingness into matrices
│   │   ├── ev.py                           # Main script for EV missingness experiments
│   │   ├── runs.py                         # perform_runs method to be used for a dataset by ev.py and synthetic.py
│   │   ├── synthetic-gmmtests.py           # Tests on GMM model selection over synthetic dataset
│   │   └── synthetic.py                    # Main script for synethtic missingness experiments
│   ├── sanity.py                           
│   ├── hpi.py                              # Script for the example clustered heatmap in my introduction :)              
│   └── time
│       ├── clusterandvisu.py               # Script that clusters a dataset and creates the visu
│       ├── clusteronly.py                  # Script that only clusters a dataset
│       └── hf_clusteronly.sh               # hyperfine runner for clusteronly.py comparing the distance methods
├── README.md
├── requirements.txt
├── results
│   ├── hyperfine                           # All results from hyperfine runs
<...>
│   ├── pyinstrument                        # All results from pyinstrument runs as html
<...>
│   └── sc-runs                             # Results of the runs performed on the compute cluster
<...>
└── sc                                      # SLURM job descriptions for the runs performed on the compute cluster
<...>
```