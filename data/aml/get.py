import pandas as pd

import pathlib
DIR = pathlib.Path(__file__).parent.resolve()

def get_dataset():
    df = pd.read_csv(DIR / "./aml.csv")
    df["ProtID"] = df["ProtID"].apply(lambda pid: pid.split(';')[0])
    df = df.set_index("ProtID")
    df.index.name = "Protein ID"
    df.columns.name = "Sample"
    return df

def get_groupmap():
    metadata_df = pd.read_csv(DIR / "./meta.csv")

    sample_groupings = {}
    mapping = metadata_df[["Sample", "Group"]].to_dict(orient="tight")["data"]
    sample_groupings["Group"] = {k:v for [k, v] in mapping}
    return sample_groupings
