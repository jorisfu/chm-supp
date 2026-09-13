import pandas as pd

def get_dataset():
    df = pd.read_csv("./ev.csv")
    df["Majority protein IDs"] = df["Majority protein IDs"].apply(lambda pid: str(pid).split(';')[0])


    # Make replicates more concise
    for i in df.columns:
        w = i.split()
        name = w[1] + "_rep" + w[-1]
        df.rename(columns={i: name}, inplace=True)

    # Get mappings
    mapping_df = df.drop(columns=["protein_repIDs", "Significant_repSignificant"]).iloc[1]
    lk = mapping_df.keys()
    lv = list(mapping_df)
    mapping = {k:v for k, v in zip(lk, lv)}
    mapping

    col_map = {"Cancer": mapping}

    # Cleanup and filter
    df = df.set_index("protein_repIDs")
    df.index.name = "Protein ID"
    df = df[df["Significant_repSignificant"] == "+"]
    df = df.drop(columns=["Significant_repSignificant"])
    df = df = df.apply(pd.to_numeric)

    def z_score_normalization(row):
        return (row - row.mean()) / row.std()

    # Apply z-score
    df.update(df.apply(z_score_normalization, axis=1))

    return df
