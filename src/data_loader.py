import pandas as pd
import os

def load_and_merge(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    for file in all_files:
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df
