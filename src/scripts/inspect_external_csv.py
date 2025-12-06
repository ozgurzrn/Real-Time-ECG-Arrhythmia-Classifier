import pandas as pd
import numpy as np

file_path = 'data/external/ecg.csv'
print(f"Loading {file_path}...")

try:
    # Read first few rows to check structure
    df = pd.read_csv(file_path, header=None, nrows=5)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    print("First row sample:")
    print(df.iloc[0, :10].values)
    print("Last column values (potential labels):")
    print(df.iloc[:, -1].values)
    
    # Check if last column is integer (labels)
    last_col = df.iloc[:, -1]
    if last_col.dtype in [np.int64, np.float64] and last_col.nunique() < 10:
        print(f"Likely labels found in last column. Unique values: {last_col.unique()}")
    else:
        print("Last column does not look like categorical labels.")
        
except Exception as e:
    print(f"Error: {e}")
