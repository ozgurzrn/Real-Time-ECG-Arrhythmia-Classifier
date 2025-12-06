import wfdb
import os

def debug_load():
    record_path = 'data/external/physionet/nsrdb/16265'
    print(f"Attempting to load {record_path}...")
    
    try:
        # Try reading header first
        header = wfdb.rdheader(record_path)
        print(f"Header loaded. FS: {header.fs}, Sig Len: {header.sig_len}")
        
        # Try reading small chunk
        sampto = 1000
        record = wfdb.rdrecord(record_path, sampto=sampto)
        print(f"Record chunk loaded. Shape: {record.p_signal.shape}")
        
        # Try reading annotation
        ann = wfdb.rdann(record_path, 'atr', sampto=sampto)
        print(f"Annotation chunk loaded. Samples: {len(ann.sample)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_load()
