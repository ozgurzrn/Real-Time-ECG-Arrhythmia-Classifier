import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

def explore_record(record_name='100', data_dir='data/raw'):
    """
    Loads and plots a sample ECG record.
    """
    record_path = os.path.join(data_dir, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        print(f"Record: {record_name}")
        print(f"Sampling Frequency: {record.fs} Hz")
        print(f"Signal Length: {record.sig_len} samples")
        print(f"Annotations: {len(annotation.sample)} beats")
        
        # Plot first 10 seconds
        signals, fields = wfdb.rdsamp(record_path, sampto=3600)
        
        plt.figure(figsize=(15, 5))
        plt.plot(signals[:, 0], label='Channel 0')
        plt.title(f'ECG Signal (Record {record_name}) - First 10s')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'data/sample_{record_name}.png')
        print(f"Plot saved to data/sample_{record_name}.png")
        
    except Exception as e:
        print(f"Error loading record: {e}")

if __name__ == "__main__":
    # Check if data exists first
    if os.path.exists('data/raw/100.dat'):
        explore_record()
    else:
        print("Data not found. Please run download_data.py first.")
