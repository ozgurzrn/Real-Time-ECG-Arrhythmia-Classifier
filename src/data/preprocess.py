import numpy as np
import pandas as pd
from scipy import signal
import wfdb
import os

def denoise_signal(data, fs):
    """
    Apply a bandpass filter to denoise the ECG signal.
    """
    lowcut = 0.5
    highcut = 50.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def segment_beats(signal_data, annotations, fs, window_size=0.6):
    """
    Segments the ECG signal into beats based on R-peak annotations.
    Returns a list of beats and their corresponding labels.
    """
    beats = []
    labels = []
    
    # Convert window size to samples (e.g., 0.6s total -> 0.3s before and after)
    window_samples = int(fs * window_size / 2)
    
    for i, sample_idx in enumerate(annotations.sample):
        label = annotations.symbol[i]
        
        # Skip if beat is too close to start or end
        if sample_idx < window_samples or sample_idx + window_samples > len(signal_data):
            continue
            
        beat = signal_data[sample_idx - window_samples : sample_idx + window_samples]
        
        # Normalize beat
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
        
        beats.append(beat)
        labels.append(label)
        
    return np.array(beats), np.array(labels)

def process_record(record_name, data_dir='data/raw'):
    """
    Process a single record: load, denoise, segment.
    """
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Use the first channel
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    
    # Denoise
    clean_signal = denoise_signal(ecg_signal, fs)
    
    # Segment
    beats, labels = segment_beats(clean_signal, annotation, fs)
    
    return beats, labels

if __name__ == "__main__":
    # Test with a dummy signal if running directly
    print("Preprocessing script ready.")
