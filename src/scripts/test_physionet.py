import wfdb
import numpy as np
import pandas as pd
import torch
import os
from scipy import signal as scipy_signal
import sys
sys.path.append('src')
from model.model import ResNet1D
# from data.preprocess import normalize_signal, segment_beats

# Constants
TARGET_FS = 360
WINDOW_SIZE = 216  # 0.6s at 360Hz
CLASSES = ['N', 'S', 'V', 'F', 'Q']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = ResNet1D(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def process_physionet_record(record_path, db_type, duration_sec=300):
    """
    Load, resample, and process a PhysioNet record.
    db_type: 'nsrdb' or 'svdb'
    duration_sec: Limit processing to first N seconds (default 300s = 5 min)
    """
    # Read header to get fs
    try:
        header = wfdb.rdheader(record_path)
        fs = header.fs
        sampto = int(duration_sec * fs)
        
        record = wfdb.rdrecord(record_path, sampto=sampto)
        annotation = wfdb.rdann(record_path, 'atr', sampto=sampto)
    except Exception as e:
        print(f"Error reading {record_path}: {e}")
        return None, None, None

    fs = record.fs
    sig = record.p_signal[:, 0]  # Use channel 0
    
    # Resample to 360Hz if needed
    if fs != TARGET_FS:
        num_samples = int(len(sig) * TARGET_FS / fs)
        sig = scipy_signal.resample(sig, num_samples)
        # Scale annotations
        ann_sample = (annotation.sample * TARGET_FS / fs).astype(int)
        ann_symbol = annotation.symbol
    else:
        ann_sample = annotation.sample
        ann_symbol = annotation.symbol

    # 1. Patient-Level Normalization (Crucial!)
    # Normalize the entire recording before segmentation
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

    # 2. Segment Beats based on annotations
    beats = []
    labels = []
    
    # Mapping to AAMI classes
    # N: Normal
    # S: Supraventricular (A, a, J, S)
    # V: Ventricular (V, E)
    # F: Fusion (F)
    # Q: Paced/Unknown (/, f, Q)
    
    aami_map = {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
        'A': 1, 'a': 1, 'J': 1, 'S': 1,
        'V': 2, 'E': 2,
        'F': 3,
        '/': 4, 'f': 4, 'Q': 4
    }

    half_window = WINDOW_SIZE // 2

    for i, sample in enumerate(ann_sample):
        symbol = ann_symbol[i]
        
        # Skip if symbol not in our map
        if symbol not in aami_map:
            continue
            
        label = aami_map[symbol]
        
        # Extract window
        start = sample - half_window
        end = sample + half_window
        
        if start < 0 or end > len(sig_norm):
            continue
            
        beat = sig_norm[start:end]
        
        # Ensure exact length
        if len(beat) != WINDOW_SIZE:
            continue
            
        beats.append(beat)
        labels.append(label)

    if not beats:
        return None, None, None

    return np.array(beats), np.array(labels), fs

def evaluate_database(db_name, records, model):
    print(f"\nEvaluating {db_name.upper()} Database...", flush=True)
    print("-" * 50, flush=True)
    
    all_preds = []
    all_labels = []
    
    for rec_name in records:
        path = f"data/external/physionet/{db_name}/{rec_name}"
        # Check if file exists (checking .dat or .hea)
        if not os.path.exists(path + ".hea"):
            print(f"Skipping {rec_name} (not found at {path}.hea)", flush=True)
            continue
            
        print(f"Processing Record {rec_name}...", end=" ", flush=True)
        
        beats, labels, original_fs = process_physionet_record(path, db_name)
        
        if beats is None:
            print("Failed to process.", flush=True)
            continue
            
        print(f"Got {len(beats)} beats.", end=" ", flush=True)

        # Convert to tensor
        X = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            
        preds = preds.cpu().numpy()
        
        # Calculate accuracy for this record
        acc = np.mean(preds == labels)
        print(f"Acc: {acc:.2%}, Orig FS: {original_fs}Hz", flush=True)
        
        # Show confusion for this record
        unique, counts = np.unique(preds, return_counts=True)
        pred_counts = dict(zip(unique, counts))
        print(f"  Preds: {pred_counts}", flush=True)
        
        all_preds.extend(preds)
        all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels)

def main():
    model = load_model("models/best_model.pth")
    
    # 1. Test NSRDB (Normal Sinus Rhythm)
    # Expectation: High accuracy on Class 0 (N)
    nsr_records = ['16265', '16272']
    nsr_preds, nsr_labels = evaluate_database('nsrdb', nsr_records, model)
    
    if len(nsr_labels) > 0:
        acc = np.mean(nsr_preds == nsr_labels)
        print(f"\nNSRDB Overall Accuracy: {acc:.2%}")
        
    # 2. Test SVDB (Supraventricular)
    # Expectation: Should detect Class 1 (S), but this is HARD.
    sv_records = ['800', '801']
    sv_preds, sv_labels = evaluate_database('svdb', sv_records, model)
    
    if len(sv_labels) > 0:
        acc = np.mean(sv_preds == sv_labels)
        print(f"\nSVDB Overall Accuracy: {acc:.2%}")
        
        # Specific S-class accuracy
        s_mask = sv_labels == 1
        if np.sum(s_mask) > 0:
            s_acc = np.mean(sv_preds[s_mask] == 1)
            print(f"Sensitivity to S-class: {s_acc:.2%}")

if __name__ == "__main__":
    main()
