import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
import torch
from torch.utils.data import Dataset, DataLoader

class ECG5000Dataset(Dataset):
    def __init__(self, csv_path, target_len=216):
        self.target_len = target_len
        
        # Load data
        print(f"Loading ECG5000 from {csv_path}...")
        df = pd.read_csv(csv_path, header=None)
        
        # Split signals and labels
        self.signals = df.iloc[:, :-1].values
        self.original_labels = df.iloc[:, -1].values
        
        # Map labels to AAMI (0=N, 1=S, 2=V, 3=F, 4=Q)
        # ECG5000: 1=N, 2=R-on-T(V), 3=PVC(V), 4=SP(S), 5=UB(Q)
        self.labels = np.zeros_like(self.original_labels, dtype=int)
        
        mapping = {
            1: 0, # Normal -> N
            2: 2, # R-on-T -> V
            3: 2, # PVC -> V
            4: 1, # SP -> S
            5: 4  # UB -> Q
        }
        
        for k, v in mapping.items():
            self.labels[self.original_labels == k] = v
            
        # Pre-process all signals
        print("Preprocessing signals (Resampling & Normalization)...")
        self.processed_signals = []
        for sig in self.signals:
            # Resample
            resampled = scipy_signal.resample(sig, target_len)
            # Normalize
            normalized = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-6)
            self.processed_signals.append(normalized)
            
        self.processed_signals = np.array(self.processed_signals, dtype=np.float32)
        self.labels = torch.LongTensor(self.labels)
        
        # Class distribution
        print("Class Distribution (AAMI):")
        from collections import Counter
        counts = Counter(self.labels.numpy())
        classes = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
        for k, v in counts.items():
            print(f"  {classes.get(k, k)}: {v}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return (1, 216) tensor for channel dim
        return torch.tensor(self.processed_signals[idx]).unsqueeze(0), self.labels[idx]

def get_ecg5000_loader(csv_path, batch_size=32, shuffle=True):
    dataset = ECG5000Dataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
