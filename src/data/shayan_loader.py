import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
import torch
from torch.utils.data import Dataset, DataLoader

class ShayanDataset(Dataset):
    def __init__(self, csv_path, target_len=216, transform=None):
        self.target_len = target_len
        self.transform = transform
        
        print(f"Loading Shayan dataset from {csv_path}...")
        # Read CSV (no header)
        df = pd.read_csv(csv_path, header=None)
        
        # Last column is label
        self.labels = df.iloc[:, -1].values.astype(int)
        self.signals = df.iloc[:, :-1].values
        
        print(f"  Samples: {len(df)}")
        print(f"  Original Length: {self.signals.shape[1]}")
        
        # Pre-process
        print("  Preprocessing (Resampling & Normalization)...")
        self.processed_signals = []
        for sig in self.signals:
            # Resample 187 -> 216
            resampled = scipy_signal.resample(sig, target_len)
            # Normalize
            normalized = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-6)
            self.processed_signals.append(normalized)
            
        self.processed_signals = np.array(self.processed_signals, dtype=np.float32)
        self.labels = torch.LongTensor(self.labels)
        
        # Class distribution
        from collections import Counter
        counts = Counter(self.labels.numpy())
        classes = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
        print("  Class Distribution:")
        for k, v in counts.items():
            print(f"    {classes.get(k, k)}: {v}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.processed_signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal, label = self.transform((signal, label))
            
        return torch.tensor(signal).unsqueeze(0), label

def get_shayan_loaders(train_path, test_path, batch_size=64, train_transform=None):
    train_dataset = ShayanDataset(train_path, transform=train_transform)
    test_dataset = ShayanDataset(test_path, transform=None) # No transform for test
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
