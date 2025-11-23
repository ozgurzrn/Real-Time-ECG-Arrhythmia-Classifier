import pandas as pd
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append('src')
from model.model import ResNet1D

# Constants
MODEL_PATH = 'models/best_model.pth'
DATA_PATH = 'data/external/ecg.csv'
TARGET_LEN = 216
CLASSES = {0: 'N (Normal)', 1: 'S (Supraventricular)', 2: 'V (Ventricular)', 3: 'F (Fusion)', 4: 'Q (Unknown)'}

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=5)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    model.to(device)
    model.eval()
    return model, device

from scipy import signal as scipy_signal

def resample_signal(sig, target_len):
    """Resample signal to target length."""
    return scipy_signal.resample(sig, target_len)

def test_external_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        # Load first 1000 rows to test
        df = pd.read_csv(DATA_PATH, header=None)
        signals = df.iloc[:, :-1].values  # All columns except last
        labels = df.iloc[:, -1].values    # Last column
        
        print(f"Loaded {len(df)} samples. Signal length: {signals.shape[1]}")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    model, device = load_model()
    
    # Process signals
    processed_signals = []
    for sig in signals:
        # 1. Resample to 216
        processed = resample_signal(sig, TARGET_LEN)
        # 2. Normalize
        normalized = (processed - np.mean(processed)) / (np.std(processed) + 1e-6)
        processed_signals.append(normalized)
        
    X = torch.FloatTensor(np.array(processed_signals)).unsqueeze(1).to(device)
    
    print("Running predictions...")
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()
        
    # Analysis
    print("\n--- Prediction Summary ---")
    from collections import Counter
    counts = Counter(preds)
    total = len(preds)
    
    for cls_idx, count in counts.items():
        cls_name = CLASSES[cls_idx]
        print(f"{cls_name}: {count} ({count/total*100:.1f}%)")
        
    # Compare with external labels (Heuristic mapping)
    # Assuming ECG5000 labels: 1=Normal, 2=R-on-T, 3=PVC, 4=SP, 5=UB
    # Our labels: 0=N, 1=S, 2=V, 3=F, 4=Q
    
    print("\n--- Sample Predictions ---")
    for i in range(5):
        print(f"Sample {i}: External Label={labels[i]} -> Prediction={CLASSES[preds[i]]} (Conf: {confidences[i]:.2f})")

if __name__ == "__main__":
    test_external_data()
