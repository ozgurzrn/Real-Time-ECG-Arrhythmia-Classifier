import torch
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path to allow imports
sys.path.append('src')
from model.ensemble_model import SOTAEnsembleModel

# Config
DATA_PATH = 'data/new_test_data_incart.csv'
MODEL_PATH = 'models/sota_ensemble_model.pth'
CLASSES = {0: 'N (Normal)', 1: 'S (Supra)', 2: 'V (Ventricular)', 3: 'F (Fusion)', 4: 'Q (Unknown)'}

def run_test():
    print("===================================================")
    print("  External Validation: St. Petersburg INCART DB")
    print("===================================================")
    
    # 1. Check Files
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("Please run GET_NEW_DATA.bat first.")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Please train the model first.")
        return

    # 2. Load Data
    print(f"[INFO] Loading data from {DATA_PATH}...")
    try:
        # Load without header
        df = pd.read_csv(DATA_PATH, header=None)
        
        # Split Signal / Label
        # Columns 0-186 are Signal (187 points), Column 187 is Label
        X = df.iloc[:, :-1].values
        y_true = df.iloc[:, -1].values.astype(int)
        
        print(f"[INFO] Samples: {len(X)}")
        
        # Convert to Tensor
        # Model expects (Batch, 1, 187)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_true, dtype=torch.long)
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # 3. Load Model
    print(f"[INFO] Loading SOTA Model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SOTAEnsembleModel(num_classes=5)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        return

    # 4. Run Inference
    print("[INFO] Running Inference...")
    batch_size = 1024 # Large batch for speed since no grad
    y_pred = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i : i+batch_size].to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            
    # 5. Metrics
    print("\n---------------------------------------------------")
    print("  RESULTS REPORT")
    print("---------------------------------------------------")
    
    # Overall Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Final Accuracy Score: {acc*100:.2f}%")
    
    # Detailed Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=[CLASSES[i] for i in range(5)], digits=4)
    print(report)
    
    # Confusion Matrix
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    cm = confusion_matrix(y_true, y_pred)
    labels = ['N', 'S', 'V', 'F', 'Q']
    
    # Pretty Print CM
    print(f"{'':<5} " + " ".join([f"{l:<6}" for l in labels]))
    for i, row in enumerate(cm):
        print(f"{labels[i]:<5} " + " ".join([f"{val:<6}" for val in row]))
        
    print("\n---------------------------------------------------")
    if acc > 0.90:
        print("[SUCCESS] Model Generalized Successfully (>90%)!")
    else:
        print("[WARNING] Generalization Gap Detected (<90%). Fine-tuning may be required.")

if __name__ == "__main__":
    run_test()
