import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append('src')
# Robust Import for when script is run from different contexts
try:
    from model.ensemble_model import SOTAEnsembleModel
except (ImportError, ModuleNotFoundError):
    # Fallback for when 'model' resolves to a module instead of package (shadowing)
    from ensemble_model import SOTAEnsembleModel

# Config
DATA_PATH = 'data/new_test_data_incart.csv'
BASE_MODEL_PATH = 'models/sota_ensemble_model.pth'
FINETUNED_MODEL_PATH = 'models/sota_finetuned_russia.pth'
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

# Classes: 0:N, 1:S, 2:V, 3:F, 4:Q
CLASSES = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

class FineTuneDataset(Dataset):
    def __init__(self, X, y):
        self.X = tuple(torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in X)
        self.y = tuple(torch.tensor(label, dtype=torch.long) for label in y)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def run_fine_tuning():
    print("===================================================")
    print("  Domain Adaptation: Fine-Tuning (INCART)")
    print("===================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data not found at {DATA_PATH}. Run GET_NEW_DATA.bat first.")
        return

    print("[INFO] Loading External Data...")
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    
    # 2. Split Data (50% Calibration, 50% Test)
    # Using stratify to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    
    print(f"[INFO] Calibration Set: {len(X_train)} samples")
    print(f"[INFO] Final Exam Set: {len(X_test)} samples")
    
    train_dataset = FineTuneDataset(X_train, y_train)
    test_dataset = FineTuneDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Pre-Trained Model
    print(f"[INFO] Loading Pre-Trained Weights from {BASE_MODEL_PATH}...")
    model = SOTAEnsembleModel(num_classes=5)
    try:
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device, weights_only=False))
    except Exception as e:
        print(f"[ERROR] Could not load base model: {e}")
        return
        
    model.to(device)
    
    # 4. Setup Optimization
    # Weighted Loss: Penalize mistakes on 'V' (Class 2) more heavily
    # Weights: N=1, S=2, V=5, F=2, Q=2
    class_weights = torch.tensor([1.0, 2.0, 5.0, 2.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Low Learning Rate for Fine-Tuning
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 5. Training Loop
    print(f"[INFO] Starting Calibration ({EPOCHS} Epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = correct / total
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/total:.4f} | Acc: {epoch_acc*100:.2f}%")
        
    # 6. Evaluation on Test Set
    print("\n[INFO] Running Final Exam on Unseen Test Set...")
    model.eval()
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
            
    # Reports
    print("\n---------------------------------------------------")
    print("  FINE-TUNING RESULTS")
    print("---------------------------------------------------")
    acc = accuracy_score(y_true_list, y_pred_list)
    print(f"Final Accuracy Score: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    # Dynamic class handling to avoid crashes if a class is missing in y_true
    unique_labels = sorted(list(set(y_true_list)))
    target_names = [CLASSES[i] for i in unique_labels]
    
    print(classification_report(y_true_list, y_pred_list, labels=unique_labels, target_names=target_names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_list, y_pred_list)
    print(cm)
    
    # 7. Save Model
    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
    print(f"\n[SUCCESS] Fine-Tuned Model saved to {FINETUNED_MODEL_PATH}")

if __name__ == "__main__":
    run_fine_tuning()
