import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# Add src to path
sys.path.append('src')
from model.ensemble_model import SOTAEnsembleModel
from data.sota_loader import get_sota_loaders
from data.augmentation import Compose, RandomShift, RandomScale, GaussianNoise, RandomStretch

def train_sota():
    # Config
    BATCH_SIZE = 64
    EPOCHS = 5 # Increase for production
    LR = 1e-3
    TRAIN_PATH = 'data/external/shayan/mitbih_train.csv'
    TEST_PATH = 'data/external/shayan/mitbih_test.csv'
    SAVE_PATH = 'models/sota_ensemble_model.pth'
    PLOT_DIR = 'reports/figures'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Augmentation
    train_transform = Compose([
        RandomShift(max_shift=10),
        RandomScale(min_scale=0.9, max_scale=1.1),
        RandomStretch(min_stretch=0.95, max_stretch=1.05),
        GaussianNoise(sigma=0.02)
    ])
    
    # 1. Load Data
    print("Initializing Data Loaders...")
    train_loader, test_loader = get_sota_loaders(TRAIN_PATH, TEST_PATH, batch_size=BATCH_SIZE, train_transform=train_transform)
    
    # 2. Model
    model = SOTAEnsembleModel(num_classes=5)
    model.to(device)
    
    # 3. Optimization
    # Since we used SMOTE, classes are balanced. We can use standard CrossEntropy or LabelSmoothing.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Lists for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("Starting Training Loop...")
    start_time = time.time()
    
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
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, SAVE_PATH)
            
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    
    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Metrics
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['N', 'S', 'V', 'F', 'Q'], yticklabels=['N', 'S', 'V', 'F', 'Q'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{PLOT_DIR}/confusion_matrix_sota.png')
    
    # ROC-AUC
    # Binarize labels
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    n_classes = y_test_bin.shape[1]
    all_probs = np.array(all_probs)
    
    try:
        roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr')
        print(f"ROC-AUC Score (OVR): {roc_auc:.4f}")
        
        # Plot ROC
        from sklearn.metrics import roc_curve, auc
        plt.figure()
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
            roc_auc_i = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc_i:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'{PLOT_DIR}/roc_sota.png')
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")

if __name__ == "__main__":
    train_sota()
