import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy
import time

# Add src to path
sys.path.append('src')
from model.model import ResNet1D
from data.shayan_loader import get_shayan_loaders
from data.augmentation import Compose, RandomShift, RandomScale, GaussianNoise

def train_shayan():
    # Config
    BATCH_SIZE = 128
    EPOCHS = 5 
    LR = 1e-4
    TRAIN_PATH = 'data/external/shayan/mitbih_train.csv'
    TEST_PATH = 'data/external/shayan/mitbih_test.csv'
    MODEL_PATH = 'models/best_model.pth'
    SAVE_PATH = 'models/shayan_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define Augmentation Pipeline
    # Shift: Simulate imperfect R-peak detection
    # Scale: Simulate different gain/amplitude
    # Noise: Simulate muscle noise/baseline wander
    train_transform = Compose([
        RandomShift(max_shift=15), 
        RandomScale(min_scale=0.8, max_scale=1.2),
        GaussianNoise(sigma=0.05)
    ])
    
    # 1. Load Data
    train_loader, test_loader = get_shayan_loaders(TRAIN_PATH, TEST_PATH, batch_size=BATCH_SIZE, train_transform=train_transform)
    
    # 2. Load Model
    print(f"Loading pre-trained model from {MODEL_PATH}...")
    model = ResNet1D(num_classes=5)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(device)
    
    # 3. Setup
    # Class weights (Calculated based on distribution: N=72k, S=2k, V=5k, F=641, Q=6k)
    # We down-weight Normal and heavily up-weight Fusion/Supraventricular
    class_weights = torch.tensor([0.5, 5.0, 5.0, 10.0, 2.0]).to(device)
    print(f"Using class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("Starting Training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Train
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
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, SAVE_PATH) # Save intermediate best
            
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")
    print(f"Time elapsed: {(time.time() - start_time)/60:.1f} min")
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_shayan()
