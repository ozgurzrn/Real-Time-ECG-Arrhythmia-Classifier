import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy

# Add src to path
sys.path.append('src')
from model.model import ResNet1D
from data.ecg5000 import get_ecg5000_loader

def train_finetune():
    # Config
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4 # Lower learning rate for fine-tuning
    DATA_PATH = 'data/external/ecg.csv'
    MODEL_PATH = 'models/best_model.pth'
    SAVE_PATH = 'models/finetuned_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_loader = get_ecg5000_loader(DATA_PATH, batch_size=BATCH_SIZE)
    
    # 2. Load Existing Model
    print(f"Loading pre-trained model from {MODEL_PATH}...")
    model = ResNet1D(num_classes=5)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(device)
    
    # 3. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("Starting Fine-Tuning...")
    
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    print(f"Fine-tuning complete. Best Acc: {best_acc:.4f}")
    
    # Save
    torch.save(best_model_wts, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
    
    # Update the main model file? 
    # For now, let's keep it separate, or ask user. 
    # But to make it work in the app immediately, we might want to overwrite or update app to load this.
    # Let's overwrite best_model.pth if it's really good, but safer to keep separate first.
    
if __name__ == "__main__":
    train_finetune()
