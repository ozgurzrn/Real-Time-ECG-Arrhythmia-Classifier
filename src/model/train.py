import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from model import ResNet1D
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_data(data_dir='data/processed'):
    """
    Load processed data from .npy files.
    """
    X_train = np.load(os.path.join(data_dir, 'X_train_res.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_res.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    return X_train, y_train, X_test, y_test

def train_model(num_epochs=20, batch_size=64, learning_rate=0.001):
    """
    Train the ResNet1D model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ResNet1D(num_classes=5).to(device)
    
    # Class weights to address imbalance - give arrhythmias higher importance
    # Order: N, S, V, F, Q
    class_weights = torch.tensor([1.0, 5.0, 5.0, 3.0, 2.0]).to(device)
    print(f"Using class weights: N=1.0, S=5.0, V=5.0, F=3.0, Q=2.0")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    train_losses, test_losses = [], []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
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
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            
    print("Training complete.")
    
    # Plot history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('models/training_history.png')
    
    return model

def evaluate_model(model=None):
    """
    Evaluate the model and save confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        model = ResNet1D(num_classes=5).to(device)
        model.load_state_dict(torch.load('models/best_model.pth'))
        model.eval()
        
    X_train, y_train, X_test, y_test = load_data()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print(classification_report(all_labels, all_preds, target_names=['N', 'S', 'V', 'F', 'Q']))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['N', 'S', 'V', 'F', 'Q'], yticklabels=['N', 'S', 'V', 'F', 'Q'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')

if __name__ == "__main__":
    if os.path.exists('data/processed/X_train_res.npy'):
        model = train_model(num_epochs=20) # Full training for production
        evaluate_model(model)
    else:
        print("Processed data not found. Run make_dataset.py first.")
