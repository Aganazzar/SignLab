#!/usr/bin/env python3
"""
Train model on WLASL dataset (extended vocabulary)

This script trains on the larger WLASL dataset after processing.
Your existing train.py and models remain unchanged.

USAGE:
    python train_wlasl.py --data wlasl_processed --epochs 50

FEATURES:
- Supports much larger vocabulary (100-2000 signs)
- Same model architecture as current system
- Saves to models/wlasl_model.pth (separate from your existing model)
- Can combine with your existing dataset folder
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from realtime.model import SignRecognitionModel


class WLASLDataset(Dataset):
    """PyTorch Dataset for WLASL processed features"""
    
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_wlasl_data(data_dir, vocab_file="wlasl_vocab.json"):
    """
    Load processed WLASL features
    
    Returns:
        X: np.array of shape (num_samples, seq_len, feature_dim)
        y: np.array of shape (num_samples,)
        vocab: dict mapping sign_name -> index
    """
    data_path = Path(data_dir)
    
    # Load vocabulary
    vocab_path = data_path / vocab_file
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"📚 Loaded vocabulary: {len(vocab)} signs")
    
    X_list = []
    y_list = []
    expected_shape = None
    skipped = 0
    
    # Load all .npy files
    for sign_name, sign_idx in vocab.items():
        sign_folder = data_path / sign_name
        
        if not sign_folder.exists():
            continue
        
        for npy_file in sign_folder.glob("*.npy"):
            try:
                features = np.load(npy_file)
                
                # Validate shape consistency
                if expected_shape is None:
                    expected_shape = features.shape
                elif features.shape != expected_shape:
                    skipped += 1
                    continue
                
                X_list.append(features)
                y_list.append(sign_idx)
            except Exception as e:
                print(f"⚠️  Error loading {npy_file}: {e}")
                skipped += 1
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} samples with inconsistent shapes")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"   Shape: {X.shape}")
    
    return X, y, vocab


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # outputs: (batch, seq_len, num_classes)
        # We take the mean prediction across sequence
        outputs_mean = outputs.mean(dim=1)
        
        loss = criterion(outputs_mean, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs_mean, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            outputs_mean = outputs.mean(dim=1)
            
            loss = criterion(outputs_mean, batch_y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs_mean, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train on WLASL dataset")
    parser.add_argument("--data", default="wlasl_processed", help="Path to processed WLASL data")
    parser.add_argument("--model_path", default="models/wlasl_model.pth", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    
    args = parser.parse_args()
    
    # Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")
    
    # Load data
    print("📂 Loading WLASL data...")
    X, y, vocab = load_wlasl_data(args.data)
    
    feature_dim = X.shape[2]
    num_classes = len(vocab)
    
    print(f"   Features: {feature_dim}")
    print(f"   Classes: {num_classes}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create dataloaders
    train_dataset = WLASLDataset(X_train, y_train)
    test_dataset = WLASLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}\n")
    
    # Initialize model
    model = SignRecognitionModel(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_test_acc = 0.0
    
    print("🚀 Starting training...\n")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_path)
            print(f"  ✅ Best model saved! ({test_acc*100:.2f}%)")
        
        print()
    
    print(f"🎉 Training complete!")
    print(f"   Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"   Model saved to: {args.model_path}")
    
    # Save vocabulary with model
    vocab_save_path = args.model_path.replace('.pth', '_vocab.json')
    with open(vocab_save_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"   Vocabulary saved to: {vocab_save_path}")


if __name__ == "__main__":
    main()
