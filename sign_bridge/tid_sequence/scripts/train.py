"""
LSTM Model Eğitim Scripti
5 kelime: MERHABA, EVET, HAYIR, LÜTFEN, TEŞEKKÜR ETMEK
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time
from datetime import datetime

# Kendi modüllerimizi import et
sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import LSTMClassifier, GRUClassifier
from scripts.dataset import create_dataloaders, save_label_map


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)  # (batch, 60, 1629)
        labels = labels.to(device)        # (batch,)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(sequences)  # (batch, num_classes)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_model(
    model_type='lstm',
    data_path='tid_sequence/data',
    batch_size=16,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    learning_rate=0.001,
    num_epochs=50,
    device='cuda',
    save_dir='tid_sequence/checkpoints'
):
    """
    LSTM/GRU modelini eğit
    
    Args:
        model_type: 'lstm' veya 'gru'
        data_path: veri klasörü
        batch_size: batch size
        hidden_size: LSTM/GRU hidden size
        num_layers: LSTM/GRU katman sayısı
        dropout: dropout oranı
        learning_rate: learning rate
        num_epochs: epoch sayısı
        device: 'cuda' veya 'cpu'
        save_dir: checkpoint kayıt klasörü
    """
    
    # Paths
    data_path = Path(data_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}")
    
    # DataLoaders
    print("\n📊 Loading data...")
    train_loader, val_loader, label_map = create_dataloaders(
        data_path, 
        batch_size=batch_size,
        train_split=0.8,
        normalize=True
    )
    
    num_classes = len(label_map)
    print(f"Classes: {num_classes}")
    print(f"Label map: {label_map}\n")
    
    # Model
    print(f"🧠 Creating {model_type.upper()} model...")
    if model_type.lower() == 'lstm':
        model = LSTMClassifier(
            input_size=1629,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type.lower() == 'gru':
        model = GRUClassifier(
            input_size=1629,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("🚀 Starting training...\n")
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start_time
        
        # Print
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_map': label_map,
                'model_type': model_type,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_classes': num_classes
            }
            
            save_path = save_dir / f"{model_type}_best.pt"
            torch.save(checkpoint, save_path)
            print(f"  ✅ Best model saved! ({val_acc:.2f}%)\n")
        else:
            print()
    
    print(f"\n🎉 Training completed!")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Label map kaydet
    label_map_path = save_dir / "label_map.json"
    save_label_map(label_map, label_map_path)
    
    return model, best_val_acc


if __name__ == "__main__":
    # Eğitim parametreleri - ORIJINAL CONFIG (çalıştı)
    config = {
        'model_type': 'lstm',
        'data_path': 'tid_sequence/data',
        'batch_size': 16,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 100,      # 50 → 100 (daha uzun eğit, belki daha iyi olur)
        'device': 'cuda',
        'save_dir': 'tid_sequence/checkpoints'
    }
    
    print("=" * 60)
    print("TID SEQUENCE MODEL TRAINING")
    print("=" * 60)
    print(f"Config: {config}\n")
    
    # Train
    model, best_acc = train_model(**config)
    
    print(f"\n✅ Training done! Best accuracy: {best_acc:.2f}%")
