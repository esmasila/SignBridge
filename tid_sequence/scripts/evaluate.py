"""
Model Evaluation - Confusion Matrix ve Metrics
"""

import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import LSTMClassifier
from scripts.dataset import create_dataloaders


def evaluate_model(checkpoint_path, data_path, device='cuda'):
    """Model'i detaylı değerlendir"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model
    model = LSTMClassifier(
        input_size=1629,
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    label_map = checkpoint['label_map']
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # Data
    _, val_loader, _ = create_dataloaders(data_path, batch_size=16, train_split=0.8)
    
    # Predictions
    all_preds = []
    all_labels = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    target_names = [idx_to_label[i] for i in range(len(label_map))]
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = Path(__file__).parent.parent / "checkpoints" / "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\n✅ Confusion matrix saved: {save_path}")
    plt.show()
    
    # Per-class analysis
    print("\nPer-class Predictions:")
    for i, label in enumerate(target_names):
        count = (all_preds == i).sum()
        correct = ((all_preds == i) & (all_labels == i)).sum()
        print(f"  {label}: {count} predictions, {correct} correct")


if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "lstm_best.pt"
    data_path = Path(__file__).parent.parent / "data"
    
    evaluate_model(checkpoint_path, data_path, device='cuda')
