"""
Veri Ön İşleme ve Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


class TIDSequenceDataset(Dataset):
    """TID Sequence Dataset"""
    
    def __init__(self, sequences, labels):
        """
        Args:
            sequences: list of numpy arrays, each (60, 1629)
            labels: list of integers (class indices)
        """
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])  # (60, 1629)
        label = torch.LongTensor([self.labels[idx]])[0]    # scalar
        return sequence, label


def load_data(data_path, normalize=True):
    """
    .npy dosyalarını yükle ve normalize et
    
    Args:
        data_path: tid_sequence/data/ path
        normalize: True ise landmark'ları [0,1] aralığına normalize et
        
    Returns:
        sequences: list of numpy arrays (60, 1629)
        labels: list of integers
        label_map: dict {class_name: class_idx}
    """
    data_path = Path(data_path)
    sequences = []
    labels = []
    label_map = {}
    
    # Klasörleri tara (her klasör bir sınıf)
    class_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
    
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        label_map[class_name] = class_idx
        
        # .npy dosyalarını yükle
        npy_files = list(class_folder.glob("*.npy"))
        print(f"Loading {class_name}: {len(npy_files)} samples")
        
        for npy_file in npy_files:
            seq = np.load(npy_file)  # (60, 1629)
            
            # Normalize (landmark'lar zaten 0-1 arası ama garantiye alalım)
            if normalize:
                # Min-max normalization (0-1 range)
                seq_min = seq.min()
                seq_max = seq.max()
                if seq_max - seq_min > 0:
                    seq = (seq - seq_min) / (seq_max - seq_min)
            
            sequences.append(seq)
            labels.append(class_idx)
    
    print(f"\nTotal: {len(sequences)} sequences, {len(label_map)} classes")
    print(f"Label map: {label_map}")
    
    return sequences, labels, label_map


def create_dataloaders(data_path, batch_size=16, train_split=0.8, normalize=True):
    """
    Train ve validation DataLoader'ları oluştur
    
    Args:
        data_path: tid_sequence/data/ path
        batch_size: batch size
        train_split: train oranı (0.8 = %80 train, %20 val)
        normalize: normalize flag
        
    Returns:
        train_loader, val_loader, label_map
    """
    # Veriyi yükle
    sequences, labels, label_map = load_data(data_path, normalize)
    
    # Train/Val split (stratified - her sınıftan eşit oranda)
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, 
        test_size=1-train_split, 
        stratify=labels,  # Her sınıftan eşit oran
        random_state=42
    )
    
    print(f"\nSplit: {len(train_seqs)} train, {len(val_seqs)} val")
    
    # Dataset'ler oluştur
    train_dataset = TIDSequenceDataset(train_seqs, train_labels)
    val_dataset = TIDSequenceDataset(val_seqs, val_labels)
    
    # DataLoader'lar
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Windows için 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, label_map


def save_label_map(label_map, save_path):
    """Label map'i JSON olarak kaydet"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Label map saved: {save_path}")


def load_label_map(load_path):
    """Label map'i JSON'dan yükle"""
    with open(load_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


if __name__ == "__main__":
    # Test
    data_path = Path(__file__).parent.parent / "data"
    
    # Veri yükle
    sequences, labels, label_map = load_data(data_path, normalize=True)
    
    # DataLoader'lar oluştur
    train_loader, val_loader, label_map = create_dataloaders(
        data_path, 
        batch_size=16, 
        train_split=0.8
    )
    
    # Bir batch test et
    for batch_seqs, batch_labels in train_loader:
        print(f"\nBatch test:")
        print(f"  Sequences: {batch_seqs.shape}")  # (batch, 60, 1629)
        print(f"  Labels: {batch_labels.shape}")    # (batch,)
        print(f"  Label values: {batch_labels[:5]}")
        break
    
    # Label map kaydet
    save_path = Path(__file__).parent.parent / "models" / "label_map.json"
    save_label_map(label_map, save_path)
