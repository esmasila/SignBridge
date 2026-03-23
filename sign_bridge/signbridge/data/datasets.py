"""
PyTorch veri setleri
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List
import logging

from signbridge.config import (
    SEQUENCE_LENGTH,
    DIGITS_MODEL_CONFIG,
    TRAINING_CONFIG
)

logger = logging.getLogger(__name__)


class DigitsDataset(Dataset):
    """
    İşaret dili rakamları (0-9) için görüntü veri seti
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform=None,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Args:
            data_dir: Dataset klasörü (içinde 0, 1, ..., 9 klasörleri olmalı)
            transform: torchvision transforms
            max_samples_per_class: Her sınıftan maksimum örnek sayısı
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Her sınıf klasörünü tara
        for class_idx in range(10):
            class_dir = self.data_dir / str(class_idx)
            if not class_dir.exists():
                logger.warning(f"Klasör bulunamadı: {class_dir}")
                continue
            
            # Görüntü dosyalarını topla
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for img_path in image_files:
                self.samples.append(img_path)
                self.labels.append(class_idx)
        
        logger.info(f"DigitsDataset yüklendi: {len(self.samples)} örnek, 10 sınıf")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükle
        image = Image.open(img_path).convert("RGB")
        
        # Transform uygula
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LandmarkSequenceDataset(Dataset):
    """
    Landmark sekansları için veri seti
    İleride AUTSL ve BosphorusSign22k için kullanılacak
    """
    
    def __init__(
        self,
        npz_path: Path,
        sequence_length: int = SEQUENCE_LENGTH,
        normalize: bool = True
    ):
        """
        Args:
            npz_path: İşlenmiş landmark verileri (.npz)
            sequence_length: Sekans uzunluğu
            normalize: Normalize edilsin mi?
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Verileri yükle
        data = np.load(npz_path)
        self.landmarks = data["landmarks"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        
        # Eğer sekans uzunluğu 1'den büyükse, sekanslar oluştur
        # Şimdilik basit: her örneği sequence_length kere tekrarla
        # TODO: Video veri setleri için gerçek sekans işleme
        
        logger.info(f"LandmarkSequenceDataset yüklendi: {len(self.landmarks)} örnek")
        logger.info(f"Feature boyutu: {self.landmarks.shape[1]}")
    
    def __len__(self) -> int:
        return len(self.landmarks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        landmarks = self.landmarks[idx]
        label = self.labels[idx]
        
        # Normalize
        if self.normalize:
            landmarks = (landmarks - landmarks.mean()) / (landmarks.std() + 1e-8)
        
        # Sekansa dönüştür (şimdilik tekrarla)
        # Shape: (sequence_length, feature_dim)
        sequence = np.tile(landmarks, (self.sequence_length, 1))
        
        return torch.from_numpy(sequence), label


class RealtimeLandmarkBuffer:
    """
    Gerçek zamanlı çıkarım için landmark buffer'ı
    """
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.buffer = []
    
    def add(self, landmarks: np.ndarray):
        """Yeni landmark ekle"""
        self.buffer.append(landmarks)
        
        # Buffer'ı sequence_length ile sınırla
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Mevcut sekansı döndür
        
        Returns:
            sequence: Shape (sequence_length, feature_dim) veya None
        """
        if len(self.buffer) < self.sequence_length:
            return None
        
        sequence = np.array(self.buffer[-self.sequence_length:])
        return sequence
    
    def reset(self):
        """Buffer'ı temizle"""
        self.buffer = []
    
    def is_ready(self) -> bool:
        """Sekans hazır mı?"""
        return len(self.buffer) >= self.sequence_length


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = TRAINING_CONFIG["batch_size"],
    validation_split: float = TRAINING_CONFIG["validation_split"],
    test_split: float = TRAINING_CONFIG["test_split"],
    num_workers: int = TRAINING_CONFIG["num_workers"],
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train, validation ve test dataloader'ları oluşturur
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch boyutu
        validation_split: Validation oranı
        test_split: Test oranı
        num_workers: DataLoader worker sayısı
        shuffle: Karıştırılsın mı?
        
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Bölümlere ayır
    test_size = int(test_split * dataset_size)
    val_size = int(validation_split * dataset_size)
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    # Subset'ler oluştur
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # DataLoader'lar oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"DataLoader'lar oluşturuldu:")
    logger.info(f"  Train: {len(train_dataset)} örnek")
    logger.info(f"  Validation: {len(val_dataset)} örnek")
    logger.info(f"  Test: {len(test_dataset)} örnek")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test kodu
    from signbridge.config import DATASET_PATHS
    
    logging.basicConfig(level=logging.INFO)
    
    # Digits dataset test
    digits_path = DATASET_PATHS["digits"]
    if digits_path.exists():
        dataset = DigitsDataset(digits_path)
        logger.info(f"İlk örnek: {dataset[0]}")
    else:
        logger.error(f"Veri seti bulunamadı: {digits_path}")
