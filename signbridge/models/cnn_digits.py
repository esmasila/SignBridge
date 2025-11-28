"""
CNN Modeli - İşaret Dili Rakamları (0-9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from signbridge.config import DIGITS_MODEL_CONFIG

logger = logging.getLogger(__name__)


class DigitsCNN(nn.Module):
    """
    Basit CNN modeli - işaret dili rakamları için
    """
    
    def __init__(
        self,
        num_classes: int = DIGITS_MODEL_CONFIG["num_classes"],
        input_channels: int = DIGITS_MODEL_CONFIG["channels"],
        dropout: float = DIGITS_MODEL_CONFIG["dropout"]
    ):
        super(DigitsCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional katmanlar
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected katmanlar
        # Input: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # 4x4x256 = 4096
        self.fc1 = nn.Linear(4 * 4 * 256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"DigitsCNN oluşturuldu: {num_classes} sınıf")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, channels, height, width)
            
        Returns:
            logits: Shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


def load_digits_model(
    checkpoint_path: str,
    device: Optional[str] = None
) -> DigitsCNN:
    """
    Eğitilmiş rakam modelini yükler
    
    Args:
        checkpoint_path: Model checkpoint dosyası
        device: 'cuda' veya 'cpu'
        
    Returns:
        model: Yüklenmiş model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DigitsCNN()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model yüklendi: {checkpoint_path}")
    logger.info(f"Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test kodu
    logging.basicConfig(level=logging.INFO)
    
    # Model oluştur ve test et
    model = DigitsCNN()
    
    # Dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 64)
    
    output = model(dummy_input)
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Toplam parametre sayısı: {total_params:,}")
