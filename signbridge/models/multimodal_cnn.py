"""
Multimodal CNN Modeli - Rakam Tanıma için Görüntü + Landmark

ÖNEMLİ: Bu model RAKAMLARI (0-9) tanımak için kullanılıyor ve aktif olarak çalışıyor.
Checkpoint: checkpoints/multimodal_digits_best.pt (99.79% validation accuracy, 27 FPS)

Mimari:
- Görüntü dalı: 4 katmanlı CNN (32→64→128→256 filtre)
- Landmark dalı: 2 katmanlı MLP (landmark_dim→512→256)
- Fusion: Concat + FC layers
- Input: 64x64 RGB görüntü + 1629-dim landmark vektörü
- Output: 10 sınıf (0-9 rakamları)

MediaPipe Holistic kullanımı:
- 543 landmark × 3 koordinat (x,y,z) = 1629 özellik
- Padding ile sabit boyutta tutuluyor
- Gerçek zamanlı çıkarım için optimize edilmiş

Kullanım:
- signbridge/api/service.py - Web API endpoint'lerinde
- web/index.html - Frontend'de gerçek zamanlı tahmin
- FPS: ~27 (GPU), ~12 (CPU)
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from signbridge.config import DIGITS_MODEL_CONFIG


class MultimodalDigitsCNN(nn.Module):
    """
    Hem görüntü hem de landmark verisi kullanabilen CNN
    """
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        use_landmarks: bool = True,
        landmark_dim: int = 543 * 3  # 543 landmark * (x, y, z)
    ):
        super().__init__()
        self.use_landmarks = use_landmarks
        
        # Görüntü dalı (CNN)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x64 -> 32x32
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x32 -> 16x16
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8 -> 4x4
        )
        
        # CNN feature size: 256 * 4 * 4 = 4096
        cnn_feature_size = 256 * 4 * 4
        
        # Landmark dalı (MLP)
        if self.use_landmarks:
            self.landmark_fc = nn.Sequential(
                nn.Linear(landmark_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            combined_size = cnn_feature_size + 256
        else:
            combined_size = cnn_feature_size
        
        # Birleştirilmiş özellikler
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None):
        """
        Args:
            x: Görüntü [B, C, H, W]
            landmarks: Landmark'lar [B, landmark_dim] (opsiyonel)
        """
        # CNN dalı
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Landmark dalı
        if self.use_landmarks and landmarks is not None:
            landmark_features = self.landmark_fc(landmarks)
            # CNN ve landmark özelliklerini birleştir
            x = torch.cat([x, landmark_features], dim=1)
        
        # Sınıflandırma
        x = self.fc(x)
        return x


def save_multimodal_model(
    model: nn.Module,
    save_path: str,
    metadata: dict = None
) -> None:
    """Model ve metadata'yı kaydet"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': model.fc[-1].out_features,
            'use_landmarks': model.use_landmarks
        }
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    print(f"Model kaydedildi: {save_path}")


def load_multimodal_model(
    model_path: str,
    device: str = "cpu"
) -> nn.Module:
    """Multimodal model yükle"""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('model_config', {})
    num_classes = config.get('num_classes', 10)
    use_landmarks = config.get('use_landmarks', True)
    
    model = MultimodalDigitsCNN(
        num_classes=num_classes,
        use_landmarks=use_landmarks
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test
    model = MultimodalDigitsCNN(num_classes=10, use_landmarks=True)
    
    # Sadece görüntü
    img = torch.randn(2, 3, 64, 64)
    output = model(img)
    print(f"Sadece görüntü output: {output.shape}")  # [2, 10]
    
    # Görüntü + landmark
    landmarks = torch.randn(2, 543 * 3)
    output = model(img, landmarks)
    print(f"Görüntü + landmark output: {output.shape}")  # [2, 10]
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre: {total_params:,}")
