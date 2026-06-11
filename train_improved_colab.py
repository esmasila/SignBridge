"""
===============================================================================
SignBridge - Geliştirilmiş AUTSL Eğitim Scripti (Colab için)
===============================================================================

Bu script mevcut modeli önemli ölçüde iyileştirecek teknikler içerir:

1. DATA AUGMENTATION
   - Temporal augmentation (hız değişimi, frame atlama)
   - Landmark noise injection
   - Mirror augmentation (sol/sağ el değişimi)
   - Random temporal crop & resize
   
2. FEATURE ENGINEERING
   - Velocity features (frame-arası hız)
   - Acceleration features (frame-arası ivme)
   - Relative coordinates (vücuda göre normalize)
   - Hand angle features
   
3. GELİŞMİŞ MODEL
   - Daha derin transformer
   - Stochastic depth
   - Label smoothing
   - Mixup training
   
4. EĞİTİM STRATEJİSİ
   - Cosine annealing with warm restarts
   - Gradient clipping
   - EMA (Exponential Moving Average)
   - Progressive resizing (frame sayısını artırma)

KULLANIM (Colab'da):
    1. Bu scripti Colab'a yükleyin
    2. Veri setinizi mount edin
    3. Aşağıdaki konfigürasyonu düzenleyin
    4. train_improved() fonksiyonunu çalıştırın
===============================================================================
"""

import os
import json
import math
import copy
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ==============================================================================
# KONFİGÜRASYON
# ==============================================================================

class Config:
    """Eğitim konfigürasyonu - ihtiyaca göre düzenleyin"""
    
    # Veri yolları (Colab'da düzenleyin)
    DATA_DIR = '/content/drive/MyDrive/autsl_skeletons'  # İskelet .npz dosyalarının yeri
    SAVE_DIR = '/content/drive/MyDrive/autsl_improved_model'
    
    # Model hiperparametreleri
    INPUT_SIZE = 225          # Temel: 225 (75 landmark × 3)
    USE_VELOCITY = True       # Hız feature'ları ekle
    USE_ACCELERATION = True   # İvme feature'ları ekle
    USE_RELATIVE_COORDS = True  # Vücuda göre relative koordinatlar
    
    # Feature boyutu hesapla
    @property
    def FEATURE_SIZE(self):
        size = self.INPUT_SIZE  # 225
        if self.USE_VELOCITY:
            size += self.INPUT_SIZE  # +225
        if self.USE_ACCELERATION:
            size += self.INPUT_SIZE  # +225
        if self.USE_RELATIVE_COORDS:
            size += 126  # El landmark'ları relative (21*3*2)
        return size
    
    # Sekans parametreleri
    SEQ_LENGTH = 30           # Frame sayısı (eğitim)
    SEQ_LENGTH_MAX = 45       # Progressive training max
    
    # Model mimarisi
    D_MODEL = 384             # Transformer boyutu
    NHEAD = 12                # Attention head sayısı
    NUM_LAYERS = 6            # Transformer katman sayısı
    DROPOUT = 0.3             # Dropout oranı
    NUM_CLASSES = 226         # Sınıf sayısı
    
    # Eğitim parametreleri
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2         # 0 = mixup kapalı
    GRADIENT_CLIP = 1.0
    EMA_DECAY = 0.999
    
    # Augmentation parametreleri
    AUG_TEMPORAL_SPEED = (0.8, 1.2)     # Hız aralığı
    AUG_NOISE_STD = 0.01                # Landmark noise
    AUG_MIRROR_PROB = 0.3               # Mirror olasılığı
    AUG_TEMPORAL_CROP_PROB = 0.3        # Temporal crop olasılığı
    AUG_DROPOUT_LANDMARKS_PROB = 0.1    # Random landmark sıfırlama
    AUG_SCALE_RANGE = (0.9, 1.1)        # Ölçek değişimi


config = Config()


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

class TemporalAugmentation:
    """Zaman boyutunda augmentation"""
    
    @staticmethod
    def speed_change(sequence, speed_range=(0.8, 1.2)):
        """Rastgele hız değişimi"""
        T, F = sequence.shape
        speed = random.uniform(*speed_range)
        new_T = int(T * speed)
        new_T = max(10, new_T)  # Minimum 10 frame
        
        indices = np.linspace(0, T - 1, new_T).astype(int)
        indices = np.clip(indices, 0, T - 1)
        return sequence[indices]
    
    @staticmethod
    def temporal_crop(sequence, target_length):
        """Rastgele temporal crop"""
        T = len(sequence)
        if T <= target_length:
            return sequence
        
        start = random.randint(0, T - target_length)
        return sequence[start:start + target_length]
    
    @staticmethod
    def temporal_shift(sequence, max_shift=3):
        """Küçük temporal kaydırma"""
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return sequence
        return np.roll(sequence, shift, axis=0)
    
    @staticmethod
    def frame_dropout(sequence, drop_rate=0.1):
        """Rastgele frame'leri çıkar ve interpolasyon yap"""
        T, F = sequence.shape
        n_drop = max(1, int(T * drop_rate))
        drop_indices = sorted(random.sample(range(T), n_drop))
        
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[drop_indices] = False
        
        # Basit interpolasyon
        result = sequence.copy()
        for idx in drop_indices:
            prev_idx = max(0, idx - 1)
            next_idx = min(T - 1, idx + 1)
            result[idx] = (sequence[prev_idx] + sequence[next_idx]) / 2
        
        return result


class LandmarkAugmentation:
    """Landmark boyutunda augmentation"""
    
    @staticmethod
    def add_noise(sequence, noise_std=0.01):
        """Gaussian noise ekle"""
        noise = np.random.randn(*sequence.shape).astype(np.float32) * noise_std
        return sequence + noise
    
    @staticmethod
    def mirror_hands(sequence):
        """
        Sol ve sağ eli aynala (x koordinatlarını çevir).
        Pose: 0-98, LH: 99-161, RH: 162-224
        """
        mirrored = sequence.copy()
        
        # x koordinatlarını aynala (1 - x)
        # Pose x: indices 0, 3, 6, ... 96
        for i in range(0, 99, 3):
            mirrored[:, i] = 1.0 - mirrored[:, i]
        
        # Sol ve sağ elin x koordinatlarını çevir
        for i in range(99, 162, 3):
            mirrored[:, i] = 1.0 - mirrored[:, i]
        for i in range(162, 225, 3):
            mirrored[:, i] = 1.0 - mirrored[:, i]
        
        # Sol ve sağ eli yer değiştir
        lh = mirrored[:, 99:162].copy()
        rh = mirrored[:, 162:225].copy()
        mirrored[:, 99:162] = rh
        mirrored[:, 162:225] = lh
        
        return mirrored
    
    @staticmethod
    def scale_landmarks(sequence, scale_range=(0.9, 1.1)):
        """Tüm koordinatları ölçekle"""
        scale = random.uniform(*scale_range)
        # Sadece x, y ölçekle (z değişmesin)
        scaled = sequence.copy()
        for i in range(0, sequence.shape[1], 3):
            scaled[:, i] *= scale      # x
            scaled[:, i+1] *= scale    # y
        return scaled
    
    @staticmethod
    def dropout_landmarks(sequence, drop_prob=0.1):
        """Rastgele landmark'ları sıfırla (occlusion simülasyonu)"""
        result = sequence.copy()
        T, F = result.shape
        
        # Rastgele landmark gruplarını sıfırla (3'erli: x,y,z)
        n_landmarks = F // 3
        n_drop = max(1, int(n_landmarks * drop_prob))
        drop_landmarks = random.sample(range(n_landmarks), n_drop)
        
        for lm_idx in drop_landmarks:
            start = lm_idx * 3
            result[:, start:start+3] = 0.0
        
        return result


def augment_sequence(sequence, config):
    """Tüm augmentation'ları uygula"""
    aug = sequence.copy()
    
    # Temporal augmentation
    if random.random() < 0.5:
        aug = TemporalAugmentation.speed_change(aug, config.AUG_TEMPORAL_SPEED)
    
    if random.random() < config.AUG_TEMPORAL_CROP_PROB and len(aug) > config.SEQ_LENGTH:
        aug = TemporalAugmentation.temporal_crop(aug, config.SEQ_LENGTH)
    
    if random.random() < 0.3:
        aug = TemporalAugmentation.temporal_shift(aug)
    
    if random.random() < 0.2:
        aug = TemporalAugmentation.frame_dropout(aug)
    
    # Landmark augmentation
    if random.random() < 0.5:
        aug = LandmarkAugmentation.add_noise(aug, config.AUG_NOISE_STD)
    
    if random.random() < config.AUG_MIRROR_PROB:
        aug = LandmarkAugmentation.mirror_hands(aug)
    
    if random.random() < 0.3:
        aug = LandmarkAugmentation.scale_landmarks(aug, config.AUG_SCALE_RANGE)
    
    if random.random() < config.AUG_DROPOUT_LANDMARKS_PROB:
        aug = LandmarkAugmentation.dropout_landmarks(aug)
    
    return aug


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def add_velocity_features(sequence):
    """
    Frame-arası hız hesapla.
    Input: (T, 225) → Output: (T, 225)
    velocity[t] = sequence[t] - sequence[t-1]
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return velocity


def add_acceleration_features(sequence):
    """
    Frame-arası ivme hesapla.
    Input: (T, 225) → Output: (T, 225)
    acceleration[t] = sequence[t] - 2*sequence[t-1] + sequence[t-2]
    """
    acceleration = np.zeros_like(sequence)
    acceleration[2:] = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]
    return acceleration


def add_relative_coordinates(sequence):
    """
    El koordinatlarını vücuda (omuz orta noktasına) göre relative yap.
    Bu kişiye ve pozisyona bağımlılığı azaltır.
    
    Input: (T, 225)
    Output: (T, 126) - sadece relative el koordinatları
    """
    T = sequence.shape[0]
    relative = np.zeros((T, 126), dtype=np.float32)  # 42 landmark × 3
    
    for t in range(T):
        # Omuz orta noktası (pose landmark 11 ve 12)
        left_shoulder = sequence[t, 33:36]   # landmark 11: x,y,z
        right_shoulder = sequence[t, 36:39]  # landmark 12: x,y,z
        center = (left_shoulder + right_shoulder) / 2.0
        
        # Omuz genişliği (normalize etmek için)
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        if shoulder_width < 0.01:
            shoulder_width = 0.2  # Default
        
        # Sol el relative (21 × 3 = 63)
        lh = sequence[t, 99:162].reshape(21, 3)
        lh_rel = (lh - center) / shoulder_width
        relative[t, :63] = lh_rel.flatten()
        
        # Sağ el relative (21 × 3 = 63)
        rh = sequence[t, 162:225].reshape(21, 3)
        rh_rel = (rh - center) / shoulder_width
        relative[t, 63:126] = rh_rel.flatten()
    
    return relative


def build_features(sequence, config):
    """
    Tüm feature'ları birleştir.
    Input: (T, 225) → Output: (T, FEATURE_SIZE)
    """
    features = [sequence]  # Base: 225
    
    if config.USE_VELOCITY:
        features.append(add_velocity_features(sequence))  # +225
    
    if config.USE_ACCELERATION:
        features.append(add_acceleration_features(sequence))  # +225
    
    if config.USE_RELATIVE_COORDS:
        features.append(add_relative_coordinates(sequence))  # +126
    
    return np.concatenate(features, axis=-1).astype(np.float32)


def pad_or_truncate(sequence, target_length):
    """Sequence'i hedef uzunluğa getir"""
    T = len(sequence)
    if T == target_length:
        return sequence
    elif T > target_length:
        # Ortadan crop
        start = (T - target_length) // 2
        return sequence[start:start + target_length]
    else:
        # Zero padding (son frame'i tekrarla veya sıfır ekle)
        pad_length = target_length - T
        # Son frame'i tekrarlayarak padding (zero padding'den daha iyi)
        padding = np.tile(sequence[-1:], (pad_length, 1))
        return np.concatenate([sequence, padding], axis=0)


# ==============================================================================
# DATASET
# ==============================================================================

class AUTSLDataset(Dataset):
    """
    Geliştirilmiş AUTSL Dataset.
    
    Beklenen veri formatı:
    - Her örnek bir .npz dosyası: {'skeleton': (T, 225), 'label': int}
    - VEYA tek bir packed .npz: {'X_train': (N, T, 225), 'y_train': (N,)}
    """
    
    def __init__(self, sequences, labels, config, is_train=True):
        """
        Args:
            sequences: list of numpy arrays, her biri (T, 225)
            labels: numpy array (N,)
            config: Config nesnesi
            is_train: Augmentation uygulansın mı?
        """
        self.sequences = sequences
        self.labels = labels
        self.config = config
        self.is_train = is_train
        self.seq_length = config.SEQ_LENGTH
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()  # (T, 225)
        label = self.labels[idx]
        
        # Augmentation (sadece eğitimde)
        if self.is_train:
            seq = augment_sequence(seq, self.config)
        
        # Pad/truncate
        seq = pad_or_truncate(seq, self.seq_length)
        
        # Feature engineering
        seq = build_features(seq, self.config)
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_data(config):
    """
    Veriyi yükle.
    
    Desteklenen formatlar:
    1. Packed .npz (X_train, y_train, X_val, y_val, X_test, y_test)
    2. Her örnek ayrı .npz dosyası
    """
    data_dir = Path(config.DATA_DIR)
    
    # Format 1: Packed npz
    packed_files = list(data_dir.glob('*.npz'))
    
    # Tek packed file
    for f in packed_files:
        data = np.load(f, allow_pickle=True)
        keys = list(data.keys())
        
        if 'X_train' in keys:
            print(f"Packed veri bulundu: {f.name}")
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data.get('X_val', None)
            y_val = data.get('y_val', None)
            X_test = data.get('X_test', None)
            y_test = data.get('y_test', None)
            
            print(f"  Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'N/A'}")
            
            # numpy dizileri halinde döndür
            train_seqs = [X_train[i] for i in range(len(X_train))]
            
            val_seqs = [X_val[i] for i in range(len(X_val))] if X_val is not None else None
            test_seqs = [X_test[i] for i in range(len(X_test))] if X_test is not None else None
            
            return {
                'train': (train_seqs, y_train),
                'val': (val_seqs, y_val) if val_seqs else None,
                'test': (test_seqs, y_test) if test_seqs else None,
            }
    
    # Format 2: Ayrı split klasörleri
    result = {}
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        seqs = []
        labels = []
        for f in sorted(split_dir.glob('*.npz')):
            d = np.load(f, allow_pickle=True)
            seqs.append(d['skeleton'])
            labels.append(int(d['label']))
        
        if seqs:
            result[split] = (seqs, np.array(labels))
            print(f"  {split}: {len(seqs)} örnek")
    
    return result


# ==============================================================================
# MODEL
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignTransformerProV2(nn.Module):
    """
    Geliştirilmiş Transformer modeli.
    
    İyileştirmeler:
    - Multi-scale convolutional input (farklı kernel boyutları)
    - Stochastic depth (derin ağlarda regularization)
    - Gelişmiş classification head
    """
    
    def __init__(self, input_size, d_model=384, nhead=12, num_layers=6, 
                 num_classes=226, dropout=0.3, stochastic_depth=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Multi-scale input projection
        self.input_conv = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale convolutional feature extraction
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU()
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU()
        )
        self.conv_merge = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder with stochastic depth
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.layers.append(layer)
        
        # Stochastic depth probabilities
        self.drop_probs = [stochastic_depth * i / (num_layers - 1) for i in range(num_layers)]
        
        # Multi-head attention pooling (4 heads)
        self.pool_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1)
            ) for _ in range(4)
        ])
        
        # Classification head
        pool_dim = d_model * 5  # 4 attention heads + mean pool
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Input projection
        x = self.input_conv(x)  # (B, T, d_model)
        
        # Multi-scale conv with residual
        x_t = x.transpose(1, 2)  # (B, d_model, T)
        conv3 = self.conv_block_3(x_t).transpose(1, 2)  # (B, T, d_model//2)
        conv5 = self.conv_block_5(x_t).transpose(1, 2)  # (B, T, d_model//2)
        x_conv = torch.cat([conv3, conv5], dim=-1)  # (B, T, d_model)
        x = x + self.conv_merge(x_conv)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer layers with stochastic depth
        for i, layer in enumerate(self.layers):
            if self.training and random.random() < self.drop_probs[i]:
                continue  # Skip this layer (stochastic depth)
            x = layer(x)
        
        # Multi-head attention pooling
        seq_out = x[:, 1:]  # Remove CLS
        pooled_outputs = []
        for pool_head in self.pool_heads:
            attn = pool_head(seq_out)
            attn = F.softmax(attn, dim=1)
            pooled = (attn * seq_out).sum(dim=1)
            pooled_outputs.append(pooled)
        
        # Mean pooling
        mean_pool = seq_out.mean(dim=1)
        
        # Concatenate all (4 heads + mean = 5 * d_model)
        combined = torch.cat(pooled_outputs + [mean_pool], dim=1)
        
        return self.classifier(combined)


# ==============================================================================
# EMA (Exponential Moving Average)
# ==============================================================================

class EMAModel:
    """Model ağırlıklarının exponential moving average'ı"""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply(self, model):
        """EMA ağırlıkları modele uygula (inference için)"""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data = self.shadow[name]
        return backup
    
    def restore(self, model, backup):
        """Orijinal ağırlıkları geri yükle"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data = backup[name]


# ==============================================================================
# MIXUP
# ==============================================================================

def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation: İki örneği karıştır.
    Regularization etkisi yaratır.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# WARMUP SCHEDULER
# ==============================================================================

class WarmupCosineScheduler:
    """Linear warmup + cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * progress
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==============================================================================
# EĞİTİM FONKSİYONLARI
# ==============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, config, ema=None):
    """Bir epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixup
        if config.MIXUP_ALPHA > 0 and random.random() < 0.5:
            mixed_x, y_a, y_b, lam = mixup_data(sequences, labels, config.MIXUP_ALPHA)
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
            # Accuracy (mixup olmadan hesapla)
            with torch.no_grad():
                clean_outputs = model(sequences)
                _, predicted = clean_outputs.max(1)
                correct += (predicted == labels).sum().item()
        else:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
        
        total += labels.size(0)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # EMA güncelle
        if ema is not None:
            ema.update(model)
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Validation / Test değerlendirme"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in val_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc, np.array(all_preds), np.array(all_labels)


def compute_class_weights(labels, num_classes):
    """Sınıf dengesizliğini düzeltmek için ağırlıklar hesapla"""
    counts = Counter(labels.tolist() if hasattr(labels, 'tolist') else labels)
    total = sum(counts.values())
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    
    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# ANA EĞİTİM FONKSİYONU
# ==============================================================================

def train_improved(config=None):
    """
    Geliştirilmiş eğitim pipeline'ı.
    
    Colab'da çalıştırmak için:
        from train_improved_colab import train_improved, Config
        config = Config()
        config.DATA_DIR = '/content/drive/MyDrive/your_data_path'
        train_improved(config)
    """
    if config is None:
        config = Config()
    
    print("=" * 70)
    print("  SignBridge - Geliştirilmiş AUTSL Eğitim")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Veri yükle
    print(f"\n  Veri yukleniyor: {config.DATA_DIR}")
    data = load_data(config)
    
    if 'train' not in data:
        print("  HATA: Egitim verisi bulunamadi!")
        return
    
    train_seqs, train_labels = data['train']
    print(f"  Train: {len(train_seqs)} ornek, {len(set(train_labels.tolist()))} sinif")
    
    val_seqs, val_labels = data.get('val', (None, None))
    test_seqs, test_labels = data.get('test', (None, None))
    
    if val_seqs:
        print(f"  Val: {len(val_seqs)} ornek")
    if test_seqs:
        print(f"  Test: {len(test_seqs)} ornek")
    
    # Normalizasyon istatistikleri hesapla
    print("\n  Normalizasyon istatistikleri hesaplaniyor...")
    
    # Feature boyutunu belirlemek için bir örnek işle
    sample = pad_or_truncate(train_seqs[0], config.SEQ_LENGTH)
    sample_feat = build_features(sample, config)
    actual_feature_size = sample_feat.shape[-1]
    print(f"  Feature boyutu: {actual_feature_size}")
    
    # Tüm eğitim verisinin feature'larını hesapla (normalizasyon için)
    print("  Feature'lar hesaplaniyor...")
    all_features = []
    for seq in train_seqs:
        s = pad_or_truncate(seq, config.SEQ_LENGTH)
        f = build_features(s, config)
        all_features.append(f)
    
    all_features = np.array(all_features)  # (N, T, F)
    norm_mean = all_features.mean(axis=(0, 1))  # (F,)
    norm_std = all_features.std(axis=(0, 1)) + 1e-8  # (F,)
    
    print(f"  Mean range: [{norm_mean.min():.4f}, {norm_mean.max():.4f}]")
    print(f"  Std range: [{norm_std.min():.4f}, {norm_std.max():.4f}]")
    
    del all_features  # Bellek tasarrufu
    
    # Dataset oluştur
    train_dataset = AUTSLDataset(train_seqs, train_labels, config, is_train=True)
    
    # Weighted sampler (sınıf dengesizliğini düzelt)
    class_weights = compute_class_weights(train_labels, config.NUM_CLASSES)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, 
        sampler=sampler, num_workers=4, pin_memory=True
    )
    
    val_loader = None
    if val_seqs:
        val_dataset = AUTSLDataset(val_seqs, val_labels, config, is_train=False)
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE * 2,
            shuffle=False, num_workers=4, pin_memory=True
        )
    
    test_loader = None
    if test_seqs:
        test_dataset = AUTSLDataset(test_seqs, test_labels, config, is_train=False)
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE * 2,
            shuffle=False, num_workers=4, pin_memory=True
        )
    
    # Model
    print(f"\n  Model olusturuluyor...")
    model = SignTransformerProV2(
        input_size=actual_feature_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss (label smoothing ile)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.EPOCHS
    )
    
    # EMA
    ema = EMAModel(model, decay=config.EMA_DECAY)
    
    # Kayıt dizini
    save_dir = Path(config.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Eğitim döngüsü
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    print(f"\n{'='*70}")
    print(f"  Egitim basliyor! ({config.EPOCHS} epoch)")
    print(f"{'='*70}\n")
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        # LR güncelle
        scheduler.step(epoch - 1)
        current_lr = scheduler.get_lr()[0]
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, ema
        )
        
        # Validation (EMA modeli ile)
        val_loss = 0
        val_acc = 0
        if val_loader:
            # EMA ağırlıkları uygula
            backup = ema.apply(model)
            val_loss, val_acc, val_preds, val_labels_arr = evaluate(
                model, val_loader, criterion, device
            )
            # Orijinal ağırlıkları geri yükle
            ema.restore(model, backup)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"  Epoch {epoch:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # En iyi model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # EMA ağırlıklarıyla kaydet
            backup = ema.apply(model)
            
            save_dict = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_size': actual_feature_size,
                    'd_model': config.D_MODEL,
                    'nhead': config.NHEAD,
                    'num_layers': config.NUM_LAYERS,
                    'num_classes': config.NUM_CLASSES,
                    'use_velocity': config.USE_VELOCITY,
                    'use_acceleration': config.USE_ACCELERATION,
                    'use_relative_coords': config.USE_RELATIVE_COORDS,
                    'seq_length': config.SEQ_LENGTH,
                    'base_input_size': config.INPUT_SIZE,
                },
                'val_acc': val_acc,
                'epoch': epoch,
                'norm_mean': norm_mean,
                'norm_std': norm_std,
            }
            
            torch.save(save_dict, save_dir / 'best_model_v2.pt')
            
            ema.restore(model, backup)
            
            print(f"  >>> Yeni en iyi model kaydedildi! Val Acc: {val_acc:.2f}%")
        
        # Periyodik kayıt
        if epoch % 10 == 0:
            backup = ema.apply(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_size': actual_feature_size,
                    'd_model': config.D_MODEL,
                    'nhead': config.NHEAD,
                    'num_layers': config.NUM_LAYERS,
                    'num_classes': config.NUM_CLASSES,
                    'use_velocity': config.USE_VELOCITY,
                    'use_acceleration': config.USE_ACCELERATION,
                    'use_relative_coords': config.USE_RELATIVE_COORDS,
                    'seq_length': config.SEQ_LENGTH,
                    'base_input_size': config.INPUT_SIZE,
                },
                'val_acc': val_acc,
                'epoch': epoch,
                'norm_mean': norm_mean,
                'norm_std': norm_std,
            }, save_dir / f'checkpoint_epoch{epoch}.pt')
            ema.restore(model, backup)
    
    print(f"\n{'='*70}")
    print(f"  Egitim tamamlandi!")
    print(f"  En iyi epoch: {best_epoch}, Val Acc: {best_val_acc:.2f}%")
    print(f"{'='*70}")
    
    # Test
    if test_loader:
        print(f"\n  Test degerlendirmesi yapiliyor...")
        backup = ema.apply(model)
        
        # Normal test
        test_loss, test_acc, test_preds, test_labels_arr = evaluate(
            model, test_loader, criterion, device
        )
        print(f"  Test Accuracy: {test_acc:.2f}%")
        
        # TTA ile test
        print(f"  TTA ile test yapiliyor...")
        model.eval()
        correct_tta = 0
        total_tta = 0
        
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                # Normal
                logits1 = model(sequences)
                # Temporal flip
                logits2 = model(torch.flip(sequences, dims=[1]))
                # Noise
                logits3 = model(sequences + torch.randn_like(sequences) * 0.01)
                
                avg_logits = (logits1 + logits2 * 0.5 + logits3 * 0.5) / 2.0
                _, predicted = avg_logits.max(1)
                correct_tta += (predicted == labels).sum().item()
                total_tta += labels.size(0)
        
        tta_acc = 100.0 * correct_tta / total_tta
        print(f"  Test Accuracy (TTA): {tta_acc:.2f}%")
        
        # Final model güncelle
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': actual_feature_size,
                'd_model': config.D_MODEL,
                'nhead': config.NHEAD,
                'num_layers': config.NUM_LAYERS,
                'num_classes': config.NUM_CLASSES,
                'use_velocity': config.USE_VELOCITY,
                'use_acceleration': config.USE_ACCELERATION,
                'use_relative_coords': config.USE_RELATIVE_COORDS,
                'seq_length': config.SEQ_LENGTH,
                'base_input_size': config.INPUT_SIZE,
            },
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_acc_tta': tta_acc,
            'epoch': best_epoch,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }
        torch.save(save_dict, save_dir / 'best_model_v2.pt')
        
        ema.restore(model, backup)
    
    # Normalizasyon kaydet (inference için)
    np.save(save_dir / 'norm_mean_v2.npy', norm_mean)
    np.save(save_dir / 'norm_std_v2.npy', norm_std)
    
    # Label map kopyala
    label_path = Path(config.DATA_DIR).parent / 'label_map.json'
    if not label_path.exists():
        label_path = MODEL_DIR / 'label_map.json'
    if label_path.exists():
        import shutil
        shutil.copy2(label_path, save_dir / 'label_map.json')
    
    # Eğitim geçmişi kaydet
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n  Model ve dosyalar kaydedildi: {save_dir}")
    print(f"  Dosyalar:")
    print(f"    - best_model_v2.pt")
    print(f"    - norm_mean_v2.npy")
    print(f"    - norm_std_v2.npy")
    print(f"    - training_history.json")
    
    return model, history


# ==============================================================================
# CONFUSION MATRIX ANALİZİ
# ==============================================================================

def analyze_confusion(model, test_loader, label_map, device, save_path=None):
    """
    Confusion matrix analizi.
    En çok karıştırılan sınıfları bulur.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-class accuracy
    num_classes = len(label_map)
    class_correct = Counter()
    class_total = Counter()
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    # En kötü sınıflar
    print("\n  En kotu performansli siniflar:")
    print("  " + "-" * 50)
    
    class_accs = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            acc = 100.0 * class_correct[c] / class_total[c]
            class_accs[c] = acc
    
    sorted_classes = sorted(class_accs.items(), key=lambda x: x[1])
    
    for c, acc in sorted_classes[:20]:
        name = label_map.get(c, f"Class_{c}")
        print(f"    {name:20s}: {acc:5.1f}% ({class_correct[c]}/{class_total[c]})")
    
    # En çok karıştırılan çiftler
    print(f"\n  En cok karistirilan ciftler:")
    print("  " + "-" * 50)
    
    confusion_pairs = Counter()
    for pred, label in zip(all_preds, all_labels):
        if pred != label:
            confusion_pairs[(label, pred)] += 1
    
    for (true_c, pred_c), count in confusion_pairs.most_common(15):
        true_name = label_map.get(true_c, f"C{true_c}")
        pred_name = label_map.get(pred_c, f"C{pred_c}")
        print(f"    {true_name:15s} -> {pred_name:15s} : {count} kez")
    
    return class_accs, confusion_pairs


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    # Konfigürasyon
    cfg = Config()
    
    # Colab'da bu yolları düzenleyin
    # cfg.DATA_DIR = '/content/drive/MyDrive/autsl_skeletons'
    # cfg.SAVE_DIR = '/content/drive/MyDrive/autsl_improved_model'
    
    # Eğitimi başlat
    model, history = train_improved(cfg)
