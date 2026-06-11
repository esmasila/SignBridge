#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SignBridge - AUTSL 226 Sınıf İşaret Dili Modeli - TAM EĞİTİM SCRIPTİ
================================================================================
 Google Colab'da çalıştırılmak üzere hazırlanmıştır.
 
 Bu script TEK DOSYA olarak çalışır. Sadece Colab'a yükle ve çalıştır.
 
 SENİN DRIVE YAPIN:
   AUTSL_Proje/
     Koordinatlar/
       all_train_packed.npz  (361 MB - iskelet verileri)
       all_val_packed.npz    (63 MB)
       all_test_packed.npz   (53 MB)
     Models_v3_80plus/
       best_model.pt         (49 MB - mevcut en iyi model)
       norm_mean.npy
       norm_std.npy
       label_map.json
     SignList_ClassId_TR_EN (1).csv
 
 İçerik:
   BÖLÜM 0: Kurulum (pip install, Drive mount)
   BÖLÜM 1: Veri Yükleme (mevcut iskelet .npz dosyalarından)
   BÖLÜM 2: Data Augmentation
   BÖLÜM 3: Feature Engineering + Dataset
   BÖLÜM 4: Model Tanımı (SignTransformerPro - mevcut modelle uyumlu)
   BÖLÜM 5: Eğitim (checkpoint/resume destekli)
   BÖLÜM 6: Test ve Analiz
   BÖLÜM 7: Modeli İndirmeye Hazırlama

 KULLANIM:
   1. Bu dosyayı Colab'a yükle (veya içeriğini bir hücreye yapıştır)
   2. Runtime > Change runtime type > GPU (T4)
   3. Runtime > Run all
   4. Drive erişimi ver
   5. Bekle (eğitim ~1-2 saat)
   6. Modeli Drive'dan indir

 CHECKPOINT DESTEĞİ:
   - Eğitim her 5 epoch'ta otomatik Drive'a kaydeder
   - Colab bağlantısı koparsa, tekrar çalıştırınca kaldığı yerden devam eder
================================================================================
"""

# ==============================================================================
# BÖLÜM 0: KURULUM
# ==============================================================================

import subprocess
import sys
import os

def install_packages():
    """Gerekli paketleri kur"""
    packages = ['torch', 'torchvision', 'numpy', 'tqdm']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_').split('==')[0])
        except ImportError:
            print(f"📦 {pkg} kuruluyor...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    print("✅ Tüm paketler hazır!")

install_packages()

# Drive bağlantısı
def mount_drive():
    """Google Drive'ı bağla"""
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
            print("✅ Google Drive bağlandı!")
        else:
            print("✅ Google Drive zaten bağlı!")
        return True
    except ImportError:
        print("⚠️ Colab dışında çalışıyorsunuz. Drive mount atlanıyor.")
        return False
    except Exception as e:
        print(f"❌ Drive mount hatası: {e}")
        return False

IS_COLAB = mount_drive()


# ==============================================================================
# KONFİGÜRASYON - SENİN DRIVE YAPINA GÖRE AYARLANDI
# ==============================================================================

class CONFIG:
    """
    ÖNEMLİ: Yollar senin Drive yapına göre ayarlandı.
    Sadece AUTSL_Proje klasör adın farklıysa değiştir.
    """
    
    # ==================== DRIVE YOLLARI ====================
    # Ana proje klasörü
    PROJE_DIR = '/content/drive/MyDrive/AUTSL_Proje'
    
    # İskelet verileri (ZATEN ÇIKARILMIŞ - hazır npz dosyaları)
    KOORDINAT_DIR = f'{PROJE_DIR}/Koordinatlar'
    
    # Mevcut en iyi model (ağırlık transferi için)
    MEVCUT_MODEL_DIR = f'{PROJE_DIR}/Models_v3_80plus'
    
    # Sınıf listesi
    CLASS_LIST_CSV = f'{PROJE_DIR}/SignList_ClassId_TR_EN (1).csv'
    
    # Yeni model kayıt yeri
    SAVE_DIR = f'{PROJE_DIR}/Models_v4_yeni_egitim'
    
    # ==================== VERİ DOSYALARI ====================
    # Senin Koordinatlar klasöründeki packed npz dosyaları
    TRAIN_NPZ = f'{KOORDINAT_DIR}/all_train_packed.npz'
    VAL_NPZ = f'{KOORDINAT_DIR}/all_val_packed.npz'
    TEST_NPZ = f'{KOORDINAT_DIR}/all_test_packed.npz'
    
    # ==================== MODEL ====================
    INPUT_SIZE = 225          # 75 landmark × 3 koordinat (Pose+LH+RH)
    D_MODEL = 384             # Transformer hidden boyutu
    NHEAD = 12                # Attention head sayısı
    NUM_LAYERS = 6            # Transformer katman sayısı
    NUM_CLASSES = 226         # AUTSL sınıf sayısı (0-225)
    DROPOUT = 0.3
    SEQ_LENGTH = 30           # Frame sayısı
    
    # ==================== EĞİTİM ====================
    EPOCHS = 60
    BATCH_SIZE = 64
    LR = 5e-5                # Fine-tuning için düşük LR (1e-3 çok yüksek!)
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 3
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2         # Mixup augmentation
    GRADIENT_CLIP = 1.0
    EMA_DECAY = 0.999
    
    # ==================== AUGMENTATION ====================
    # İlk eğitimde velocity/acceleration KAPALI (mevcut modelle uyumlu)
    USE_VELOCITY = False
    USE_ACCELERATION = False
    USE_RELATIVE_COORDS = False
    
    AUG_NOISE_STD = 0.01
    AUG_MIRROR_PROB = 0.3
    AUG_SPEED_RANGE = (0.8, 1.2)
    AUG_SCALE_RANGE = (0.9, 1.1)
    AUG_DROPOUT_PROB = 0.1
    
    # ==================== CHECKPOINT ====================
    CHECKPOINT_EVERY = 5      # Her 5 epoch'ta kaydet
    RESUME_TRAINING = True    # True = kaldığı yerden devam et
    
    # ==================== PRETRAINED ====================
    # Mevcut modelden başla (transfer learning)
    USE_PRETRAINED = True     # True = mevcut ağırlıklardan başla

cfg = CONFIG()


# ==============================================================================
# BÖLÜM 1: VERİ YÜKLEME (MEVCUT İSKELET DOSYALARINDAN)
# ==============================================================================

import numpy as np
import json
import csv
import math
import time
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def load_class_names(csv_path):
    """
    SignList_ClassId_TR_EN CSV dosyasını oku.
    Format: ClassId,TR,EN
    """
    class_names = {}
    if not os.path.exists(csv_path):
        print(f"  ⚠️ Sınıf listesi bulunamadı: {csv_path}")
        return class_names
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    cid = int(row[0])
                    name = row[1].strip()
                    class_names[cid] = name
                except (ValueError, IndexError):
                    continue
    
    print(f"  📋 {len(class_names)} sınıf ismi yüklendi")
    return class_names


def load_skeleton_data():
    """
    Mevcut iskelet verilerini yükle.
    
    Senin dosyaların:
      Koordinatlar/all_train_packed.npz  (keys: 'x', 'y')
      Koordinatlar/all_val_packed.npz
      Koordinatlar/all_test_packed.npz
    """
    print("\n" + "="*70)
    print("  BÖLÜM 1: VERİ YÜKLEME")
    print("="*70)
    
    data = {}
    
    files = {
        'train': cfg.TRAIN_NPZ,
        'val': cfg.VAL_NPZ,
        'test': cfg.TEST_NPZ,
    }
    
    for split, fpath in files.items():
        if not os.path.exists(fpath):
            print(f"  ❌ {split} dosyası bulunamadı: {fpath}")
            continue
        
        d = np.load(fpath)
        keys = list(d.keys())
        
        # Key isimlerini tespit et (x/y veya X_train/y_train olabilir)
        if 'x' in keys:
            X = d['x'].astype(np.float32)
            y = d['y'].astype(np.int64)
        elif f'X_{split}' in keys:
            X = d[f'X_{split}'].astype(np.float32)
            y = d[f'y_{split}'].astype(np.int64)
        elif 'X_train' in keys:
            X = d['X_train'].astype(np.float32)
            y = d['y_train'].astype(np.int64)
        else:
            # İlk iki key'i dene
            X = d[keys[0]].astype(np.float32)
            y = d[keys[1]].astype(np.int64)
        
        data[split] = (X, y)
        n_classes = len(set(y.tolist()))
        print(f"  ✅ {split:5s}: {X.shape[0]:6d} örnek | shape={X.shape} | {n_classes} sınıf | keys={keys}")
    
    if not data:
        print("\n  ❌ Hiç veri yüklenemedi!")
        print(f"     Kontrol edin: {cfg.KOORDINAT_DIR}")
        return None
    
    return data


# ==============================================================================
# BÖLÜM 2: DATA AUGMENTATION
# ==============================================================================

class TemporalAugmentation:
    @staticmethod
    def speed_change(sequence, speed_range=(0.8, 1.2)):
        """Rastgele hız değişimi (yavaşlat/hızlandır)"""
        T, F = sequence.shape
        speed = random.uniform(*speed_range)
        new_T = max(10, int(T * speed))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        indices = np.clip(indices, 0, T - 1)
        return sequence[indices]
    
    @staticmethod
    def temporal_shift(sequence, max_shift=3):
        """Zaman ekseninde küçük kaydırma"""
        return np.roll(sequence, random.randint(-max_shift, max_shift), axis=0)
    
    @staticmethod
    def frame_dropout(sequence, drop_rate=0.1):
        """Rastgele frame çıkar, interpolasyon yap"""
        T, F = sequence.shape
        result = sequence.copy()
        n_drop = max(1, int(T * drop_rate))
        for idx in random.sample(range(T), n_drop):
            prev_idx = max(0, idx - 1)
            next_idx = min(T - 1, idx + 1)
            result[idx] = (sequence[prev_idx] + sequence[next_idx]) / 2
        return result


class LandmarkAugmentation:
    @staticmethod
    def add_noise(sequence, noise_std=0.01):
        """Gaussian noise ekle"""
        return sequence + np.random.randn(*sequence.shape).astype(np.float32) * noise_std
    
    @staticmethod
    def mirror_hands(sequence):
        """Sol-sağ el aynala (x koordinatlarını flip + el yer değiştir)"""
        mirrored = sequence.copy()
        # X koordinatlarını çevir (1 - x)
        for i in range(0, 225, 3):
            mirrored[:, i] = 1.0 - mirrored[:, i]
        # Sol ve sağ eli yer değiştir
        lh = mirrored[:, 99:162].copy()
        rh = mirrored[:, 162:225].copy()
        mirrored[:, 99:162] = rh
        mirrored[:, 162:225] = lh
        return mirrored
    
    @staticmethod
    def scale_landmarks(sequence, scale_range=(0.9, 1.1)):
        """Tüm x,y koordinatlarını ölçekle"""
        scale = random.uniform(*scale_range)
        scaled = sequence.copy()
        for i in range(0, sequence.shape[1], 3):
            scaled[:, i] *= scale      # x
            scaled[:, i+1] *= scale    # y
        return scaled
    
    @staticmethod
    def dropout_landmarks(sequence, drop_prob=0.1):
        """Rastgele landmark'ları sıfırla (occlusion simülasyonu)"""
        result = sequence.copy()
        n_landmarks = result.shape[1] // 3
        n_drop = max(1, int(n_landmarks * drop_prob))
        for lm_idx in random.sample(range(n_landmarks), n_drop):
            result[:, lm_idx*3:lm_idx*3+3] = 0.0
        return result


def augment_sequence(seq, cfg):
    """Tüm augmentation'ları rastgele uygula"""
    aug = seq.copy()
    
    if random.random() < 0.5:
        aug = TemporalAugmentation.speed_change(aug, cfg.AUG_SPEED_RANGE)
    if random.random() < 0.3:
        aug = TemporalAugmentation.temporal_shift(aug)
    if random.random() < 0.2:
        aug = TemporalAugmentation.frame_dropout(aug)
    if random.random() < 0.5:
        aug = LandmarkAugmentation.add_noise(aug, cfg.AUG_NOISE_STD)
    if random.random() < cfg.AUG_MIRROR_PROB:
        aug = LandmarkAugmentation.mirror_hands(aug)
    if random.random() < 0.3:
        aug = LandmarkAugmentation.scale_landmarks(aug, cfg.AUG_SCALE_RANGE)
    if random.random() < cfg.AUG_DROPOUT_PROB:
        aug = LandmarkAugmentation.dropout_landmarks(aug)
    
    return aug


# ==============================================================================
# BÖLÜM 3: FEATURE ENGINEERING + DATASET
# ==============================================================================

def add_velocity(sequence):
    """Frame-arası hız: v[t] = x[t] - x[t-1]"""
    v = np.zeros_like(sequence)
    v[1:] = sequence[1:] - sequence[:-1]
    return v

def add_acceleration(sequence):
    """Frame-arası ivme"""
    a = np.zeros_like(sequence)
    a[2:] = sequence[2:] - 2*sequence[1:-1] + sequence[:-2]
    return a

def add_relative_coords(sequence):
    """El koordinatlarını omuz merkezine göre normalize et"""
    T = sequence.shape[0]
    relative = np.zeros((T, 126), dtype=np.float32)
    for t in range(T):
        ls = sequence[t, 33:36]   # sol omuz
        rs = sequence[t, 36:39]   # sağ omuz
        center = (ls + rs) / 2.0
        sw = max(0.01, np.linalg.norm(ls[:2] - rs[:2]))
        lh = sequence[t, 99:162].reshape(21, 3)
        rh = sequence[t, 162:225].reshape(21, 3)
        relative[t, :63] = ((lh - center) / sw).flatten()
        relative[t, 63:126] = ((rh - center) / sw).flatten()
    return relative

def build_features(sequence, cfg):
    """Tüm feature'ları birleştir"""
    features = [sequence]  # 225
    if cfg.USE_VELOCITY:
        features.append(add_velocity(sequence))       # +225
    if cfg.USE_ACCELERATION:
        features.append(add_acceleration(sequence))   # +225
    if cfg.USE_RELATIVE_COORDS:
        features.append(add_relative_coords(sequence)) # +126
    return np.concatenate(features, axis=-1).astype(np.float32)

def pad_or_truncate(sequence, target_length):
    """Sequence'i hedef uzunluğa getir"""
    T = len(sequence)
    if T == target_length:
        return sequence
    elif T > target_length:
        start = (T - target_length) // 2
        return sequence[start:start + target_length]
    else:
        pad = np.tile(sequence[-1:], (target_length - T, 1))
        return np.concatenate([sequence, pad], axis=0)

def compute_feature_size(cfg):
    """Feature boyutunu hesapla"""
    size = cfg.INPUT_SIZE  # 225
    if cfg.USE_VELOCITY: size += cfg.INPUT_SIZE
    if cfg.USE_ACCELERATION: size += cfg.INPUT_SIZE
    if cfg.USE_RELATIVE_COORDS: size += 126
    return size


class AUTSLDataset(Dataset):
    def __init__(self, X, y, cfg, is_train=True):
        self.X = X
        self.y = y
        self.cfg = cfg
        self.is_train = is_train
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        seq = self.X[idx].copy()  # (T, 225)
        label = self.y[idx]
        
        # Augmentation (sadece eğitimde)
        if self.is_train:
            seq = augment_sequence(seq, self.cfg)
        
        # Pad/truncate
        seq = pad_or_truncate(seq, self.cfg.SEQ_LENGTH)
        
        # Feature engineering
        seq = build_features(seq, self.cfg)
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ==============================================================================
# BÖLÜM 4: MODEL TANIMI (SignTransformerPro - MEVCUT MODELLE UYUMLU)
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


class SignTransformerPro(nn.Module):
    """
    MEVCUT modelle BİREBİR UYUMLU mimari.
    Models_v3_80plus/best_model.pt ağırlıklarını yükleyebilir.
    """
    def __init__(self, input_size, d_model=384, nhead=12, num_layers=6, 
                 num_classes=226, dropout=0.35):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//4), nn.Tanh(), nn.Linear(d_model//4, 1))
            for _ in range(4)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model*5), nn.Dropout(dropout),
            nn.Linear(d_model*5, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(d_model, num_classes))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.input_conv(x)
        x = x + self.conv_block(x.transpose(1,2)).transpose(1,2)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1)
        x = self.transformer(self.pos_encoder(x))
        seq = x[:, 1:]
        pooled = [F.softmax(h(seq), dim=1) * seq for h in self.pool_heads]
        pooled = [p.sum(dim=1) for p in pooled]
        return self.classifier(torch.cat(pooled + [seq.mean(dim=1)], dim=1))


# ==============================================================================
# BÖLÜM 5: EĞİTİM YARDIMCI SINIFLAR
# ==============================================================================

class EMAModel:
    """Exponential Moving Average — eğitim sırasında model ağırlıklarını düzgünleştirir"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply(self, model):
        """EMA ağırlıkları modele uygula (val/test için)"""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data = self.shadow[name]
        return backup
    
    def restore(self, model, backup):
        """Orijinal ağırlıkları geri yükle (eğitime devam için)"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data = backup[name]
    
    def state_dict(self):
        return dict(self.shadow)
    
    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class WarmupCosineScheduler:
    """İlk N epoch linear warmup, sonra cosine decay"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * progress
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def mixup_data(x, y, alpha=0.2):
    """Mixup: iki örneği karıştırarak regularization sağlar"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ==============================================================================
# BÖLÜM 5b: ANA EĞİTİM DÖNGÜSÜ (CHECKPOINT DESTEKLİ)
# ==============================================================================

def train_model(data, cfg):
    """
    Ana eğitim fonksiyonu.
    
    CHECKPOINT DESTEĞİ:
    - Her 5 epoch'ta Drive'a kaydeder
    - Colab düşerse, tekrar çalıştırınca kaldığı epoch'tan devam eder
    - Hem optimizer hem EMA state'i korunur
    
    PRETRAINED DESTEK:
    - Models_v3_80plus/best_model.pt'den ağırlık yükleyerek başlar
    - Sıfırdan eğitmekten çok daha hızlı yakınsar
    """
    print("\n" + "="*70)
    print("  BÖLÜM 5: MODEL EĞİTİMİ")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ GPU bulunamadı! Eğitim çok yavaş olacak.")
        print("     Runtime > Change runtime type > GPU seçin!")
    
    # ==================== VERİ ====================
    X_train, y_train = data['train']
    X_val, y_val = data.get('val', (None, None))
    X_test, y_test = data.get('test', (None, None))
    
    print(f"\n  📊 Veri:")
    print(f"     Train: {len(y_train)} örnek")
    if y_val is not None: print(f"     Val:   {len(y_val)} örnek")
    if y_test is not None: print(f"     Test:  {len(y_test)} örnek")
    
    # Feature boyutu
    feature_size = compute_feature_size(cfg)
    print(f"     Feature boyutu: {feature_size}")
    
    # ==================== NORMALİZASYON ====================
    print("\n  📐 Normalizasyon istatistikleri hesaplanıyor...")
    
    # Tüm train verisinden mean/std hesapla
    all_feats = []
    sample_count = min(len(X_train), 5000)  # Bellek tasarrufu: max 5000 örnekten hesapla
    sample_indices = np.random.choice(len(X_train), sample_count, replace=False)
    
    for i in sample_indices:
        seq = pad_or_truncate(X_train[i], cfg.SEQ_LENGTH)
        feat = build_features(seq, cfg)
        all_feats.append(feat)
    all_feats = np.array(all_feats)
    
    norm_mean = all_feats.mean(axis=(0, 1))  # (F,)
    norm_std = all_feats.std(axis=(0, 1)) + 1e-8  # (F,) — ÖNEMLİ: 1e-8, 0.1 DEĞİL!
    
    print(f"     Mean range: [{norm_mean.min():.6f}, {norm_mean.max():.6f}]")
    print(f"     Std range:  [{norm_std.min():.6f}, {norm_std.max():.6f}]")
    print(f"     Düşük std feature sayısı (< 0.1): {(norm_std < 0.1).sum()}")
    
    del all_feats
    
    # ==================== DATASET ====================
    train_dataset = AUTSLDataset(X_train, y_train, cfg, is_train=True)
    
    # Weighted sampler — az örnekli sınıfları daha sık göster
    class_counts = Counter(y_train.tolist())
    total_samples = len(y_train)
    class_weights = {c: total_samples / (cfg.NUM_CLASSES * count) 
                     for c, count in class_counts.items()}
    sample_weights = [class_weights[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    val_loader = None
    if X_val is not None:
        val_dataset = AUTSLDataset(X_val, y_val, cfg, is_train=False)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False, num_workers=2, pin_memory=True)
    
    test_loader = None
    if X_test is not None:
        test_dataset = AUTSLDataset(X_test, y_test, cfg, is_train=False)
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False, num_workers=2, pin_memory=True)
    
    # ==================== MODEL ====================
    model = SignTransformerPro(
        input_size=feature_size,
        d_model=cfg.D_MODEL,
        nhead=cfg.NHEAD,
        num_layers=cfg.NUM_LAYERS,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  🧠 Model: {total_params:,} parametre")
    
    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, cfg.WARMUP_EPOCHS, cfg.EPOCHS)
    ema = EMAModel(model, decay=cfg.EMA_DECAY)
    
    # ==================== CHECKPOINT / PRETRAINED ====================
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 1
    best_val_acc = 0.0
    history = []
    
    checkpoint_path = save_dir / 'training_checkpoint.pt'
    
    # Öncelik 1: Devam checkpoint'u (yarım kalmış eğitim)
    checkpoint_loaded = False
    if cfg.RESUME_TRAINING and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_best = ckpt.get('best_val_acc', 0)
        if ckpt_best < 50.0:
            print(f"\n  ⚠️ Checkpoint val_acc={ckpt_best:.1f}% çok düşük, SİLİNİYOR...")
            print(f"     Pretrained modelden tekrar başlanacak.")
            os.remove(checkpoint_path)
            if (save_dir / 'best_model.pt').exists():
                os.remove(save_dir / 'best_model.pt')
            del ckpt
        else:
            print(f"\n  📌 CHECKPOINT BULUNDU! Yarım kalan eğitimden devam ediliyor...")
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_acc = ckpt.get('best_val_acc', 0.0)
            history = ckpt.get('history', [])
            if 'ema_state_dict' in ckpt:
                ema.load_state_dict(ckpt['ema_state_dict'])
            if 'norm_mean' in ckpt:
                norm_mean = ckpt['norm_mean']
                norm_std = ckpt['norm_std']
            print(f"  ✅ Epoch {start_epoch - 1}'den devam! (best_val_acc={best_val_acc:.2f}%)")
            checkpoint_loaded = True
    
    # Öncelik 2: Pretrained model (sıfırdan değil, mevcut ağırlıklardan başla)
    if not checkpoint_loaded and cfg.USE_PRETRAINED:
        pretrained_path = os.path.join(cfg.MEVCUT_MODEL_DIR, 'best_model.pt')
        if os.path.exists(pretrained_path):
            print(f"\n  📌 PRETRAINED MODEL yükleniyor: {pretrained_path}")
            pretrained = torch.load(pretrained_path, map_location=device, weights_only=False)
            
            pretrained_state = pretrained['model_state_dict']
            model_state = model.state_dict()
            
            loaded = 0
            skipped = 0
            for name, param in pretrained_state.items():
                if name in model_state and param.shape == model_state[name].shape:
                    model_state[name] = param
                    loaded += 1
                else:
                    skipped += 1
                    print(f"     ⚠️ Atlandı: {name} (shape uyumsuz)")
            
            model.load_state_dict(model_state)
            ema = EMAModel(model, decay=cfg.EMA_DECAY)  # EMA'yı yeni ağırlıklarla yeniden oluştur
            
            print(f"  ✅ {loaded} katman yüklendi, {skipped} atlandı")
            print(f"     Val={pretrained.get('val_acc', '?')}, Test={pretrained.get('test_acc', '?')}")
        else:
            print(f"\n  ⚠️ Pretrained model bulunamadı: {pretrained_path}")
            print(f"     Sıfırdan eğitim başlayacak.")
    
    # ==================== EĞİTİM DÖNGÜSÜ ====================
    print(f"\n{'='*70}")
    print(f"  🚀 Eğitim başlıyor! Epoch {start_epoch} → {cfg.EPOCHS}")
    print(f"     Checkpoint her {cfg.CHECKPOINT_EVERY} epoch'ta Drive'a kaydedilir")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        epoch_start = time.time()
        
        # LR güncelle
        scheduler.step(epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ========== TRAIN ==========
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{cfg.EPOCHS}', leave=False)
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixup (rastgele %50 olasılıkla)
            if cfg.MIXUP_ALPHA > 0 and random.random() < 0.5:
                mixed_x, y_a, y_b, lam = mixup_data(sequences, labels, cfg.MIXUP_ALPHA)
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                
                # Accuracy: mixup olmadan hesapla
                with torch.no_grad():
                    _, predicted = model(sequences).max(1)
                    correct += (predicted == labels).sum().item()
            else:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
            
            total += labels.size(0)
            
            loss.backward()
            if cfg.GRADIENT_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            optimizer.step()
            ema.update(model)
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # ========== VALIDATION (EMA ağırlıkları ile) ==========
        val_loss = 0.0
        val_acc = 0.0
        if val_loader:
            backup = ema.apply(model)
            model.eval()
            v_loss = 0.0; v_correct = 0; v_total = 0
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    outputs = model(sequences)
                    v_loss += criterion(outputs, labels).item()
                    _, predicted = outputs.max(1)
                    v_correct += (predicted == labels).sum().item()
                    v_total += labels.size(0)
            
            val_loss = v_loss / len(val_loader)
            val_acc = 100.0 * v_correct / v_total
            ema.restore(model, backup)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"  Epoch {epoch:3d}/{cfg.EPOCHS} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | {epoch_time:.1f}s"
              f"{' ⭐' if val_acc > best_val_acc else ''}")
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'lr': current_lr
        })
        
        # ========== EN İYİ MODEL KAYDET ==========
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            backup = ema.apply(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_size': feature_size,
                    'd_model': cfg.D_MODEL, 'nhead': cfg.NHEAD,
                    'num_layers': cfg.NUM_LAYERS, 'num_classes': cfg.NUM_CLASSES,
                },
                'val_acc': val_acc, 'epoch': epoch,
                'norm_mean': norm_mean, 'norm_std': norm_std,
                'model_type': 'ema',
            }, save_dir / 'best_model.pt')
            ema.restore(model, backup)
            print(f"  >>> ✅ YENİ EN İYİ MODEL! Val Acc: {val_acc:.2f}%")
        
        # ========== CHECKPOINT KAYDET (her N epoch) ==========
        if epoch % cfg.CHECKPOINT_EVERY == 0:
            print(f"  💾 Checkpoint kaydediliyor (epoch {epoch})...")
            backup = ema.apply(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'history': history,
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'config': {
                    'input_size': feature_size,
                    'd_model': cfg.D_MODEL, 'nhead': cfg.NHEAD,
                    'num_layers': cfg.NUM_LAYERS, 'num_classes': cfg.NUM_CLASSES,
                },
            }, checkpoint_path)
            ema.restore(model, backup)
            print(f"  ✅ Checkpoint kaydedildi → {checkpoint_path}")
    
    # ==================== EĞİTİM BİTTİ ====================
    print(f"\n{'='*70}")
    print(f"  ✅ Eğitim tamamlandı!")
    print(f"  🏆 En iyi Val Acc: {best_val_acc:.2f}%")
    print(f"{'='*70}")
    
    # ========== TEST ==========
    if test_loader:
        print(f"\n  🎯 Test değerlendirmesi yapılıyor...")
        backup = ema.apply(model)
        model.eval()
        t_correct = 0; t_total = 0
        
        with torch.no_grad():
            for sequences, labels in tqdm(test_loader, desc='Test'):
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                _, predicted = outputs.max(1)
                t_correct += (predicted == labels).sum().item()
                t_total += labels.size(0)
        
        test_acc = 100.0 * t_correct / t_total
        print(f"  🎯 Test Accuracy: {test_acc:.2f}%")
        
        # Final best_model güncelle (test acc ekle)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': feature_size,
                'd_model': cfg.D_MODEL, 'nhead': cfg.NHEAD,
                'num_layers': cfg.NUM_LAYERS, 'num_classes': cfg.NUM_CLASSES,
            },
            'val_acc': best_val_acc, 'test_acc': test_acc,
            'norm_mean': norm_mean, 'norm_std': norm_std,
            'model_type': 'ema',
        }, save_dir / 'best_model.pt')
        
        ema.restore(model, backup)
    
    # ========== DOSYALARI KAYDET ==========
    # norm_mean ve norm_std (inference'ta kullanılacak)
    np.save(save_dir / 'norm_mean.npy', norm_mean.reshape(1, 1, -1))
    np.save(save_dir / 'norm_std.npy', norm_std.reshape(1, 1, -1))
    
    # Eğitim geçmişi
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n  📁 Kaydedilen dosyalar ({save_dir}):")
    print(f"     best_model.pt          → En iyi model")
    print(f"     norm_mean.npy          → Normalizasyon mean")
    print(f"     norm_std.npy           → Normalizasyon std")
    print(f"     training_history.json  → Eğitim geçmişi")
    print(f"     training_checkpoint.pt → Devam checkpoint'u")
    
    return model, history, norm_mean, norm_std


# ==============================================================================
# BÖLÜM 6: TEST VE ANALİZ
# ==============================================================================

def analyze_results(data, cfg, class_names=None):
    """Eğitim sonrası detaylı analiz"""
    print("\n" + "="*70)
    print("  BÖLÜM 6: SONUÇ ANALİZİ")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg.SAVE_DIR)
    model_path = save_dir / 'best_model.pt'
    
    if not model_path.exists():
        print("  ❌ best_model.pt bulunamadı, analiz atlanıyor")
        return
    
    # Model yükle
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    
    model = SignTransformerPro(
        input_size=model_cfg['input_size'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        num_classes=model_cfg['num_classes'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    norm_mean = ckpt['norm_mean']
    norm_std = ckpt['norm_std']
    
    print(f"  Model: Val={ckpt.get('val_acc', 0):.2f}%, Test={ckpt.get('test_acc', '?')}")
    
    # Test verisi
    if 'test' not in data:
        print("  ⚠️ Test verisi yok")
        return
    
    X_test, y_test = data['test']
    
    all_preds = []
    all_labels = []
    all_confs = []
    
    for i in range(0, len(X_test), 128):
        bx = X_test[i:i+128]
        by = y_test[i:i+128]
        
        batch_feats = []
        for j in range(len(bx)):
            seq = pad_or_truncate(bx[j], cfg.SEQ_LENGTH)
            feat = build_features(seq, cfg)
            batch_feats.append(feat)
        batch_feats = np.array(batch_feats)
        bn = (batch_feats - norm_mean) / norm_std
        
        with torch.no_grad():
            out = model(torch.tensor(bn, dtype=torch.float32).to(device))
            probs = F.softmax(out, dim=1)
            conf, pred = probs.max(1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(by)
        all_confs.extend(conf.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)
    
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\n  🎯 Genel Doğruluk: {overall_acc:.2f}%")
    
    # Confidence analizi
    correct_mask = all_preds == all_labels
    print(f"\n  📊 Confidence Analizi:")
    print(f"     Doğru tahmin ort. conf: {all_confs[correct_mask].mean():.3f}")
    print(f"     Yanlış tahmin ort. conf: {all_confs[~correct_mask].mean():.3f}")
    
    for thr in [0.3, 0.5, 0.7, 0.9]:
        mask = all_confs >= thr
        if mask.sum() > 0:
            acc = correct_mask[mask].mean() * 100
            cover = mask.mean() * 100
            print(f"     conf >= {thr}: doğruluk={acc:.1f}%, kapsam={cover:.1f}%")
    
    # En kötü sınıflar
    class_correct = Counter()
    class_total = Counter()
    confusions = Counter()
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
        else:
            confusions[(label, pred)] += 1
    
    print(f"\n  📉 En Kötü 15 Sınıf:")
    accs = [(c, 100*class_correct[c]/class_total[c], class_correct[c], class_total[c]) 
            for c in range(cfg.NUM_CLASSES) if class_total[c] > 0]
    accs.sort(key=lambda x: x[1])
    
    for c, a, cr, tot in accs[:15]:
        name = class_names.get(c, f"Sinif_{c}") if class_names else f"Sinif_{c}"
        print(f"     [{c:3d}] {name:25s}: {a:5.1f}% ({cr}/{tot})")
    
    # Doğruluk dağılımı
    acc_vals = [a for _, a, _, _ in accs]
    print(f"\n  📊 Sınıf Doğruluk Dağılımı:")
    print(f"     0%:      {sum(1 for a in acc_vals if a == 0)} sınıf")
    print(f"     1-25%:   {sum(1 for a in acc_vals if 0 < a <= 25)} sınıf")
    print(f"     25-50%:  {sum(1 for a in acc_vals if 25 < a <= 50)} sınıf")
    print(f"     50-75%:  {sum(1 for a in acc_vals if 50 < a <= 75)} sınıf")
    print(f"     75-100%: {sum(1 for a in acc_vals if a > 75)} sınıf")
    
    # En çok karıştırılan çiftler
    print(f"\n  🔄 En Çok Karıştırılan Çiftler:")
    for (tc, pc), cnt in confusions.most_common(10):
        tn = class_names.get(tc, f"C{tc}") if class_names else f"C{tc}"
        pn = class_names.get(pc, f"C{pc}") if class_names else f"C{pc}"
        print(f"     {tn:22s} → {pn:22s}: {cnt}x")


# ==============================================================================
# BÖLÜM 7: MODELİ İNDİRMEYE HAZIRLAMA
# ==============================================================================

def prepare_for_deployment(cfg, class_names=None):
    """
    Eğitilmiş modeli bilgisayarına indirmeye hazırla.
    """
    print("\n" + "="*70)
    print("  BÖLÜM 7: MODELİ İNDİRMEYE HAZIRLAMA")
    print("="*70)
    
    save_dir = Path(cfg.SAVE_DIR)
    
    # label_map.json oluştur
    if class_names:
        label_map = {str(k): v for k, v in class_names.items()}
    else:
        # Models_v3_80plus'tan kopyala
        existing = os.path.join(cfg.MEVCUT_MODEL_DIR, 'label_map.json')
        if os.path.exists(existing):
            with open(existing, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
        else:
            label_map = {str(i): f"Sinif_{i}" for i in range(cfg.NUM_CLASSES)}
    
    label_map_path = save_dir / 'label_map.json'
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"\n  📁 İndirilecek dosyalar ({save_dir}):")
    
    files_to_download = ['best_model.pt', 'norm_mean.npy', 'norm_std.npy', 'label_map.json']
    
    for fname in files_to_download:
        fpath = save_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"     ✅ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"     ❌ {fname} (BULUNAMADI!)")
    
    print(f"\n  📋 Bu 4 dosyayı indir ve bilgisayarında şuraya koy:")
    print(f"     C:\\Projects\\sign_bridge\\autsl_transformer\\model\\")
    print(f"")
    print(f"     best_model.pt  → best_model.pt")
    print(f"     norm_mean.npy  → norm_mean.npy")
    print(f"     norm_std.npy   → norm_std.npy")
    print(f"     label_map.json → label_map.json")
    print(f"")
    print(f"  💡 Drive'da klasör: {cfg.SAVE_DIR}")
    print(f"     Sol panel > Files > Drive > AUTSL_Proje > Models_v4_yeni_egitim")
    print(f"     Sağ tık > Download")
    
    print(f"\n  ✅ Model kullanıma hazır!")


# ==============================================================================
# ANA ÇALIŞTIRMA
# ==============================================================================

def main():
    """Ana pipeline — sırasıyla tüm adımları çalıştırır"""
    
    print("\n" + "🔥" * 35)
    print("  SignBridge - AUTSL Model Eğitim Pipeline v4")
    print("  İskeletler HAZIR — Doğrudan eğitime geçiliyor!")
    print("🔥" * 35)
    
    # Klasör kontrolü
    if not os.path.exists(cfg.KOORDINAT_DIR):
        print(f"\n❌ Koordinatlar klasörü bulunamadı: {cfg.KOORDINAT_DIR}")
        print(f"   Drive'daki AUTSL_Proje klasör adını kontrol edin!")
        return
    
    # Adım 1: Sınıf isimleri
    class_names = load_class_names(cfg.CLASS_LIST_CSV)
    
    # Adım 2: Veri yükle (mevcut iskelet packed npz)
    data = load_skeleton_data()
    if data is None:
        return
    
    # Adım 3: Eğitim (checkpoint destekli)
    model, history, norm_mean, norm_std = train_model(data, cfg)
    
    # Adım 4: Analiz
    analyze_results(data, cfg, class_names)
    
    # Adım 5: İndirmeye hazırla
    prepare_for_deployment(cfg, class_names)
    
    print("\n" + "✅" * 35)
    print("  Pipeline tamamlandı!")
    print("  Modeli Drive'dan indir ve bilgisayarına koy.")
    print("✅" * 35)


# Çalıştır!
if __name__ == '__main__':
    main()
