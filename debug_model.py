"""
Model Test - Eğitim verisi formatını kontrol et
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path

# Model tanımı
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
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
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.35):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//4), nn.Tanh(), nn.Linear(d_model//4, 1))
            for _ in range(4)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model*5), nn.Dropout(dropout),
            nn.Linear(d_model*5, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(d_model, num_classes)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.input_conv(x)
        x = x + self.conv_block(x.transpose(1,2)).transpose(1,2)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1)
        x = self.transformer(self.pos_encoder(x))
        seq_out = x[:, 1:]
        pooled = [F.softmax(h(seq_out), dim=1) * seq_out for h in self.pool_heads]
        pooled = [p.sum(dim=1) for p in pooled]
        combined = torch.cat(pooled + [seq_out.mean(dim=1)], dim=1)
        return self.classifier(combined)


def main():
    MODEL_DIR = Path('autsl_transformer/model')
    
    # Model yükle
    checkpoint = torch.load(MODEL_DIR / 'autsl_pro_final.pt', map_location='cpu')
    config = checkpoint['config']
    
    print("="*50)
    print("MODEL CONFIG")
    print("="*50)
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    model = SignTransformerPro(
        input_size=config['input_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Label map
    with open(MODEL_DIR / 'label_map.json', 'r') as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    
    # Normalizasyon
    norm_mean = np.load(MODEL_DIR / 'norm_mean.npy')
    norm_std = np.load(MODEL_DIR / 'norm_std.npy')
    
    print("\n" + "="*50)
    print("NORMALIZASYON DEĞERLERİ")
    print("="*50)
    mean_flat = norm_mean.flatten()
    std_flat = norm_std.flatten()
    
    print(f"Mean shape: {norm_mean.shape}")
    print(f"Std shape: {norm_std.shape}")
    print(f"Mean range: {mean_flat.min():.4f} to {mean_flat.max():.4f}")
    print(f"Std range: {std_flat.min():.8f} to {std_flat.max():.4f}")
    
    # Çok düşük std değerleri
    low_std_count = np.sum(std_flat < 0.01)
    print(f"Düşük std (<0.01) sayısı: {low_std_count} / 225")
    
    print("\n" + "="*50)
    print("TEST 1: Sıfır veri (el yok durumu)")
    print("="*50)
    
    # El yok durumunu simüle et
    zero_sequence = np.zeros((30, 225), dtype=np.float32)
    
    std_fixed = np.clip(norm_std.squeeze(), 0.1, None)
    normalized = (zero_sequence - norm_mean.squeeze()) / std_fixed
    
    x = torch.FloatTensor(normalized).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = probs.topk(5)
        
        print("Top-5 tahminler (sıfır veri):")
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item()
            label = label_map.get(idx, f'Class_{idx}')
            print(f"  {i+1}. {label}: {prob*100:.2f}%")
    
    print("\n" + "="*50)
    print("TEST 2: Ortalama değerlere yakın veri")
    print("="*50)
    
    # Ortalama değerlere yakın veri
    avg_sequence = np.tile(mean_flat, (30, 1)).astype(np.float32)
    
    normalized = (avg_sequence - norm_mean.squeeze()) / std_fixed
    
    x = torch.FloatTensor(normalized).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = probs.topk(5)
        
        print("Top-5 tahminler (ortalama veri):")
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item()
            label = label_map.get(idx, f'Class_{idx}')
            print(f"  {i+1}. {label}: {prob*100:.2f}%")
    
    print("\n" + "="*50)
    print("TEST 3: Rastgele gerçekçi veri")
    print("="*50)
    
    np.random.seed(42)
    # Gerçekçi değerler (x,y: 0-1 arası, z: -1 ile 1 arası)
    realistic_sequence = np.zeros((30, 225), dtype=np.float32)
    
    for i in range(30):
        # Pose (33 nokta)
        for j in range(33):
            realistic_sequence[i, j*3] = np.random.uniform(0.3, 0.7)  # x
            realistic_sequence[i, j*3+1] = np.random.uniform(0.2, 0.8)  # y
            realistic_sequence[i, j*3+2] = np.random.uniform(-0.5, 0.5)  # z
        
        # Left hand (21 nokta)
        for j in range(21):
            idx = 99 + j*3
            realistic_sequence[i, idx] = np.random.uniform(0.2, 0.4)  # x
            realistic_sequence[i, idx+1] = np.random.uniform(0.3, 0.5)  # y
            realistic_sequence[i, idx+2] = np.random.uniform(-0.1, 0.1)  # z
        
        # Right hand (21 nokta)
        for j in range(21):
            idx = 162 + j*3
            realistic_sequence[i, idx] = np.random.uniform(0.6, 0.8)  # x
            realistic_sequence[i, idx+1] = np.random.uniform(0.3, 0.5)  # y
            realistic_sequence[i, idx+2] = np.random.uniform(-0.1, 0.1)  # z
    
    print(f"Sequence min/max: {realistic_sequence.min():.3f} / {realistic_sequence.max():.3f}")
    
    normalized = (realistic_sequence - norm_mean.squeeze()) / std_fixed
    print(f"Normalized min/max: {normalized.min():.3f} / {normalized.max():.3f}")
    
    x = torch.FloatTensor(normalized).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = probs.topk(5)
        
        print("\nTop-5 tahminler (rastgele gerçekçi veri):")
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item()
            label = label_map.get(idx, f'Class_{idx}')
            print(f"  {i+1}. {label}: {prob*100:.2f}%")
    
    print("\n" + "="*50)
    print("SONUÇ")
    print("="*50)
    print("Eğer tüm testlerde düşük güven (<20%) görüyorsanız:")
    print("1. Model eğitim verisiyle uyumsuz")
    print("2. Normalizasyon parametreleri yanlış")
    print("3. Checkpoint doğru yüklenmemiş olabilir")


if __name__ == "__main__":
    main()
