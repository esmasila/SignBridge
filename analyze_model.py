"""
Model Analiz Aracı
Mevcut modeli test edip zayıf noktaları tespit eder.
Webcam'den veya test verisinden analiz yapar.

Kullanım:
    python analyze_model.py --mode test_data
    python analyze_model.py --mode webcam
    python analyze_model.py --mode compare_models
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
import math
import time


# Model tanımı (inference.py ile aynı)
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


def load_model(model_name='best_model.pt'):
    """Model yükle"""
    model_dir = Path(__file__).parent / 'autsl_transformer' / 'model'
    
    # Model dosyaları
    model_path = model_dir / model_name
    mean_path = model_dir / 'norm_mean.npy'
    std_path = model_dir / 'norm_std.npy'
    label_path = model_dir / 'label_map.json'
    
    # Yükle
    norm_mean = np.load(mean_path).flatten()
    norm_std = np.maximum(np.load(std_path).flatten(), 1e-8)
    
    with open(label_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    model = SignTransformerPro(
        input_size=config.get('input_size', 225),
        d_model=config.get('d_model', 384),
        nhead=config.get('nhead', 12),
        num_layers=config.get('num_layers', 6),
        num_classes=config.get('num_classes', 226)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model: {model_name}")
    print(f"Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    print(f"Test Acc: {checkpoint.get('test_acc', 'N/A')}")
    
    return model, label_map, norm_mean, norm_std, config


def analyze_test_data(data_path=None):
    """Test verisi üzerinde detaylı analiz"""
    print("\n" + "="*60)
    print("  Model Analiz Raporu")
    print("="*60)
    
    # Model yükle
    model_dir = Path(__file__).parent / 'autsl_transformer' / 'model'
    
    # Tüm modelleri karşılaştır
    model_files = [f for f in model_dir.glob('*.pt')]
    print(f"\n  Bulunan model dosyalari ({len(model_files)}):")
    
    for mf in model_files:
        try:
            cp = torch.load(mf, map_location='cpu', weights_only=False)
            val_acc = cp.get('val_acc', 'N/A')
            test_acc = cp.get('test_acc', 'N/A')
            epoch = cp.get('epoch', 'N/A')
            config = cp.get('config', {})
            
            print(f"    {mf.name:30s} | Val: {val_acc:>8} | Test: {test_acc:>8} | Epoch: {epoch}")
            if config:
                print(f"      Config: d_model={config.get('d_model')}, layers={config.get('num_layers')}, "
                      f"heads={config.get('nhead')}, input={config.get('input_size')}")
        except Exception as e:
            print(f"    {mf.name:30s} | HATA: {e}")
    
    # Test verisi varsa analiz et
    test_data_path = Path(data_path) if data_path else None
    
    if test_data_path and test_data_path.exists():
        print(f"\n  Test verisi analiz ediliyor: {test_data_path}")
        
        model, label_map, norm_mean, norm_std, config = load_model()
        
        # Test verisini yükle
        data = np.load(test_data_path, allow_pickle=True)
        
        if 'X_test' in data:
            X_test = data['X_test']
            y_test = data['y_test']
        elif 'sequences' in data:
            X_test = data['sequences']
            y_test = data['labels']
        else:
            print(f"  Desteklenmeyen veri formati. Anahtarlar: {list(data.keys())}")
            return
        
        print(f"  Test verisi: {X_test.shape}")
        
        # Tahmin yap
        device = torch.device('cpu')
        model.to(device)
        
        correct = 0
        total = 0
        class_correct = Counter()
        class_total = Counter()
        confusion_pairs = Counter()
        confidence_by_class = defaultdict(list)
        all_confidences = []
        
        batch_size = 64
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            
            # Normalize
            batch_norm = (batch_x - norm_mean) / norm_std
            x = torch.tensor(batch_norm, dtype=torch.float32)
            
            with torch.no_grad():
                outputs = model(x)
                probs = F.softmax(outputs, dim=-1)
                confidence, predicted = probs.max(1)
            
            for j in range(len(batch_y)):
                true_label = int(batch_y[j])
                pred_label = predicted[j].item()
                conf = confidence[j].item()
                
                class_total[true_label] += 1
                all_confidences.append(conf)
                confidence_by_class[true_label].append(conf)
                
                if pred_label == true_label:
                    correct += 1
                    class_correct[true_label] += 1
                else:
                    confusion_pairs[(true_label, pred_label)] += 1
                
                total += 1
        
        overall_acc = 100.0 * correct / total
        avg_conf = np.mean(all_confidences)
        
        print(f"\n  Genel Dogruluk: {overall_acc:.2f}%")
        print(f"  Ortalama Guven: {avg_conf*100:.1f}%")
        
        # Per-class accuracy
        class_accs = {}
        for c in range(len(label_map)):
            if class_total[c] > 0:
                acc = 100.0 * class_correct[c] / class_total[c]
                class_accs[c] = acc
        
        # En kötü 20 sınıf
        sorted_classes = sorted(class_accs.items(), key=lambda x: x[1])
        
        print(f"\n  EN KOTU 20 SINIF:")
        print(f"  {'Sinif':25s} | {'Dogruluk':>10s} | {'Dogru/Toplam':>15s} | {'Ort. Guven':>12s}")
        print(f"  " + "-"*70)
        
        for c, acc in sorted_classes[:20]:
            name = label_map.get(c, f"Class_{c}")
            avg_c = np.mean(confidence_by_class[c]) * 100 if confidence_by_class[c] else 0
            print(f"  {name:25s} | {acc:8.1f}%  | {class_correct[c]:>5d}/{class_total[c]:<5d}   | {avg_c:8.1f}%")
        
        # En iyi 10 sınıf
        print(f"\n  EN IYI 10 SINIF:")
        print(f"  {'Sinif':25s} | {'Dogruluk':>10s} | {'Dogru/Toplam':>15s} | {'Ort. Guven':>12s}")
        print(f"  " + "-"*70)
        
        for c, acc in sorted_classes[-10:]:
            name = label_map.get(c, f"Class_{c}")
            avg_c = np.mean(confidence_by_class[c]) * 100 if confidence_by_class[c] else 0
            print(f"  {name:25s} | {acc:8.1f}%  | {class_correct[c]:>5d}/{class_total[c]:<5d}   | {avg_c:8.1f}%")
        
        # En çok karıştırılan çiftler
        print(f"\n  EN COK KARISTIRILAN CIFTLER:")
        print(f"  {'Gercek':20s} -> {'Tahmin':20s} | {'Sayi':>5s}")
        print(f"  " + "-"*55)
        
        for (true_c, pred_c), count in confusion_pairs.most_common(20):
            true_name = label_map.get(true_c, f"C{true_c}")
            pred_name = label_map.get(pred_c, f"C{pred_c}")
            print(f"  {true_name:20s} -> {pred_name:20s} | {count:>5d}")
        
        # Güven dağılımı
        confs = np.array(all_confidences)
        print(f"\n  GUVEN DAGILIMI:")
        print(f"  %0-20:   {np.sum(confs < 0.2):>5d} ({100*np.mean(confs < 0.2):.1f}%)")
        print(f"  %20-40:  {np.sum((confs >= 0.2) & (confs < 0.4)):>5d} ({100*np.mean((confs >= 0.2) & (confs < 0.4)):.1f}%)")
        print(f"  %40-60:  {np.sum((confs >= 0.4) & (confs < 0.6)):>5d} ({100*np.mean((confs >= 0.4) & (confs < 0.6)):.1f}%)")
        print(f"  %60-80:  {np.sum((confs >= 0.6) & (confs < 0.8)):>5d} ({100*np.mean((confs >= 0.6) & (confs < 0.8)):.1f}%)")
        print(f"  %80-100: {np.sum(confs >= 0.8):>5d} ({100*np.mean(confs >= 0.8):.1f}%)")
        
        # Rapor kaydet
        report = {
            'overall_accuracy': overall_acc,
            'avg_confidence': float(avg_conf),
            'num_classes': len(label_map),
            'total_test': total,
            'per_class_accuracy': {label_map.get(c, f"C{c}"): acc for c, acc in class_accs.items()},
            'worst_classes': [(label_map.get(c, f"C{c}"), acc) for c, acc in sorted_classes[:20]],
            'best_classes': [(label_map.get(c, f"C{c}"), acc) for c, acc in sorted_classes[-10:]],
            'top_confusions': [
                (label_map.get(tc, f"C{tc}"), label_map.get(pc, f"C{pc}"), cnt) 
                for (tc, pc), cnt in confusion_pairs.most_common(30)
            ]
        }
        
        report_path = Path(__file__).parent / 'model_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Rapor kaydedildi: {report_path}")
    
    else:
        print("\n  Test verisi bulunamadi.")
        print("  Kullanim: python analyze_model.py --data_path /yol/test_data.npz")
        print("\n  Sadece model dosyalari karsilastirildi.")


def compare_normalization():
    """Normalizasyon farklarını kontrol et"""
    print("\n" + "="*60)
    print("  Normalizasyon Analizi")
    print("="*60)
    
    model_dir = Path(__file__).parent / 'autsl_transformer' / 'model'
    
    mean = np.load(model_dir / 'norm_mean.npy')
    std = np.load(model_dir / 'norm_std.npy')
    
    print(f"\n  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    
    mean_flat = mean.flatten()
    std_flat = std.flatten()
    
    print(f"\n  Mean istatistikleri:")
    print(f"    Min: {mean_flat.min():.6f}")
    print(f"    Max: {mean_flat.max():.6f}")
    print(f"    Mean: {mean_flat.mean():.6f}")
    print(f"    Std: {mean_flat.std():.6f}")
    
    print(f"\n  Std istatistikleri:")
    print(f"    Min: {std_flat.min():.6f}")
    print(f"    Max: {std_flat.max():.6f}")
    print(f"    Mean: {std_flat.mean():.6f}")
    print(f"    Near-zero (< 0.001): {np.sum(std_flat < 0.001)}")
    
    # std < 0.1 olan feature'lar (eski kodda 0.1'e clip ediliyordu!)
    problematic = np.sum(std_flat < 0.1)
    print(f"\n  POTANSIYEL SORUN:")
    print(f"    Std < 0.1 olan feature: {problematic}/{len(std_flat)}")
    print(f"    Bu feature'lar eski kodda 0.1'e clip ediliyordu,")
    print(f"    bu egitimle uyumsuzluk yaratir!")
    
    # Pose vs Hand std karşılaştırması
    pose_std = std_flat[:99]
    lh_std = std_flat[99:162]
    rh_std = std_flat[162:225]
    
    print(f"\n  Bolge bazli std ortalamalari:")
    print(f"    Pose (0-99):       {pose_std.mean():.4f}")
    print(f"    Left Hand (99-162): {lh_std.mean():.4f}")
    print(f"    Right Hand (162-225): {rh_std.mean():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AUTSL Model Analiz Araci')
    parser.add_argument('--mode', type=str, default='info',
                        choices=['info', 'test_data', 'normalization'],
                        help='Analiz modu')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Test verisi yolu (.npz)')
    
    args = parser.parse_args()
    
    if args.mode == 'info':
        analyze_test_data(args.data_path)
    elif args.mode == 'test_data':
        analyze_test_data(args.data_path)
    elif args.mode == 'normalization':
        compare_normalization()
