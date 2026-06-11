"""
TİD 5-Kelime GRU Eğitim Scripti v2
225 raw → 718 feature (velocity + acceleration + distances)
Web inference ile birebir uyumlu.

KULLANIM:
    python tid_sequence/scripts/train_v2.py

Veri: tid_sequence/data_v2/{MERHABA,TEŞEKKÜR,EVET,HAYIR,LÜTFEN}/*.npy
Her .npy: (30, 225)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import time
import sys

# Proje kökü
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ── Ayarlar ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data_v2"
SAVE_DIR = Path(__file__).parent.parent / "checkpoints_v2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
RAW_FEATURES = 225  # pose(99) + left_hand(63) + right_hand(63)

# ===== INTER-LANDMARK DISTANCE PAIRS (app_transformer.py ile birebir aynı) =====
DISTANCE_PAIRS = [
    (32, 11), (32, 14), (32, 18),
    (53, 11), (53, 14), (53, 18),
    (36, 11), (36, 14),
    (57, 11), (57, 18),
    (40, 11), (61, 11),
    (32, 53), (36, 57), (40, 61), (44, 65), (48, 69),
    (36, 48), (57, 69), (36, 44), (57, 65),
    (33, 36), (54, 57), (33, 40), (54, 61),
    (32, 0), (53, 0), (32, 2), (53, 2), (32, 3), (53, 3),
    (36, 32), (40, 32), (44, 32), (48, 32),
    (57, 53), (61, 53), (65, 53), (69, 53),
    (36, 0), (57, 0), (32, 6), (53, 6),
]
NUM_DISTANCES = len(DISTANCE_PAIRS)
TOTAL_FEATURES = RAW_FEATURES * 3 + NUM_DISTANCES  # 225+225+225+43 = 718


# ── Feature Engineering (app_transformer.py ile birebir aynı) ───────────────

def compute_distances(seq):
    """seq: (T, 225) -> (T, NUM_DISTANCES)"""
    T = seq.shape[0]
    dists = np.zeros((T, NUM_DISTANCES), dtype=np.float32)
    for i, (a, b) in enumerate(DISTANCE_PAIRS):
        if a * 3 + 2 < seq.shape[1] and b * 3 + 2 < seq.shape[1]:
            ax, ay = seq[:, a * 3], seq[:, a * 3 + 1]
            bx, by = seq[:, b * 3], seq[:, b * 3 + 1]
            dists[:, i] = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + 1e-8)
    return dists


def build_features(sequence):
    """Raw 225 landmarks → 718 features (web inference ile aynı)"""
    parts = [sequence]

    # Velocity
    v = np.zeros_like(sequence)
    v[1:] = sequence[1:] - sequence[:-1]
    parts.append(v)

    # Acceleration
    a = np.zeros_like(sequence)
    a[2:] = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]
    parts.append(a)

    # Distances
    d = compute_distances(sequence)
    parts.append(d)

    return np.concatenate(parts, axis=-1).astype(np.float32)


# ── Dataset ─────────────────────────────────────────────────────────────────

# ── Data Augmentation ───────────────────────────────────────────────────────

def augment_sequence(seq, raw_dim=225):
    """Raw 225-boyutlu sekans üzerinde augmentation uygula (normalize öncesi)"""
    aug = seq.copy()

    # 1) Gaussian noise (küçük)
    if np.random.random() < 0.8:
        noise = np.random.normal(0, 0.005, aug.shape).astype(np.float32)
        aug += noise

    # 2) Time shift (1-3 frame kaydır)
    if np.random.random() < 0.5:
        shift = np.random.randint(1, 4)
        if np.random.random() < 0.5:
            aug = np.concatenate([aug[shift:], np.tile(aug[-1:], (shift, 1))], axis=0)
        else:
            aug = np.concatenate([np.tile(aug[:1], (shift, 1)), aug[:-shift]], axis=0)

    # 3) Speed change (yavaşlat/hızlandır)
    if np.random.random() < 0.4:
        T = aug.shape[0]
        speed = np.random.uniform(0.8, 1.2)
        indices = np.clip(np.arange(T) * speed, 0, T - 1).astype(int)
        aug = aug[indices]

    # 4) Spatial scale (x,y koordinatlarını hafifçe ölçekle)
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.95, 1.05)
        aug *= scale

    # 5) Random landmark dropout (bazı frame'lerde bazı landmark'ları sıfırla)
    if np.random.random() < 0.3:
        mask = np.random.random(aug.shape) > 0.05  # %5 dropout
        aug *= mask

    return aug.astype(np.float32)


class TIDDatasetV2(Dataset):
    def __init__(self, features, labels, augment=False, raw_seqs=None, norm_mean=None, norm_std=None):
        self.features = features  # list of (30, 718) arrays
        self.labels = labels
        self.augment = augment
        self.raw_seqs = raw_seqs  # augmentation için raw (30,225) sekanslar
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.augment and self.raw_seqs is not None:
            # Raw üzerinde augment → feature tekrar hesapla → normalize
            raw_aug = augment_sequence(self.raw_seqs[idx])
            feat = build_features(raw_aug)
            feat = (feat - self.norm_mean) / self.norm_std
            return torch.FloatTensor(feat), torch.LongTensor([self.labels[idx]])[0]
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def load_data():
    """
    data_v2/{LABEL}/*.npy yükle → 718 feature'a dönüştür.
    Returns: features, labels, label_map, raw_sequences (normalizasyon için)
    """
    all_features = []
    all_labels = []
    all_raw = []
    label_map = {}

    class_folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])
    if not class_folders:
        print(f"❌ Veri bulunamadı: {DATA_DIR}")
        print("   Önce collect_v2.py ile veri toplayın.")
        sys.exit(1)

    for class_idx, folder in enumerate(class_folders):
        class_name = folder.name
        label_map[class_name] = class_idx

        npy_files = sorted(folder.glob("*.npy"))
        print(f"  {class_name:15s} : {len(npy_files)} örnek")

        for npy_file in npy_files:
            raw = np.load(npy_file).astype(np.float32)  # (30, 225)

            if raw.shape != (SEQ_LEN, RAW_FEATURES):
                print(f"  ⚠️ Atlanıyor {npy_file.name}: shape={raw.shape}, beklenen=({SEQ_LEN}, {RAW_FEATURES})")
                continue

            all_raw.append(raw)
            feat = build_features(raw)  # (30, 718)
            all_features.append(feat)
            all_labels.append(class_idx)

    print(f"\n  Toplam: {len(all_features)} örnek, {len(label_map)} sınıf")
    return all_features, all_labels, label_map, all_raw


def compute_normalization(features_list):
    """Tüm eğitim verisinden mean/std hesapla (718 boyut)"""
    all_data = np.concatenate(features_list, axis=0)  # (N*30, 718)
    mean = all_data.mean(axis=0).astype(np.float32)   # (718,)
    std = np.maximum(all_data.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def normalize_features(features_list, mean, std):
    """Her sekansı normalize et"""
    return [(f - mean) / std for f in features_list]


# ── Model (Bidirectional GRU — küçük veri seti için ideal) ──────────────────

class SignGRU(nn.Module):
    """
    Bidirectional GRU + Attention — 5 sınıf için optimize
    Input: (batch, 30, 718)
    Output: (batch, num_classes)
    """
    def __init__(self, input_size=718, hidden_size=128, num_layers=2,
                 num_classes=5, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 30, 718)
        gru_out, _ = self.gru(x)  # (B, 30, hidden*2)

        # Attention pooling
        attn_weights = self.attention(gru_out)        # (B, 30, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (gru_out * attn_weights).sum(dim=1)  # (B, hidden*2)

        return self.classifier(context)


# ── Eğitim ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), 100 * correct / total


def main():
    print("=" * 60)
    print("  TİD 5-KELİME GRU EĞİTİMİ v2")
    print("  225 raw → 718 feature — Web uyumlu")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Device: {device}")

    # ── Veri yükle ──────────────────────────────────────────────────────
    print("\n📊 Veri yükleniyor...")
    features, labels, label_map, raw_seqs = load_data()

    if len(features) < 10:
        print("❌ Yeterli veri yok! En az 10 örnek gerekli.")
        sys.exit(1)

    # Normalizasyon hesapla ve kaydet
    print("\n📐 Normalizasyon hesaplanıyor...")
    norm_mean, norm_std = compute_normalization(features)
    np.save(SAVE_DIR / "norm_mean.npy", norm_mean)
    np.save(SAVE_DIR / "norm_std.npy", norm_std)
    print(f"  ✅ norm_mean.npy, norm_std.npy kaydedildi (shape={norm_mean.shape})")

    # Raw sekansları sakla (augmentation için)
    raw_seqs_norm_pre = [np.load(f).astype(np.float32)
                         for f in sorted(sum([sorted(d.glob('*.npy'))
                         for d in sorted(DATA_DIR.iterdir()) if d.is_dir()], []))]

    # Normalize et
    features = normalize_features(features, norm_mean, norm_std)

    # Label map kaydet
    with open(SAVE_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  ✅ label_map.json kaydedildi: {label_map}")

    # Train/Val split
    train_feat, val_feat, train_lbl, val_lbl = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"\n  Train: {len(train_feat)}, Val: {len(val_feat)}")

    # Raw sekansları da split et (augmentation için)
    train_raw, val_raw, _, _ = train_test_split(
        raw_seqs_norm_pre, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(
        TIDDatasetV2(train_feat, train_lbl, augment=True,
                     raw_seqs=train_raw, norm_mean=norm_mean, norm_std=norm_std),
        batch_size=16, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(TIDDatasetV2(val_feat, val_lbl),
                            batch_size=16, shuffle=False, num_workers=0)

    # ── Model ───────────────────────────────────────────────────────────
    num_classes = len(label_map)
    model = SignGRU(
        input_size=TOTAL_FEATURES,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: SignGRU (BiGRU + Attention)")
    print(f"   Parametreler: {total_params:,}")
    print(f"   Input: ({SEQ_LEN}, {TOTAL_FEATURES})")
    print(f"   Sınıflar: {num_classes}")

    # ── Eğitim ──────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    num_epochs = 200
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 40
    MIN_EPOCHS = 50  # En az 50 epoch eğit

    print(f"\n🚀 Eğitim başlıyor... ({num_epochs} epoch, early stop={early_stop_patience})\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch:3d}/{num_epochs}  "
                  f"Train: {train_loss:.4f} / {train_acc:.1f}%  "
                  f"Val: {val_loss:.4f} / {val_acc:.1f}%  "
                  f"({elapsed:.1f}s)")

        # val_acc aynıysa val_loss'a bak (daha düşük loss = daha yüksek güven)
        improved = (val_acc > best_val_acc) or \
                   (val_acc == best_val_acc and val_loss < best_val_loss)

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_map': label_map,
                'model_type': 'gru',
                'hidden_size': 128,
                'num_layers': 2,
                'num_classes': num_classes,
                'input_size': TOTAL_FEATURES,
                'seq_len': SEQ_LEN,
            }
            torch.save(checkpoint, SAVE_DIR / "gru_v2_best.pt")
            print(f"  ✅ Best model kaydedildi! (Val: {val_acc:.1f}%, Loss: {val_loss:.4f})")
        else:
            if epoch >= MIN_EPOCHS:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

    print(f"\n{'=' * 60}")
    print(f"🎉 Eğitim tamamlandı!")
    print(f"   Best Val Acc: {best_val_acc:.1f}% (Epoch {best_epoch})")
    print(f"   Dosyalar: {SAVE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
