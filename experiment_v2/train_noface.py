"""
Yuz landmarksiz model egitimi (DENEY)
- Mevcut data_aug/ klasorundan .npy dosyalarini okur
- Her frame: 1629 ozellik → ilk 1404 (yuz) ATILIR → kalan 225 (pose+eller) kullanilir
- GRU(input=225) egitilir, best_model_noface.pt olarak kaydedilir
- Port 5051'de web_app_noface.py ile test edilir

Kullanim:
    python experiment_v2/train_noface.py
"""

import sys
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE     = Path(__file__).parent
AUG_DIR  = BASE / "data_aug"
CKPT_DIR = BASE / "checkpoints"
CKPT     = CKPT_DIR / "best_model_noface.pt"
CKPT_DIR.mkdir(exist_ok=True)

# Yuz: 468*3=1404, Pose: 33*3=99, Sol el: 21*3=63, Sag el: 21*3=63 → Toplam: 1629
# Yuzsuz: 1629 - 1404 = 225
FACE_DIM   = 468 * 3   # 1404
INPUT_SIZE = 225        # pose + eller


# ---- Model ----
class GRUModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=256, layers=2, classes=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, classes)
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])


# ---- Dataset (yuz kirp) ----
class NoFaceDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.label_map = {}
        cls_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        for idx, cls_dir in enumerate(cls_dirs):
            self.label_map[cls_dir.name] = idx
            for f in cls_dir.glob("*.npy"):
                self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        seq = np.load(path).astype(np.float32)          # (30, 1629)
        seq = np.clip(seq, 0, 1)
        seq = seq[:, FACE_DIM:]                          # (30, 225) — yuzu at
        return torch.tensor(seq), torch.tensor(label, dtype=torch.long)


# ---- Veri ----
print("Veri yukleniyor (yuz atiliyor)...")
dataset = NoFaceDataset(AUG_DIR)
label_map   = dataset.label_map
num_classes = len(label_map)
print(f"  Sinif: {num_classes} | Ornek: {len(dataset)} | Ozellik: {INPUT_SIZE}")

# Sinif agirliklari
counts = [0] * num_classes
for _, lbl in dataset.samples:
    counts[lbl] += 1
total   = sum(counts)
weights = torch.tensor([total / (num_classes * c) for c in counts], dtype=torch.float32)

val_size   = max(1, int(len(dataset) * 0.2))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
print(f"  Train: {train_size} | Val: {val_size}")

# ---- Egitim ----
NUM_EPOCHS = 50
LR         = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model     = GRUModel(classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

best_acc   = 0.0
best_epoch = 0

print(f"\n{'='*60}")
print(f"YUZSUZ MODEL EGITIMI ({NUM_EPOCHS} epoch, LR={LR})")
print(f"{'='*60}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    t_correct = t_total = 0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(seqs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_correct += (out.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)

    # Val
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            out = model(seqs)
            v_loss    += criterion(out, labels).item()
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total   += labels.size(0)

    t_acc = 100 * t_correct / t_total
    v_acc = 100 * v_correct / v_total
    scheduler.step(v_acc)

    tag = ""
    if v_acc > best_acc:
        best_acc   = v_acc
        best_epoch = epoch
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "val_acc":          v_acc,
            "label_map":        label_map,
            "num_classes":      num_classes,
            "input_size":       INPUT_SIZE,
            "model":            "GRU_noface",
        }, CKPT)
        tag = "  *** BEST ***"

    print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} | Train {t_acc:5.1f}% | Val {v_acc:5.1f}% | Loss {v_loss/len(val_loader):.4f}{tag}")

print(f"\n{'='*60}")
print(f"TAMAMLANDI! En iyi val_acc: {best_acc:.2f}% (Epoch {best_epoch})")
print(f"Model: {CKPT}")
print(f"Simdi test: python experiment_v2/web_app_noface.py")
print(f"{'='*60}\n")
