"""
V2 egitim: data_aug_v2/ (onceden normalize + velocity + clip edilmis)
Once: python experiment_v2/precompute_v2.py (bir kere)
Sonra: python experiment_v2/finetune.py (her seferinde, cok hizli)
"""

import sys, io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data_aug_v2"     # ONCEDEN ISLENMIS (1755 dim)
CKPT_DIR = BASE / "checkpoints"
CKPT     = CKPT_DIR / "best_model_v2.pt"
CKPT_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    raise SystemExit(
        f"\n{DATA_DIR} yok!\n"
        f"Once calistir: python experiment_v2/precompute_v2.py\n"
    )

INPUT_SIZE = 1755

# ---- Model ----
class GRUModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=256, layers=2, classes=100, dropout=0.3):
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

# ---- Dataset: TUM veriyi bir kereye RAM'e yukle (disk I/O ortadan kalkar) ----
class SeqDataset(Dataset):
    def __init__(self, data_dir):
        self.label_map = {}
        cls_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

        # Once dosya listesi + label
        files_labels = []
        for idx, cls_dir in enumerate(cls_dirs):
            self.label_map[cls_dir.name] = idx
            for f in sorted(cls_dir.glob("*.npy")):
                files_labels.append((f, idx))

        n = len(files_labels)
        print(f"  RAM'e yukleniyor ({n} dosya)...", flush=True)
        import time
        t0 = time.time()
        # Tek buyuk tensor: (N, 30, 1755)
        self.data   = np.empty((n, 30, 1755), dtype=np.float32)
        self.labels = np.empty(n, dtype=np.int64)
        for i, (path, lbl) in enumerate(files_labels):
            self.data[i]   = np.load(path)
            self.labels[i] = lbl
            if (i + 1) % 10000 == 0:
                print(f"    {i+1}/{n}  ({time.time()-t0:.1f}s)", flush=True)
        print(f"  Yuklendi: {n} ornek, {self.data.nbytes/(1024**3):.2f} GB, {time.time()-t0:.1f}s", flush=True)

        # Tensor'a cevir (GPU'ya tasinmadan)
        self.data   = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    @property
    def samples(self):
        # Sinif agirligi hesaplamasi icin
        return [(None, int(l)) for l in self.labels]

# ---- Veri ----
print("Veri yukleniyor (on-islenmis)...")
dataset     = SeqDataset(DATA_DIR)
label_map   = dataset.label_map
num_classes = len(label_map)
print(f"  Sinif: {num_classes} | Ornek: {len(dataset)} | Input: {INPUT_SIZE}")

counts = [0] * num_classes
for _, lbl in dataset.samples:
    counts[lbl] += 1
total   = sum(counts)
weights = torch.tensor([total / (num_classes * c) for c in counts], dtype=torch.float32)

val_size   = max(1, int(len(dataset) * 0.2))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)
print(f"  Train: {train_size} | Val: {val_size}")

# ---- Egitim ----
NUM_EPOCHS = 80
LR         = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model     = GRUModel(classes=num_classes).to(device)
params    = sum(p.numel() for p in model.parameters())
print(f"  Parametre: {params:,}")

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

best_acc   = 0.0
best_epoch = 0

print(f"\n{'='*60}")
print(f"EGITIM V2 ({NUM_EPOCHS} epoch, LR={LR}, batch=64)")
print(f"{'='*60}\n")

import time
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    t_correct = t_total = 0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        out  = model(seqs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_correct += (out.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)

    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            out = model(seqs)
            v_loss    += criterion(out, labels).item()
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total   += labels.size(0)

    t_acc = 100 * t_correct / t_total
    v_acc = 100 * v_correct / v_total
    scheduler.step(v_acc)
    dt = time.time() - t0

    tag = ""
    if v_acc > best_acc:
        best_acc   = v_acc
        best_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "val_acc": v_acc, "label_map": label_map,
            "num_classes": num_classes, "input_size": INPUT_SIZE,
            "model": "GRU_norm_vel",
        }, CKPT)
        tag = "  *** BEST ***"

    print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} | Train {t_acc:5.1f}% | Val {v_acc:5.1f}% | Loss {v_loss/len(val_loader):.4f} | {dt:.1f}s{tag}")

print(f"\n{'='*60}")
print(f"TAMAMLANDI! En iyi val_acc: {best_acc:.2f}% (Epoch {best_epoch})")
print(f"Model: {CKPT}")
print(f"{'='*60}\n")
