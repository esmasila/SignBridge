"""Poster için: 122 kelime modelinin gerçek validation accuracy + confusion matrix."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import torch, torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = Path(__file__).parent
DATA = BASE / "data_aug_v2"
CKPT = BASE / "checkpoints" / "best_model.pt"
OUT = BASE / "poster_artifacts"
OUT.mkdir(exist_ok=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
label_map = ck["label_map"]
num_classes = len(label_map)
inv = {v: k for k, v in label_map.items()}
print(f"Model: {num_classes} sinif, kayitli val_acc={ck['val_acc']:.2f}%")

# Model mimari - basit GRU
class GRUNet(nn.Module):
    def __init__(self, inp=1755, hid=256, layers=2, out=122, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(inp, hid, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, out),
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])

model = GRUNet(out=num_classes)
model.load_state_dict(ck["model_state_dict"])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

# Stratified sample: her sınıftan en fazla 30 ornek
rng = np.random.RandomState(42)
X, y = [], []
for cls_name, cls_idx in label_map.items():
    cls_dir = DATA / cls_name
    if not cls_dir.exists(): continue
    files = list(cls_dir.glob("*.npy"))
    rng.shuffle(files)
    for f in files[:30]:
        try:
            arr = np.load(f)
            if arr.shape == (30, 1755):
                X.append(arr)
                y.append(cls_idx)
        except: pass

X = np.stack(X).astype(np.float32)
y = np.array(y)
print(f"Eval set: {len(y)} ornek, {num_classes} sinif")

# Predict batched
preds = []
with torch.no_grad():
    for i in range(0, len(X), 256):
        b = torch.from_numpy(X[i:i+256]).to(device)
        out = model(b)
        preds.append(out.argmax(1).cpu().numpy())
preds = np.concatenate(preds)

# Per-class accuracy
acc = (preds == y).mean() * 100
print(f"\nGenel dogruluk: {acc:.2f}%")

per_cls = defaultdict(lambda: [0, 0])
for p, t in zip(preds, y):
    per_cls[t][1] += 1
    if p == t: per_cls[t][0] += 1

# En kotuler
worst = sorted(per_cls.items(), key=lambda x: x[1][0]/max(x[1][1],1))[:10]
print("\nEn dusuk dogrulukli 10 kelime:")
for cls_idx, (c, n) in worst:
    print(f"  {inv[cls_idx]:<15} {c}/{n}  ({c/max(n,1)*100:.1f}%)")

# Confusion matrix - sadece top karisanlari goster
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds, labels=list(range(num_classes)))
# Diagonal'i sifirla, en cok karisan ciftleri bul
cm_noDiag = cm.copy()
np.fill_diagonal(cm_noDiag, 0)
flat = cm_noDiag.flatten()
top_pairs = np.argsort(flat)[-15:][::-1]  # en cok karisan 15 cift
print("\nEn cok karisan 15 cift (gercek -> tahmin):")
for idx in top_pairs:
    if flat[idx] == 0: continue
    r, c = idx // num_classes, idx % num_classes
    print(f"  {inv[r]:<15} -> {inv[c]:<15} ({flat[idx]} adet)")

# Save metrics
metrics = {
    "num_classes": num_classes,
    "checkpoint_val_acc": float(ck["val_acc"]),
    "eval_accuracy_percent": float(acc),
    "eval_samples": int(len(y)),
    "device": str(device),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

# Plot top-30 confused matrix
top_classes_count = 30
class_counts = [(i, per_cls[i][1]) for i in range(num_classes)]
chosen = [i for i, _ in sorted(class_counts, key=lambda x: -x[1])[:top_classes_count]]
cm_small = cm[np.ix_(chosen, chosen)]
labels_small = [inv[i] for i in chosen]

# Normalize per row
cm_norm = cm_small / np.maximum(cm_small.sum(axis=1, keepdims=True), 1)

fig, ax = plt.subplots(figsize=(11, 10), dpi=120)
im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(top_classes_count))
ax.set_yticks(range(top_classes_count))
ax.set_xticklabels(labels_small, rotation=75, fontsize=8, ha="right")
ax.set_yticklabels(labels_small, fontsize=8)
ax.set_xlabel("Tahmin Edilen", fontsize=12, fontweight="bold")
ax.set_ylabel("Gerçek Sınıf", fontsize=12, fontweight="bold")
ax.set_title(f"Karışıklık Matrisi (En Sık 30 Kelime)\nGenel Doğruluk: %{acc:.2f}",
             fontsize=14, fontweight="bold", pad=15)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_122.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nKaydedildi: {OUT / 'confusion_matrix_122.png'}")

# Eğitim grafiği - train_v2.log'tan parse et (varsa)
import re
log = BASE / "train_v2.log"
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    epochs, tr, va, lo = [], [], [], []
    for ln in lines:
        m = re.search(r"Epoch\s+(\d+)/\d+\s*\|\s*Train\s+([\d.]+)%\s*\|\s*Val\s+([\d.]+)%\s*\|\s*Loss\s+([\d.]+)", ln)
        if m:
            epochs.append(int(m.group(1)))
            tr.append(float(m.group(2)))
            va.append(float(m.group(3)))
            lo.append(float(m.group(4)))
    if epochs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        a1.plot(epochs, tr, "-o", label="Eğitim", color="#6c63ff", linewidth=2, markersize=4)
        a1.plot(epochs, va, "-s", label="Doğrulama", color="#22d3ee", linewidth=2, markersize=4)
        a1.set_xlabel("Epoch", fontweight="bold")
        a1.set_ylabel("Doğruluk (%)", fontweight="bold")
        a1.set_title("Eğitim & Doğrulama Doğruluğu", fontweight="bold", fontsize=13)
        a1.legend(fontsize=11, loc="lower right")
        a1.grid(alpha=0.3)
        a1.set_ylim(90, 101)
        a2.plot(epochs, lo, "-o", color="#ec4899", linewidth=2, markersize=4)
        a2.set_xlabel("Epoch", fontweight="bold")
        a2.set_ylabel("Kayıp (Loss)", fontweight="bold")
        a2.set_title("Doğrulama Kaybı", fontweight="bold", fontsize=13)
        a2.grid(alpha=0.3)
        plt.suptitle(f"SignBridge GRU Eğitim Süreci — Final %{ck['val_acc']:.2f} doğruluk",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(OUT / "training_history_122.png", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Kaydedildi: {OUT / 'training_history_122.png'}")

print("\n=== TAMAMLANDI ===")
print(f"Artifacts: {OUT}")
