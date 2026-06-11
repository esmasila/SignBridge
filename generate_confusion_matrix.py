"""Generate confusion matrix from GRU model + validation data."""
import sys, json, numpy as np
from pathlib import Path

BASE = Path("experiment_v2")
CKPT = BASE / "checkpoints"
AUG_DIR = BASE / "data_aug"
OUT = Path("report_images/gru_confusion_matrix.png")

# Load label map
with open(CKPT / "label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)
print(f"Classes: {num_classes}")

# Load data
SEQ_LEN = 30
X_all, y_all = [], []
for cls_name, cls_idx in sorted(label_map.items(), key=lambda x: x[1]):
    cls_dir = AUG_DIR / cls_name
    if not cls_dir.exists():
        continue
    files = sorted(cls_dir.glob("*.npy"))
    for f in files:
        seq = np.load(f)
        if seq.shape[0] >= SEQ_LEN:
            seq = seq[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        X_all.append(seq)
        y_all.append(cls_idx)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)
print(f"Total samples: {len(y_all)}")

# Split same as training (80/20)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f"Validation samples: {len(y_val)}")

# Load model
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1629, hidden=256, layers=2, classes=100, dropout=0.3):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(classes=num_classes).to(device)

ckpt = torch.load(CKPT / "best_model.pt", map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"Model loaded. Device: {device}")

# Predict on validation set
all_preds = []
all_true = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(X_val), batch_size):
        batch_x = torch.tensor(X_val[i:i+batch_size]).to(device)
        batch_y = y_val[i:i+batch_size]
        out = model(batch_x)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(batch_y)

all_preds = np.array(all_preds)
all_true = np.array(all_true)
acc = (all_preds == all_true).mean()
print(f"Validation Accuracy: {acc*100:.1f}%")

# Generate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = [idx_to_label[i] for i in range(num_classes)]
cm = confusion_matrix(all_true, all_preds)

# Normalize
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(24, 20))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
ax.set_title(f'GRU Model - Confusion Matrix (100 Kelime)\nValidation Accuracy: {acc*100:.1f}%',
             fontsize=16, fontweight='bold', pad=20)

# Labels
tick_marks = np.arange(num_classes)
ax.set_xticks(tick_marks)
ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=5)
ax.set_yticks(tick_marks)
ax.set_yticklabels(labels, fontsize=5)

ax.set_xlabel('Tahmin', fontsize=14)
ax.set_ylabel('Gerçek', fontsize=14)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(OUT), dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved: {OUT}")
print(f"Size: {OUT.stat().st_size / 1024:.0f} KB")

# Also save classification report
report = classification_report(all_true, all_preds, target_names=labels, output_dict=False)
print(f"\nClassification Report:\n{report}")
