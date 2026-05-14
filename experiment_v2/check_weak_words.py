"""
Hangi kelimeler zayif? Val seti uzerinde her sinifin dogruluk oranini goster.

Calistir: python experiment_v2/check_weak_words.py
"""

import sys
import io
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE   = Path(__file__).parent
CKPT   = BASE / "checkpoints" / "best_model.pt"
AUG_DIR = BASE / "data_aug"

class GRUModel(nn.Module):
    def __init__(self, input_size=1629, hidden=256, layers=2, classes=3, dropout=0.3):
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

class SeqDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.label_map = {}
        for idx, cls_dir in enumerate(sorted([d for d in data_dir.iterdir() if d.is_dir()])):
            self.label_map[cls_dir.name] = idx
            for f in cls_dir.glob("*.npy"):
                self.samples.append((f, idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        seq = np.clip(np.load(path).astype(np.float32), 0, 1)
        return torch.tensor(seq), torch.tensor(label, dtype=torch.long)

ckpt        = torch.load(CKPT, map_location="cpu")
label_map   = ckpt["label_map"]
num_classes = ckpt["num_classes"]
idx_to_label = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GRUModel(classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dataset = SeqDataset(AUG_DIR)
_, val_ds = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)],
                          generator=torch.Generator().manual_seed(42))

# Her sinif icin dogru/yanlis say
correct_per_class = {k: 0 for k in label_map}
total_per_class   = {k: 0 for k in label_map}
confused_with     = {k: {} for k in label_map}   # hangi sinifla karistirildi

loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

with torch.no_grad():
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        out  = model(seqs)
        preds = out.argmax(1)
        for true, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            true_name = idx_to_label[true]
            pred_name = idx_to_label[pred]
            total_per_class[true_name] += 1
            if true == pred:
                correct_per_class[true_name] += 1
            else:
                confused_with[true_name][pred_name] = confused_with[true_name].get(pred_name, 0) + 1

# Sonuclari yazdir
print("\n" + "="*60)
print("KELİME BAZLI DOĞRULUK RAPORU")
print("="*60)

results = []
for label in label_map:
    total   = total_per_class[label]
    correct = correct_per_class[label]
    acc     = 100 * correct / total if total > 0 else 0
    results.append((acc, label, correct, total, confused_with[label]))

results.sort(key=lambda x: x[0])   # en kotu basta

WEAK_THRESHOLD = 85.0

print(f"\n{'Kelime':<20} {'Doğruluk':>10}  {'Doğru/Toplam':>14}  {'Karıştırıldığı'}")
print("-"*70)
for acc, label, correct, total, confused in results:
    flag  = " *** ZAYIF ***" if acc < WEAK_THRESHOLD else ""
    top_confused = sorted(confused.items(), key=lambda x: -x[1])[:2]
    confused_str = ", ".join([f"{k}({v}x)" for k,v in top_confused]) if top_confused else "-"
    bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
    print(f"  {label:<18} {acc:>6.1f}%  {bar}  {correct:>3}/{total:<4}  {confused_str}{flag}")

weak = [label for acc, label, *_ in results if acc < WEAK_THRESHOLD]
print("\n" + "="*60)
if weak:
    print(f"ZAYIF KELİMELER ({len(weak)} adet): {', '.join(weak)}")
    print("\nONERİ: Bu kelimeler icin 20-30 ornek daha topla:")
    for w in weak:
        print(f"  python experiment_v2/collect.py  -> '{w}'")
    print("\nSonra yeniden egit:")
    print("  python experiment_v2/pipeline.py --skip-extract")
else:
    print("Tum kelimeler guclu! (%85 uzeri)")
print("="*60 + "\n")
