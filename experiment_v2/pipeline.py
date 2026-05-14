"""
experiment_v2 pipeline - diger modellerden bagimsiz
Adimlar:
  1. videos/ klasoründeki videolari MediaPipe ile isle -> data/
  2. data/ klasorunu augment et -> data_aug/
  3. data_aug/ uzerinde LSTM eğit -> checkpoints/

Kullanim:
  python experiment_v2/pipeline.py
  python experiment_v2/pipeline.py --skip-extract   (sadece aug + train)
  python experiment_v2/pipeline.py --skip-aug       (sadece extract + train)
  python experiment_v2/pipeline.py --only-train     (sadece train)
"""

import sys
import argparse
import shutil
import numpy as np
import json
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
VIDEO_DIR   = BASE / "videos"
DATA_DIR    = BASE / "data"
AUG_DIR     = BASE / "data_aug"
CKPT_DIR    = BASE / "checkpoints"

SEQUENCE_LENGTH = 30
STRIDE          = 3

# ============================================================
# ADIM 1: Video -> MediaPipe -> .npy
# ============================================================

def extract_from_videos():
    import cv2
    import mediapipe as mp

    video_files = (
        list(VIDEO_DIR.glob("*.mp4")) +
        list(VIDEO_DIR.glob("*.avi")) +
        list(VIDEO_DIR.glob("*.mov"))
    )

    if not video_files:
        print(f"[!] Hic video yok: {VIDEO_DIR}")
        print("    Videoları buraya koy: experiment_v2/videos/KELIME.mp4")
        return False

    print(f"\n{'='*60}")
    print(f"ADIM 1: VIDEO -> LANDMARKS")
    print(f"{'='*60}")
    print(f"  {len(video_files)} video bulundu\n")

    mp_holistic = mp.solutions.holistic

    def extract_landmarks(results):
        lm = []
        if results.face_landmarks:
            for p in results.face_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0.0] * 468 * 3)
        if results.pose_landmarks:
            for p in results.pose_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0.0] * 33 * 3)
        if results.left_hand_landmarks:
            for p in results.left_hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0.0] * 21 * 3)
        if results.right_hand_landmarks:
            for p in results.right_hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0.0] * 21 * 3)
        return np.array(lm, dtype=np.float32)

    for vf in video_files:
        label = vf.stem.upper().replace(" ", "_")
        print(f"  [{label}] isleniyor: {vf.name}")

        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"    {fps:.1f} FPS, {total} frame ({total/max(fps,1):.1f}s)")

        all_lm = []
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                all_lm.append(extract_landmarks(results))
        cap.release()

        # Sliding window
        sequences = []
        for i in range(0, len(all_lm) - SEQUENCE_LENGTH + 1, STRIDE):
            seq = np.array(all_lm[i:i + SEQUENCE_LENGTH])
            sequences.append(seq)

        out_dir = DATA_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, seq in enumerate(sequences):
            np.save(out_dir / f"{ts}_{idx:04d}.npy", seq)

        print(f"    {len(sequences)} ornek kaydedildi -> data/{label}/\n")

    return True


# ============================================================
# ADIM 2: Landmark-level augmentation
# ============================================================

FACE_END   = 468 * 3
POSE_END   = FACE_END + 33 * 3
LEFT_END   = POSE_END + 21 * 3
RIGHT_END  = LEFT_END + 21 * 3   # 1629

def _chunks(seq):
    T = len(seq)
    return (
        seq[:, :FACE_END].reshape(T, 468, 3),
        seq[:, FACE_END:POSE_END].reshape(T, 33, 3),
        seq[:, POSE_END:LEFT_END].reshape(T, 21, 3),
        seq[:, LEFT_END:RIGHT_END].reshape(T, 21, 3),
    )

def _pack(T, face, pose, left, right):
    return np.concatenate([
        face.reshape(T, -1),
        pose.reshape(T, -1),
        left.reshape(T, -1),
        right.reshape(T, -1),
    ], axis=1).astype(np.float32)

def aug_noise(s):
    return np.clip(s + np.random.normal(0, 0.004, s.shape).astype(np.float32), 0, 1)

def aug_scale(s, lo=0.92, hi=1.08):
    f = np.random.uniform(lo, hi)
    T = len(s)
    face, pose, left, right = _chunks(s)
    for c in (face, pose, left, right):
        c[:,:,:2] = (c[:,:,:2] - 0.5) * f + 0.5
    return np.clip(_pack(T, face, pose, left, right), 0, 1.5)

def aug_translate(s, mx=0.04):
    dx, dy = np.random.uniform(-mx, mx), np.random.uniform(-mx, mx)
    T = len(s)
    face, pose, left, right = _chunks(s)
    for c in (face, pose, left, right):
        c[:,:,0] += dx
        c[:,:,1] += dy
    return np.clip(_pack(T, face, pose, left, right), 0, 1.5)

def aug_rotate(s, max_deg=8.0):
    rad = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    ca, sa = np.cos(rad), np.sin(rad)
    T = len(s)
    face, pose, left, right = _chunks(s)
    for c in (face, pose, left, right):
        x = c[:,:,0] - 0.5
        y = c[:,:,1] - 0.5
        c[:,:,0] = ca * x - sa * y + 0.5
        c[:,:,1] = sa * x + ca * y + 0.5
    return np.clip(_pack(T, face, pose, left, right), 0, 1.5)

def aug_timewarp(s, lo=0.88, hi=1.12):
    T, F = s.shape
    f = np.random.uniform(lo, hi)
    new_T = max(int(T * f), T // 2)
    old_idx = np.linspace(0, T-1, new_T)
    new_idx = np.linspace(0, new_T-1, T)
    warped = np.stack([np.interp(old_idx, np.arange(T), s[:, j]) for j in range(F)], axis=1)
    return np.stack([np.interp(new_idx, np.arange(new_T), warped[:, j]) for j in range(F)], axis=1).astype(np.float32)

def aug_mirror(s):
    T = len(s)
    face, pose, left, right = _chunks(s.copy())
    for c in (face, pose):
        c[:,:,0] = 1.0 - c[:,:,0]
    left[:,:,0]  = 1.0 - left[:,:,0]
    right[:,:,0] = 1.0 - right[:,:,0]
    return _pack(T, face, pose, right, left)   # sol<->sag swap

AUGS = [aug_noise, aug_scale, aug_translate, aug_rotate, aug_timewarp, aug_mirror]
AUG_NAMES = ["noise", "scale", "translate", "rotate", "timewarp", "mirror"]

def augment_data():
    print(f"\n{'='*60}")
    print("ADIM 2: AUGMENTATION")
    print(f"{'='*60}\n")

    class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[!] data/ klasoru bos: {DATA_DIR}")
        return False

    total_orig = total_new = 0
    for cls_dir in class_dirs:
        files = sorted(cls_dir.glob("*.npy"))
        if not files:
            continue
        out = AUG_DIR / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy2(f, out / f.name)

        new_cnt = 0
        for f in files:
            seq = np.load(f)
            for fn, name in zip(AUGS, AUG_NAMES):
                aug_seq = fn(seq)
                np.save(out / f"{f.stem}_aug_{name}.npy", aug_seq)
                new_cnt += 1

        total_orig += len(files)
        total_new  += new_cnt
        print(f"  {cls_dir.name:15s}: {len(files)} orijinal + {new_cnt} aug = {len(files)+new_cnt}")

    print(f"\n  TOPLAM: {total_orig} -> {total_orig + total_new} ornek")
    return True


# ============================================================
# ADIM 3: Egitim
# ============================================================

def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split

    print(f"\n{'='*60}")
    print("ADIM 3: EGITIM")
    print(f"{'='*60}\n")

    # Dataset
    class SeqDataset(Dataset):
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
            seq = np.load(path).astype(np.float32)
            # Normalize [0,1] araligina
            seq = np.clip(seq, 0, 1)
            return torch.tensor(seq), torch.tensor(label, dtype=torch.long)

    dataset = SeqDataset(AUG_DIR)
    if len(dataset) == 0:
        print("[!] data_aug/ bos, once extract ve augment calistir")
        return

    label_map = dataset.label_map
    num_classes = len(label_map)
    print(f"  Siniflar ({num_classes}): {label_map}")
    print(f"  Toplam ornek: {len(dataset)}")

    # --- Sinif agirliklarini hesapla (az ornekli sinife daha yuksek agirlik) ---
    counts = [0] * num_classes
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    total = sum(counts)
    weights = torch.tensor([total / (num_classes * c) for c in counts], dtype=torch.float32)
    print(f"  Sinif sayilari: { {k: counts[v] for k,v in label_map.items()} }")
    print(f"  Sinif agirliklari: { {k: round(weights[v].item(),2) for k,v in label_map.items()} }")

    # Train/Val split
    val_size   = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
    print(f"  Train: {train_size}, Val: {val_size}\n")

    # Model - hafif GRU
    class GRUModel(nn.Module):
        def __init__(self, input_size=1629, hidden=256, layers=2, classes=5, dropout=0.3):
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
    print(f"  Device: {device}")

    model = GRUModel(classes=num_classes).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parametre sayisi: {params:,}\n")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=8)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc   = 0.0
    best_epoch = 0
    NUM_EPOCHS = 100

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        t_loss = t_correct = t_total = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)

        # Val
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                out = model(seqs)
                v_loss += criterion(out, labels).item()
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        scheduler.step(v_acc)

        if v_acc > best_acc:
            best_acc   = v_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': v_acc,
                'label_map': label_map,
                'num_classes': num_classes,
                'model': 'GRU',
            }, CKPT_DIR / "best_model.pt")
            tag = "  *** BEST ***"
        else:
            tag = ""

        if epoch % 10 == 0 or tag:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train {t_acc:5.1f}% | Val {v_acc:5.1f}% | "
                  f"Loss {v_loss/len(val_loader):.4f}{tag}")

    # Label map kaydet
    with open(CKPT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n  En iyi val accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model kaydedildi: {CKPT_DIR / 'best_model.pt'}")
    print(f"  Label map      : {CKPT_DIR / 'label_map.json'}")
    print(f"\n{'='*60}")
    print("TAMAMLANDI!")
    print(f"{'='*60}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-aug",     action="store_true")
    parser.add_argument("--only-train",   action="store_true")
    args = parser.parse_args()

    if args.only_train:
        train()
        sys.exit(0)

    if not args.skip_extract:
        ok = extract_from_videos()
        if not ok:
            sys.exit(1)

    if not args.skip_aug:
        ok = augment_data()
        if not ok:
            sys.exit(1)

    train()
