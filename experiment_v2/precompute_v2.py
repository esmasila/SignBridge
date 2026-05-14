"""
V2 icin on-isleme:
  data_aug/<class>/*.npy  (30, 1629)
    -> her frame normalize edilir (omuz + bilek)
    -> velocity eklenir
    -> clip[-5, 5]
  data_aug_v2/<class>/*.npy  (30, 1755)

Bu bir KERE calisir. Egitim sonra 10x hizli olur (sadece tensor load + GRU).
"""

import sys, io, time
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

BASE    = Path(__file__).parent
SRC_DIR = BASE / "data_aug"
DST_DIR = BASE / "data_aug_v2"

FACE_END  = 468 * 3
POSE_END  = FACE_END + 33*3
LEFT_END  = POSE_END + 21*3
RIGHT_END = LEFT_END + 21*3

# ---- Vektorize normalize (30 frame tek seferde) ----
def normalize_seq_vec(seq):
    """seq (30, 1629) -> normalized (30, 1629)"""
    out = seq.copy()

    # KATMAN 1: omuz-merkezli (her frame icin)
    pose = out[:, FACE_END:POSE_END].reshape(-1, 33, 3)      # (30, 33, 3)
    ls   = pose[:, 11]                                         # (30, 3)
    rs   = pose[:, 12]                                         # (30, 3)

    ls_zero = np.all(ls == 0, axis=1)                          # (30,)
    rs_zero = np.all(rs == 0, axis=1)
    both_zero = ls_zero & rs_zero

    # center: her frame icin
    center = np.where(ls_zero[:, None], rs,
             np.where(rs_zero[:, None], ls, (ls + rs) / 2.0))  # (30, 3)
    scale  = np.linalg.norm(rs - ls, axis=1)                   # (30,)

    valid = (~both_zero) & (scale >= 1e-4)                     # (30,)

    if valid.any():
        all_lm = out.reshape(30, -1, 3)                         # (30, 543, 3)
        c = center[:, None, :]                                  # (30, 1, 3)
        s = scale[:, None, None]                                # (30, 1, 1)
        s = np.where(s < 1e-8, 1.0, s)
        normalized = (all_lm - c) / s
        # sadece valid frame'lerde uygula
        all_lm = np.where(valid[:, None, None], normalized, all_lm)
        out = all_lm.reshape(30, -1)

    # KATMAN 2: sol el bilek-merkezli
    for hand_start, hand_end in [(POSE_END, LEFT_END), (LEFT_END, RIGHT_END)]:
        hand = out[:, hand_start:hand_end].reshape(30, 21, 3)   # (30, 21, 3)
        wrist = hand[:, 0].copy()                               # (30, 3)
        wrist_zero = np.all(wrist == 0, axis=1)
        palm_scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)  # (30,)

        valid_h = (~wrist_zero) & (palm_scale > 1e-4)
        if valid_h.any():
            w = wrist[:, None, :]                               # (30, 1, 3)
            ps = np.where(palm_scale < 1e-8, 1.0, palm_scale)[:, None, None]
            fingers = (hand[:, 1:] - w) / ps
            hand[:, 1:] = np.where(valid_h[:, None, None], fingers, hand[:, 1:])
            out[:, hand_start:hand_end] = hand.reshape(30, -1)

    return out

def add_velocity(seq):
    hands = seq[:, POSE_END:]
    vel   = np.zeros_like(hands)
    vel[1:] = hands[1:] - hands[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)

# ---- Calistir ----
DST_DIR.mkdir(exist_ok=True)
classes = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])
print(f"Siniflar: {len(classes)}")

total_files = 0
t0 = time.time()
for ci, cls in enumerate(classes, 1):
    dst_cls = DST_DIR / cls.name
    dst_cls.mkdir(exist_ok=True)
    files = list(cls.glob("*.npy"))
    for f in files:
        seq = np.load(f).astype(np.float32)      # (30, 1629)
        seq = normalize_seq_vec(seq)
        seq = add_velocity(seq)                  # (30, 1755)
        seq = np.clip(seq, -5, 5)
        np.save(dst_cls / f.name, seq)
        total_files += 1
    elapsed = time.time() - t0
    rate = total_files / elapsed if elapsed > 0 else 0
    remaining = len(classes) - ci
    eta = remaining * (elapsed / ci) if ci > 0 else 0
    print(f"[{ci:3d}/{len(classes)}] {cls.name:20s} | {len(files):5d} dosya | toplam {total_files:6d} | {rate:.0f} f/s | ETA {eta:5.0f}s")

print(f"\nTAMAMLANDI! {total_files} dosya | {time.time()-t0:.1f}s")
print(f"Kaynak: {DST_DIR}")
