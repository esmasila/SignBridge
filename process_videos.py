"""
Masaüstündeki video dosyalarını işle → data_v2'ye ekle
Format: (30, 225) — pose(99) + left_hand(63) + right_hand(63)
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Ayarlar ──────────────────────────────────────────────────────────────────
SEQ_LEN  = 30   # model kaç frame bekliyor
STRIDE   = 3    # sliding window stride (küçük → daha fazla örnek)
DATA_DIR = Path("tid_sequence/data_v2")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = {
    "ANNE"    : r"C:\Users\leven\OneDrive\Desktop\ANNE.mp4",
    "ABLA"    : r"C:\Users\leven\OneDrive\Desktop\ABLAmp4.mp4",
    "BABA"    : r"C:\Users\leven\OneDrive\Desktop\BABA.mp4",
    "ARKADAS" : r"C:\Users\leven\OneDrive\Desktop\ARKADAS.mp4",
    "AKSAM"   : r"C:\Users\leven\OneDrive\Desktop\AKSAM.mp4",
}
# ─────────────────────────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic


def extract_225(results):
    """Pose(99) + sol el(63) + sağ el(63) = 225 özellik"""
    lm = []

    # Pose — 33 nokta × 3 = 99
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 99)

    # Sol el — 21 × 3 = 63
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 63)

    # Sağ el — 21 × 3 = 63
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 63)

    return np.array(lm, dtype=np.float32)  # (225,)


def process_video(label, video_path):
    path = Path(video_path)
    if not path.exists():
        print(f"  HATA: Dosya yok: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{'='*55}")
    print(f"  {label}  —  {path.name}")
    print(f"  {fps:.0f} FPS  |  {total} frame  |  {total/fps:.1f}s")
    print(f"{'='*55}")

    all_lm = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            all_lm.append(extract_225(res))
    cap.release()

    print(f"  Islendi: {len(all_lm)} frame")

    if len(all_lm) < SEQ_LEN:
        # Pad
        pad = [all_lm[-1]] * (SEQ_LEN - len(all_lm))
        sequences = [np.array(all_lm + pad)]
    else:
        # Sliding window
        sequences = []
        for i in range(0, len(all_lm) - SEQ_LEN + 1, STRIDE):
            sequences.append(np.array(all_lm[i:i+SEQ_LEN]))

    label_dir = DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved = 0
    for idx, seq in enumerate(sequences):
        out = label_dir / f"{ts}_{idx:04d}.npy"
        np.save(out, seq)
        saved += 1

    print(f"  KAYDEDILDI: {saved} ornek  ->  {label_dir}")
    return saved


def main():
    print("\nVIDEO ISLEME BASLIYOR")
    print(f"SEQ_LEN={SEQ_LEN}  STRIDE={STRIDE}")

    total = 0
    for label, path in VIDEOS.items():
        total += process_video(label, path)

    print(f"\n{'='*55}")
    print(f"  TOPLAM: {total} ornek eklendi")
    print(f"  Cikti: {DATA_DIR.resolve()}")
    print(f"{'='*55}")

    print("\nKlasor ozeti:")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.npy")))
            print(f"  {d.name:15s}: {n} ornek")

    print("\nEgitim icin:")
    print("  python tid_sequence/scripts/train_v2.py")


if __name__ == "__main__":
    main()
