"""
Ekran Kaydedici + Otomatik MediaPipe İşleme
YouTube veya sözlük sitesindeki işaret videolarını yakala → dataset'e ekle
"""

import cv2
import numpy as np
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# mss yoksa yükle
try:
    import mss
except ImportError:
    print("mss kuruluyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mss"])
    import mss

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("⚠️  MediaPipe bulunamadı — sadece video kaydedilecek")

# ─── AYARLAR ─────────────────────────────────────────────────────────────────
COUNTDOWN     = 3      # kayıt öncesi geri sayım (sn)
RECORD_SEC    = 2.5    # kayıt süresi (sn)
FPS_TARGET    = 30     # hedef FPS
SEQUENCE_LEN  = 30     # model kaç frame bekliyor

OUTPUT_VIDEO  = Path("tid_sequence/videos")
OUTPUT_DATA   = Path("tid_sequence/data_v2")
OUTPUT_VIDEO.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


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


def record_screen(label: str, sample_idx: int):
    """Ekranı kaydeder, video olarak kaydeder, mediapipe ile işler."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = OUTPUT_VIDEO / f"{label}_{timestamp}.mp4"

    # ── Geri sayım ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Label : {label}  |  Örnek #{sample_idx}")
    print(f"{'='*55}")
    for i in range(COUNTDOWN, 0, -1):
        print(f"  Kayıt başlıyor: {i}...", end="\r", flush=True)
        time.sleep(1)
    print(f"  🔴 KAYIT YAPILIYOR ({RECORD_SEC:.1f}s)...          ")

    # ── Ekran kaydı ─────────────────────────────────────────────────────────
    frames_bgr = []
    with mss.mss() as sct:
        monitor = sct.monitors[1]          # birinci monitör
        frame_interval = 1.0 / FPS_TARGET
        start = time.time()
        next_capture = start

        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= RECORD_SEC:
                break
            if now >= next_capture:
                img = np.array(sct.grab(monitor))       # BGRA
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                frames_bgr.append(bgr)
                next_capture += frame_interval
            else:
                time.sleep(0.001)

    print(f"  ✅ {len(frames_bgr)} frame kaydedildi")

    if not frames_bgr:
        print("  ❌ Hiç frame alınamadı!")
        return False

    h, w = frames_bgr[0].shape[:2]

    # ── Video'ya yaz ────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS_TARGET, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    print(f"  💾 Video kaydedildi: {video_path.name}")

    # ── MediaPipe ile işle ──────────────────────────────────────────────────
    if not MP_AVAILABLE:
        print("  ⚠️  MediaPipe yok — video kaydedildi, manuel işleme gerekli")
        return True

    print("  🔄 MediaPipe işleniyor...")
    mp_holistic = mp.solutions.holistic

    all_lm = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            all_lm.append(extract_landmarks(results))

    # SEQUENCE_LEN'e göre kes ya da doldur
    if len(all_lm) >= SEQUENCE_LEN:
        # ortayı al (başı + sonu kırp)
        mid = len(all_lm) // 2
        start_i = mid - SEQUENCE_LEN // 2
        seq = all_lm[start_i: start_i + SEQUENCE_LEN]
    else:
        pad = [all_lm[-1]] * (SEQUENCE_LEN - len(all_lm))
        seq = all_lm + pad

    seq_np = np.array(seq, dtype=np.float32)

    # Kaydet
    label_dir = OUTPUT_DATA / label.upper()
    label_dir.mkdir(parents=True, exist_ok=True)
    save_path = label_dir / f"{timestamp}_{sample_idx:03d}.npy"
    np.save(save_path, seq_np)
    print(f"  ✅ Landmark kaydedildi: {save_path.name}  shape={seq_np.shape}")
    return True


def main():
    print("\n" + "="*55)
    print("  EKRAN KAYDEDİCİ — İşaret Dili Dataset Toplayıcı")
    print("="*55)
    print(f"  Kayıt süresi  : {RECORD_SEC}s")
    print(f"  FPS           : {FPS_TARGET}")
    print(f"  Sequence len  : {SEQUENCE_LEN} frame")
    print(f"  Video çıktı   : {OUTPUT_VIDEO}")
    print(f"  Data çıktı    : {OUTPUT_DATA}")
    print("="*55)

    while True:
        print()
        label = input("  Kelime girin (çıkmak için q): ").strip().upper()
        if label.lower() == "q" or label == "":
            break

        try:
            n = int(input(f"  '{label}' için kaç örnek? [varsayılan 3]: ").strip() or "3")
        except ValueError:
            n = 3

        print(f"\n  ➡️  Şimdi tarayıcıda '{label}' videosunu hazırla/oynat!")
        print("  (Her kayıt öncesi 3 saniye geri sayım olacak)\n")

        for i in range(n):
            ok = record_screen(label, i)
            if not ok:
                print("  Örnek alınamadı, tekrar dene.")
            if i < n - 1:
                input(f"  Sonraki örnek için Enter'a bas ({i+2}/{n})...")

        print(f"\n  ✅ '{label}' için {n} örnek tamamlandı!")
        print(f"  📁 {OUTPUT_DATA / label}")

    print("\n🎉 Bitti! Şimdi train edebilirsin.")
    print(f"   python tid_sequence/scripts/train_v2.py\n")


if __name__ == "__main__":
    main()
