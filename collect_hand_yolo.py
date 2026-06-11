"""
YOLO El Dedektoru - Veri Toplama
================================
Webcam'den resim cekip, MediaPipe el landmarklarindan otomatik YOLO bbox uretir.

Kullanim:
    python collect_hand_yolo.py

Kontroller:
    S : Kaydi baslat/durdur
    Q : Cikis
    C : Sayaci gosterir

Hedef: 500-1000 resim (hem sol hem sag eli dahil, farkli mesafelerde)

Cikti:
    hand_yolo_training/images/frame_XXXX.jpg
    hand_yolo_training/labels/frame_XXXX.txt  (YOLO format: 0 cx cy w h)
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path("C:/Projects/sign_bridge/hand_yolo_training")
IMG_DIR = BASE / "images"
LBL_DIR = BASE / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 1000  # Hedef resim sayisi
PAD_RATIO = 0.15  # Bbox etrafina %15 pad (ellerin cercevede kalmasi icin)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def get_hand_bbox(landmarks, img_w, img_h, pad_ratio=PAD_RATIO):
    """El landmarklari -> YOLO bbox (normalized cx, cy, w, h)"""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = x2 - x1
    h = y2 - y1
    # Pad ekle
    x1 = max(0, x1 - w * pad_ratio)
    y1 = max(0, y1 - h * pad_ratio)
    x2 = min(1, x2 + w * pad_ratio)
    y2 = min(1, y2 + h * pad_ratio)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1
    return cx, cy, bw, bh

def main():
    # Mevcut sayiyi kontrol et
    existing = len(list(IMG_DIR.glob("*.jpg")))
    print("=" * 60)
    print("YOLO El Dedektoru - Veri Toplama")
    print("=" * 60)
    print(f"Klasor   : {BASE}")
    print(f"Mevcut   : {existing} resim")
    print(f"Hedef    : {TARGET} resim")
    print()
    print("Kontroller:")
    print("  S : Kaydi baslat/durdur")
    print("  Q : Cikis")
    print()
    print("IPUCLARI:")
    print("  - Farkli isaretler yap (MERHABA, BABA, EV, YEMEK vb.)")
    print("  - Elini farkli mesafe/acilarda tut")
    print("  - Hem tek el hem iki el kullan")
    print("  - Kameraya yakin/uzak, sag/sol yerlerde dur")
    print()
    input("ENTER ile basla...")

    cap = cv2.VideoCapture(0)
    count = existing
    recording = False
    skip_frame = 0  # Her 3 frame'de 1 kayit (cesitlilik icin)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display = frame.copy()
            num_hands = 0
            bboxes = []

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                for hand_lm in results.multi_hand_landmarks:
                    cx, cy, bw, bh = get_hand_bbox(hand_lm, w, h)
                    bboxes.append((cx, cy, bw, bh))
                    # Gorsellestir
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = (0, 255, 0) if recording else (255, 200, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    mp_draw.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Kayit yap
            if recording and num_hands > 0:
                skip_frame += 1
                if skip_frame >= 3:  # Her 3 frame'de 1 kaydet
                    skip_frame = 0
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    img_name = f"frame_{ts}.jpg"
                    lbl_name = f"frame_{ts}.txt"

                    cv2.imwrite(str(IMG_DIR / img_name), frame)
                    with open(LBL_DIR / lbl_name, "w") as f:
                        for cx, cy, bw, bh in bboxes:
                            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    count += 1

            # UI
            status = "KAYIT" if recording else "BEKLIYOR"
            status_color = (0, 255, 0) if recording else (0, 165, 255)
            cv2.rectangle(display, (0, 0), (w, 40), (30, 30, 30), -1)
            cv2.putText(display, f"[{status}]  Resim: {count}/{TARGET}  El: {num_hands}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display, "S: Baslat/Dur   Q: Cikis",
                        (w - 320, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Progress bar
            pct = min(count / TARGET, 1.0)
            bar_w = int(pct * w)
            cv2.rectangle(display, (0, h - 8), (bar_w, h), (0, 255, 0), -1)

            cv2.imshow("YOLO El Dedektoru - Veri Toplama", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                recording = not recording
                print(f"{'BASLATILDI' if recording else 'DURDURULDU'}  |  Resim: {count}")

            if count >= TARGET:
                print(f"\nHEDEFE ULASILDI: {count} resim")
                recording = False

    cap.release()
    cv2.destroyAllWindows()

    # Train/val split icin data.yaml yaz
    final_count = len(list(IMG_DIR.glob("*.jpg")))
    yaml_content = f"""path: {BASE.as_posix()}
train: images
val: images
nc: 1
names:
- hand
"""
    (BASE / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"TAMAMLANDI: {final_count} resim")
    print(f"Konum     : {BASE}")
    print(f"data.yaml : {BASE / 'data.yaml'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
