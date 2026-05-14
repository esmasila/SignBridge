"""
experiment_v2 icin veri toplama
- 30 frame (pipeline ile ayni)
- experiment_v2/data/KELIME/ klasorune kaydeder
- S: baslat, Q: cikis, E: yeni kelime

Kullanim:
    python experiment_v2/collect.py
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

BASE       = Path(__file__).parent
DATA_DIR   = BASE / "data"
SEQ_LEN    = 30   # pipeline.py ile ayni olmali
TARGET     = 150  # kelime basina kac ornek

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_landmarks(results):
    lm = []
    if results.face_landmarks:
        for p in results.face_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 468 * 3)
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 33 * 3)
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    return np.array(lm, dtype=np.float32)

def main():
    label = input("Kelime (ornek: ABLA): ").strip().upper()
    if not label:
        print("Kelime bos!")
        return

    save_dir = DATA_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)

    # Mevcut ornek sayisi
    existing = len(list(save_dir.glob("*.npy")))
    print(f"\nKelime  : {label}")
    print(f"Klasor  : {save_dir}")
    print(f"Mevcut  : {existing} ornek")
    print(f"Hedef   : {TARGET} yeni ornek")
    print(f"\nKontroller:")
    print(f"  S  : Kaydi baslat")
    print(f"  Q  : Cikis\n")

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        seq      = []
        counter  = 0
        recording = False
        waiting   = True   # S'ye basilmasi bekleniyor

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Landmark ciz
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                       mp.solutions.hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                       mp.solutions.hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                       mp.solutions.pose.POSE_CONNECTIONS)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.putText(frame, f"{label}  {counter}/{TARGET}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)

            if waiting:
                cv2.putText(frame, "S tusuna bas -> kayit baslar",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 200, 255), 2)

            elif recording:
                lm = extract_landmarks(results)
                seq.append(lm)

                # Progress bar
                fill = int(len(seq) / SEQ_LEN * w)
                cv2.rectangle(frame, (0, h - 8), (fill, h), (0, 0, 220), -1)
                cv2.putText(frame, f"KAYIT {len(seq)}/{SEQ_LEN}",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)

                if len(seq) >= SEQ_LEN:
                    arr = np.array(seq, dtype=np.float32)
                    ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    np.save(save_dir / f"{ts}.npy", arr)
                    counter += 1
                    print(f"  Kaydedildi {counter}/{TARGET}: {label}")
                    seq = []

                    if counter >= TARGET:
                        print(f"\nTamamlandi! {TARGET} ornek kaydedildi.")
                        break
                    # Bir sonraki ornek icin hemen devam et
                    # (durma yok, kayit aralıksız devam eder)

            cv2.imshow(f"Veri Toplama - {label}  |  S: baslat  Q: cikis", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                if waiting:
                    waiting   = False
                    recording = True
                    seq       = []
                    print("Kayit basliyor...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nKaydedilen: {counter} yeni ornek -> {save_dir}")
    if counter > 0:
        print(f"\nSimdi egitmek icin calistir:")
        print(f"  python experiment_v2/pipeline.py --skip-extract")

if __name__ == "__main__":
    main()
