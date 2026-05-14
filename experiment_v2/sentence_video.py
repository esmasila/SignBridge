"""
Video dosyasini cumleye cevir

Kullanim:
  python experiment_v2/sentence_video.py video.mp4
  python experiment_v2/sentence_video.py video.mp4 --show   (ekranda goster)
  python experiment_v2/sentence_video.py video.mp4 --out sonuc.txt
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from collections import deque

BASE    = Path(__file__).parent
CKPT    = BASE / "checkpoints" / "best_model.pt"
SEQ_LEN = 30

# ---- Model ----
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

ckpt         = torch.load(CKPT, map_location="cpu")
label_map    = ckpt["label_map"]
num_classes  = ckpt["num_classes"]
idx_to_label = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GRUModel(classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

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


def translate_video(video_path: str, show: bool = False, out_file: str = None,
                    conf_threshold: float = 0.80, min_word_frames: int = 8):
    """
    Video dosyasini isle ve kelime dizisi olarak dondur.

    Algoritma:
      - Her frame icin landmark cikar
      - Sliding window (SEQ_LEN=30) ile tahmin yap
      - Ayni kelime min_word_frames kez ust uste gelirse kelime onaylanir
      - Onaylanan kelimeler cumleye eklenir, tekrar engellenir (cooldown: 30 frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Video acilamadi: {video_path}")
        return []

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo   : {video_path}")
    print(f"FPS     : {fps:.1f}  |  Toplam frame: {total}  ({total/fps:.1f}s)")
    print(f"Kelimeler: {list(label_map.keys())}\n")

    buffer        = deque(maxlen=SEQ_LEN)
    smooth_counts = {k: 0 for k in label_map}
    sentence      = []
    cooldown      = 0          # kelime eklendikten sonra bekleme (frame sayisi)
    COOLDOWN_LEN  = int(fps * 0.8)   # ~0.8 saniye

    frame_idx = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            lm      = extract_landmarks(results)
            buffer.append(lm)

            hand_visible = (results.left_hand_landmarks is not None or
                            results.right_hand_landmarks is not None)

            pred_label = None
            pred_conf  = 0.0

            if len(buffer) == SEQ_LEN:
                seq = np.clip(np.array(buffer, dtype=np.float32), 0, 1)
                inp = torch.tensor(seq).unsqueeze(0).to(device)

                with torch.no_grad():
                    out  = model(inp)
                    prob = torch.softmax(out, dim=1)[0]
                    conf, idx = prob.max(0)
                    pred_conf  = conf.item()
                    pred_label = idx_to_label[idx.item()]

                if cooldown > 0:
                    cooldown -= 1
                    smooth_counts = {k: 0 for k in label_map}

                elif pred_conf >= conf_threshold and hand_visible:
                    smooth_counts[pred_label] += 1
                    for k in smooth_counts:
                        if k != pred_label:
                            smooth_counts[k] = max(0, smooth_counts[k] - 1)

                    if smooth_counts[pred_label] >= min_word_frames:
                        sentence.append(pred_label)
                        cooldown = COOLDOWN_LEN
                        smooth_counts = {k: 0 for k in label_map}
                        ts = frame_idx / fps
                        print(f"  [{ts:6.2f}s] Kelime: {pred_label:15s}  ({pred_conf*100:.0f}%)")
                else:
                    if not hand_visible:
                        smooth_counts = {k: 0 for k in label_map}

            # Gorsel goster
            if show and pred_label:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
                color = (0, 220, 0) if pred_conf >= conf_threshold else (100, 100, 100)
                cv2.putText(frame, f"{pred_label}  %{pred_conf*100:.0f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
                # Cumle
                cv2.rectangle(frame, (0, h - 55), (w, h), (15, 15, 40), -1)
                cv2.putText(frame, " ".join(sentence) if sentence else "(bos)",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

                # Progress
                prog = int(frame_idx / max(total, 1) * w)
                cv2.rectangle(frame, (0, h - 58), (prog, h - 54), (0, 180, 100), -1)

                cv2.imshow("Video Cevirisi  |  Q: cikis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  ... {frame_idx}/{total} frame islendi", end="\r")

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return sentence


def main():
    parser = argparse.ArgumentParser(description="Video -> Cumle cevirici")
    parser.add_argument("video", help="Video dosyasi yolu")
    parser.add_argument("--show",  action="store_true", help="Isleme sirasinda goster")
    parser.add_argument("--out",   default=None,        help="Sonucu dosyaya yaz")
    parser.add_argument("--conf",  type=float, default=0.80, help="Confidence esigi (varsayilan 0.80)")
    args = parser.parse_args()

    sentence = translate_video(
        video_path     = args.video,
        show           = args.show,
        conf_threshold = args.conf,
    )

    result = " ".join(sentence) if sentence else "(hicbir kelime taninamadi)"

    print(f"\n{'='*60}")
    print(f"SONUC: {result}")
    print(f"{'='*60}\n")

    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"Dosyaya yazildi: {args.out}")


if __name__ == "__main__":
    main()
