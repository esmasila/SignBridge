"""
Gercek zamanli cumle olusturucu - webcam
Isaret yap -> kelime tanınır -> cumleye eklenir

Kontroller:
  C     : Cumleyi temizle
  Space : Mevcut kelimeyi hemen ekle
  Q     : Cikis
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
import time

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

print(f"Model: {list(label_map.keys())} | Device: {device}")

# ---- MediaPipe ----
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

# ---- Cumle durumu ----
CONF_THRESHOLD = 0.80
SMOOTH_NEEDED  = 8       # kac ardisik frame ayni tahmin -> kelime onaylanir
COOLDOWN_SEC   = 1.2     # kelime eklendikten sonra bekleme suresi (saniye)
MAX_WORDS      = 12      # cümlede max kelime

sentence      = []       # onaylanan kelimeler listesi
smooth_counts = {k: 0 for k in label_map}
confirmed_label = None
last_add_time   = 0.0
current_pred    = "..."
current_conf    = 0.0

cap    = cv2.VideoCapture(0)
buffer = deque(maxlen=SEQ_LEN)

def draw_sentence(frame, sentence, current_pred, current_conf, h, w):
    """Alt kisma cumleyi, ust kisma anlık tahmini ciz"""
    # Ust bar - anlik tahmin
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    hand_color = (0, 220, 0) if current_conf >= CONF_THRESHOLD else (100, 100, 100)
    cv2.putText(frame, current_pred, (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, hand_color, 3)
    cv2.putText(frame, f"%{current_conf*100:.0f}", (w - 90, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, hand_color, 2)

    # Confidence bar
    bar = int(current_conf * (w - 20))
    cv2.rectangle(frame, (10, 58), (10 + bar, 66), hand_color, -1)

    # Alt bar - cumle
    cv2.rectangle(frame, (0, h - 80), (w, h), (15, 15, 40), -1)
    cv2.putText(frame, "CUMLE:", (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1)

    sentence_text = " ".join(sentence) if sentence else "(bos)"
    # Uzun cumleyi kisalt
    if len(sentence_text) > 55:
        sentence_text = "..." + sentence_text[-52:]

    cv2.putText(frame, sentence_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Kontroller
    cv2.putText(frame, "C:Temizle  Space:Ekle  Q:Cikis",
                (w // 2 - 140, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

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

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Landmark ciz
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                   mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                   mp.solutions.hands.HAND_CONNECTIONS)

        lm = extract_landmarks(results)
        buffer.append(lm)

        hand_visible = (results.left_hand_landmarks is not None or
                        results.right_hand_landmarks is not None)

        now = time.time()
        in_cooldown = (now - last_add_time) < COOLDOWN_SEC

        if len(buffer) == SEQ_LEN:
            seq = np.clip(np.array(buffer, dtype=np.float32), 0, 1)
            inp = torch.tensor(seq).unsqueeze(0).to(device)

            with torch.no_grad():
                out  = model(inp)
                prob = torch.softmax(out, dim=1)[0]
                conf, idx = prob.max(0)
                conf  = conf.item()
                label = idx_to_label[idx.item()]

            current_pred = label
            current_conf = conf

            if conf >= CONF_THRESHOLD and hand_visible and not in_cooldown:
                smooth_counts[label] += 1
                for k in smooth_counts:
                    if k != label:
                        smooth_counts[k] = max(0, smooth_counts[k] - 1)

                if smooth_counts[label] >= SMOOTH_NEEDED:
                    # Kelimeyi cumleye ekle
                    if len(sentence) < MAX_WORDS:
                        sentence.append(label)
                    else:
                        sentence.pop(0)
                        sentence.append(label)
                    last_add_time = now
                    smooth_counts = {k: 0 for k in label_map}
                    print(f"  + {label:15s} | Cumle: {' '.join(sentence)}")
            else:
                if not hand_visible or in_cooldown:
                    smooth_counts = {k: 0 for k in label_map}
        else:
            current_pred = f"Dolduruluyor {len(buffer)}/{SEQ_LEN}"
            current_conf = 0.0

        # Cooldown gostergesi
        if in_cooldown:
            remaining = COOLDOWN_SEC - (now - last_add_time)
            cv2.rectangle(frame, (0, 68), (int(remaining / COOLDOWN_SEC * frame.shape[1]), 72),
                          (0, 180, 255), -1)

        h, w = frame.shape[:2]
        draw_sentence(frame, sentence, current_pred, current_conf, h, w)
        cv2.imshow("SignBridge - Cumle Modu", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            sentence = []
            smooth_counts = {k: 0 for k in label_map}
            print("  Cumle temizlendi")
        elif key == ord(' '):
            # Mevcut tahmini hemen ekle
            if current_conf >= CONF_THRESHOLD and current_pred not in ("...", "") and len(sentence) < MAX_WORDS:
                sentence.append(current_pred)
                last_add_time = time.time()
                smooth_counts = {k: 0 for k in label_map}
                print(f"  + {current_pred} (manuel) | Cumle: {' '.join(sentence)}")

cap.release()
cv2.destroyAllWindows()
print(f"\nSon cumle: {' '.join(sentence)}")
