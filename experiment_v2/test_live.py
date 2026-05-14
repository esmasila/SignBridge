"""
Canli webcam testi - experiment_v2 modeli
Calistir: python experiment_v2/test_live.py
Cikis: Q tusuna bas
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
from collections import deque

BASE   = Path(__file__).parent
CKPT   = BASE / "checkpoints" / "best_model.pt"
SEQ_LEN = 30   # pipeline.py ile ayni olmali

# ---- Model tanimi (pipeline.py ile ayni) ----
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

# ---- Yukle ----
ckpt = torch.load(CKPT, map_location="cpu")
label_map   = ckpt["label_map"]           # {"ABLA": 0, "ANNE": 1, ...}
num_classes = ckpt["num_classes"]
idx_to_label = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"Model yuklendi | Siniflar: {list(label_map.keys())} | Device: {device}")

# ---- MediaPipe ----
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

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

# ---- Ana dongu ----
cap     = cv2.VideoCapture(0)
buffer  = deque(maxlen=SEQ_LEN)
pred_label    = "Bekliyor..."
pred_conf     = 0.0
smooth_counts = {k: 0 for k in label_map}   # smoothing icin
SMOOTH_NEEDED  = 6       # kac ardisik frame ayni tahmin gelmeli
CONF_THRESHOLD = 0.80    # dusuk guvenilirlikte gosterme

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

        lm = extract_landmarks(results)
        buffer.append(lm)

        # Buffer dolunca tahmin yap
        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer, dtype=np.float32)
            seq = np.clip(seq, 0, 1)
            inp = torch.tensor(seq).unsqueeze(0).to(device)

            with torch.no_grad():
                out  = model(inp)
                prob = torch.softmax(out, dim=1)[0]
                conf, idx = prob.max(0)
                conf  = conf.item()
                label = idx_to_label[idx.item()]

            # El gorunmuyorsa tahmin yapma
            hand_visible = (results.left_hand_landmarks is not None or
                            results.right_hand_landmarks is not None)

            if conf >= CONF_THRESHOLD and hand_visible:
                smooth_counts[label] += 1
                for k in smooth_counts:
                    if k != label:
                        smooth_counts[k] = max(0, smooth_counts[k] - 1)
                if smooth_counts[label] >= SMOOTH_NEEDED:
                    pred_label = label
                    pred_conf  = conf
            else:
                # El yoksa veya dusuk guven -> tum sayaclari sifirla
                for k in smooth_counts:
                    smooth_counts[k] = 0
                pred_label = "El gosteriniz..." if not hand_visible else "?"
                pred_conf  = conf

        # Ekranda goster
        h, w = frame.shape[:2]
        bar_w = int(pred_conf * 300)

        # Arkaplan kutusu
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)

        # Buffer doluluk cubugu (mavi)
        fill = int(len(buffer) / SEQ_LEN * w)
        cv2.rectangle(frame, (0, 75), (fill, 80), (200, 150, 0), -1)

        # Tahmin yazisi
        color = (0, 220, 0) if pred_conf >= CONF_THRESHOLD else (0, 150, 220)
        cv2.putText(frame, f"{pred_label}",
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f"{pred_conf*100:.0f}%",
                    (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Confidence bar (yesil)
        cv2.rectangle(frame, (0, 60), (bar_w, 72), color, -1)

        # Sinif listesi (sag ust)
        for i, (lbl, cidx) in enumerate(label_map.items()):
            is_pred = (lbl == pred_label)
            c = (0, 220, 0) if is_pred else (180, 180, 180)
            cv2.putText(frame, lbl, (w - 140, 110 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        cv2.imshow("SignBridge - experiment_v2  |  Q: cikis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
