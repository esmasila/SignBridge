"""
Aynisini ama eventlet monkey_patch SONRASI calistir.
Eger eller bulunmuyorsa => eventlet/holistic conflict.
"""
import eventlet
eventlet.monkey_patch()

import cv2, os, glob
import numpy as np
import mediapipe as mp

BASE = os.path.dirname(__file__)
INC  = os.path.join(BASE, "debug_incoming")

mp_hol = mp.solutions.holistic

# Server'in WebSocket session'da kullandigi AYNI parametreler
hol = mp_hol.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    smooth_landmarks=True,
    refine_face_landmarks=False,
)

files = sorted(glob.glob(os.path.join(INC, "*.jpg")))
stats = {"face": 0, "pose": 0, "L": 0, "R": 0}

print(f"[EVENTLET] {len(files)} JPEG islenir...")
for fp in files:
    img = cv2.imread(fp)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hol.process(rgb)
    f = res.face_landmarks is not None
    p = res.pose_landmarks is not None
    L = res.left_hand_landmarks is not None
    R = res.right_hand_landmarks is not None
    if f: stats["face"] += 1
    if p: stats["pose"] += 1
    if L: stats["L"] += 1
    if R: stats["R"] += 1
    print(f"  {os.path.basename(fp)}: face={int(f)} pose={int(p)} L={int(L)} R={int(R)}")

n = max(1, len(files))
print(f"\nSONUC: face={stats['face']}/{n} pose={stats['pose']}/{n} L={stats['L']}/{n} R={stats['R']}/{n}")
