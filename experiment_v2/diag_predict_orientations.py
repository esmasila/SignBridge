"""
Kayitli 15 mobil JPEG'ini farkli oryantasyonlarla isleyip
modele ver — hangi varyant dogru kelimeyi cikariyor?
"""
import cv2, os, glob, json
import numpy as np
import torch, torch.nn as nn
import mediapipe as mp
from collections import deque

BASE = os.path.dirname(__file__)
INC  = os.path.join(BASE, "debug_incoming")
CKPT = os.path.join(BASE, "checkpoints", "best_model.pt")

# ---- Layout ----
FACE_END  = 468 * 3
POSE_END  = FACE_END + 33*3
LEFT_END  = POSE_END + 21*3
RIGHT_END = LEFT_END + 21*3
HAND_VEL  = 21*3 + 21*3
INPUT_SIZE = RIGHT_END + HAND_VEL
SEQ_LEN = 30

class GRUModel(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, 256, 2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.0), nn.Linear(128, classes))
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
label_map = ckpt["label_map"]; idx_to_label = {v:k for k,v in label_map.items()}
model = GRUModel(classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

def extract(res):
    lm = []
    if res.face_landmarks:
        for p in res.face_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else: lm.extend([0.0]*468*3)
    if res.pose_landmarks:
        for p in res.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else: lm.extend([0.0]*33*3)
    if res.left_hand_landmarks:
        for p in res.left_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else: lm.extend([0.0]*21*3)
    if res.right_hand_landmarks:
        for p in res.right_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else: lm.extend([0.0]*21*3)
    return np.array(lm, dtype=np.float32)

def normalize(lm):
    result = lm.copy()
    pose = result[FACE_END:POSE_END].reshape(33, 3)
    ls, rs = pose[11].copy(), pose[12].copy()
    if not (np.allclose(ls, 0) and np.allclose(rs, 0)):
        center = (ls + rs) / 2.0 if not (np.allclose(ls,0) or np.allclose(rs,0)) else (ls if np.allclose(rs,0) else rs)
        scale = float(np.linalg.norm(rs - ls))
        if scale >= 1e-4:
            all_lm = result.reshape(-1, 3)
            all_lm = (all_lm - center) / scale
            result = all_lm.flatten()
    lh = result[POSE_END:LEFT_END].reshape(21, 3)
    if not np.allclose(lh[0], 0):
        wrist = lh[0].copy()
        ps = np.linalg.norm(lh[9] - lh[0])
        if ps > 1e-4:
            lh[1:] = (lh[1:] - wrist) / ps
            result[POSE_END:LEFT_END] = lh.flatten()
    rh = result[LEFT_END:RIGHT_END].reshape(21, 3)
    if not np.allclose(rh[0], 0):
        wrist = rh[0].copy()
        ps = np.linalg.norm(rh[9] - rh[0])
        if ps > 1e-4:
            rh[1:] = (rh[1:] - wrist) / ps
            result[LEFT_END:RIGHT_END] = rh.flatten()
    return result

def add_velocity(seq):
    hands = seq[:, POSE_END:]
    vel = np.zeros_like(hands)
    vel[1:] = hands[1:] - hands[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)

def predict_top3(frames_arr):
    """frames_arr: list of (1629,) -> top3 from model"""
    while len(frames_arr) < SEQ_LEN:
        frames_arr.append(frames_arr[-1] if frames_arr else np.zeros(RIGHT_END))
    seq = np.array(frames_arr[-SEQ_LEN:], dtype=np.float32)
    seq = add_velocity(seq)
    seq = np.clip(seq, -5, 5)
    inp = torch.tensor(seq).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        prob = torch.softmax(out, dim=1)[0]
        v, i = prob.topk(5)
    return [(idx_to_label[int(i_)], float(v_)) for v_, i_ in zip(v, i)]

mp_hol = mp.solutions.holistic

def process_variant(transform, name):
    """transform: cv2 image -> cv2 image"""
    files = sorted(glob.glob(os.path.join(INC, "*.jpg")))
    hol = mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                           min_detection_confidence=0.3, min_tracking_confidence=0.3,
                           refine_face_landmarks=False, smooth_landmarks=True)
    frames = []
    hand_count = 0
    for fp in files:
        img = cv2.imread(fp)
        img = transform(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hol.process(rgb)
        if res.left_hand_landmarks or res.right_hand_landmarks:
            hand_count += 1
        lm = extract(res)
        lm = normalize(lm)
        frames.append(lm)
    top3 = predict_top3(frames)
    print(f"\n[{name}]  el bulunan frame={hand_count}/{len(files)}")
    for label, prob in top3:
        print(f"   {label:20s} {prob*100:5.1f}%")

print("="*60)
print("Mobil kayitli 15 JPEG -> farkli oryantasyon -> model tahmini")
print("="*60)

# Identitiy: portre olarak gonder (mobil simdi boyle yapiyor)
process_variant(lambda x: x, "1) ORIJINAL portre 480x720")

# Landscape CW
process_variant(lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                "2) 90 CW landscape 720x480")

# Landscape CCW
process_variant(lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
                "3) 90 CCW landscape 720x480")

# Center-crop kare
def cc(x):
    h, w = x.shape[:2]
    cy = (h - w) // 2
    return x[cy:cy+w, :]
process_variant(cc, "4) CENTER-CROP kare 480x480")

# Warp 640x480 (training ratio)
process_variant(lambda x: cv2.resize(x, (640, 480)), "5) WARP 640x480 (egitim orani)")

# Pad to 4:3 landscape: 480x720 -> 960x720 (left/right black pad) -> resize 640x480
def pad43(x):
    h, w = x.shape[:2]  # 720x480
    # 4:3 landscape: w_target = h * 4/3 = 960
    target_w = int(h * 4 / 3)
    if target_w > w:
        pad = (target_w - w) // 2
        padded = cv2.copyMakeBorder(x, 0, 0, pad, target_w - w - pad, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = x
    return cv2.resize(padded, (640, 480))
process_variant(pad43, "6) PAD-TO-4:3 sonra 640x480")
