"""
Mobilden gelen kayitli JPEG'lere MediaPipe Holistic uygula,
landmark var mi kontrol et + skeleton overlay'i kaydet.

Boylece sorun:
  (a) MediaPipe gercekten elleri bulamiyor mu (icerik problemi)
  (b) Yoksa server pipeline'inda baska bir bozukluk mu
ayirt edilebilir.
"""
import cv2, os, glob
import numpy as np
import mediapipe as mp

BASE = os.path.dirname(__file__)
INC  = os.path.join(BASE, "debug_incoming")
OUT  = os.path.join(BASE, "debug_overlay")
os.makedirs(OUT, exist_ok=True)

mp_hol = mp.solutions.holistic
mp_drw = mp.solutions.drawing_utils

hol = mp_hol.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

files = sorted(glob.glob(os.path.join(INC, "*.jpg")))
print(f"Toplam {len(files)} JPEG bulundu.\n")

stats = {"face": 0, "pose": 0, "left_hand": 0, "right_hand": 0}

for fp in files:
    img = cv2.imread(fp)
    if img is None:
        print(f"  HATA: {fp}")
        continue
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hol.process(rgb)

    has_face  = res.face_landmarks is not None
    has_pose  = res.pose_landmarks is not None
    has_left  = res.left_hand_landmarks is not None
    has_right = res.right_hand_landmarks is not None

    if has_face:  stats["face"]  += 1
    if has_pose:  stats["pose"]  += 1
    if has_left:  stats["left_hand"]  += 1
    if has_right: stats["right_hand"] += 1

    name = os.path.basename(fp)
    print(f"{name}  size={w}x{h}  face={int(has_face)} pose={int(has_pose)} "
          f"L={int(has_left)} R={int(has_right)}")

    # Overlay
    overlay = img.copy()
    if has_face:
        mp_drw.draw_landmarks(overlay, res.face_landmarks,
                              mp_hol.FACEMESH_TESSELATION,
                              landmark_drawing_spec=None,
                              connection_drawing_spec=mp_drw.DrawingSpec(
                                  color=(80,80,80), thickness=1))
    if has_pose:
        mp_drw.draw_landmarks(overlay, res.pose_landmarks,
                              mp_hol.POSE_CONNECTIONS,
                              connection_drawing_spec=mp_drw.DrawingSpec(
                                  color=(0,255,0), thickness=2))
    if has_left:
        mp_drw.draw_landmarks(overlay, res.left_hand_landmarks,
                              mp_hol.HAND_CONNECTIONS,
                              connection_drawing_spec=mp_drw.DrawingSpec(
                                  color=(255,0,255), thickness=3))
    if has_right:
        mp_drw.draw_landmarks(overlay, res.right_hand_landmarks,
                              mp_hol.HAND_CONNECTIONS,
                              connection_drawing_spec=mp_drw.DrawingSpec(
                                  color=(0,200,255), thickness=3))
    cv2.imwrite(os.path.join(OUT, name), overlay)

print("\n=== OZET ===")
n = max(1, len(files))
for k, v in stats.items():
    print(f"  {k:11s}: {v}/{n}  ({v*100/n:.0f}%)")
print(f"\nOverlay'ler: {OUT}")
