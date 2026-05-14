"""
Mobil portre 480x720 JPEG vs landscape'e cevirilmis varyantlar:
landmark vector'larin sayisal farkini olc.

Eger normalize SONRASI vector'lar farkliysa, frame orani onemli =>
mobile'i veya server'i landscape moduna almak gerek.
"""
import cv2, os, glob
import numpy as np
import mediapipe as mp

BASE = os.path.dirname(__file__)
INC  = os.path.join(BASE, "debug_incoming")

# Server pipeline ile AYNI normalize fonksiyonu
FACE_END  = 468 * 3
POSE_END  = FACE_END + 33*3
LEFT_END  = POSE_END + 21*3
RIGHT_END = LEFT_END + 21*3

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
        center = (ls + rs) / 2.0
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

mp_hol = mp.solutions.holistic

def run(img, label):
    hol = mp_hol.Holistic(static_image_mode=True, model_complexity=1,
                           min_detection_confidence=0.3,
                           refine_face_landmarks=False)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hol.process(rgb)
    raw  = extract(res)
    norm = normalize(raw)
    h, w = img.shape[:2]
    has_L = res.left_hand_landmarks is not None
    has_R = res.right_hand_landmarks is not None
    # Pose'un omuz mesafesi (pixel)
    pose = raw[FACE_END:POSE_END].reshape(33, 3)
    sh_dist = float(np.linalg.norm(pose[11][:2] - pose[12][:2]))
    print(f"  {label:30s} {w}x{h}  L={int(has_L)} R={int(has_R)} "
          f"shoulder_norm_dist={sh_dist:.3f}")
    return norm

files = sorted(glob.glob(os.path.join(INC, "*.jpg")))
# Orta frame en aktif
target = files[len(files)//2]
print(f"Test frame: {os.path.basename(target)}\n")

img_p = cv2.imread(target)
h, w = img_p.shape[:2]
print(f"Orijinal: {w}x{h} (portre)\n")

# Varyant 1: orijinal portre 480x720
v_portrait = run(img_p, "(1) orijinal portre")

# Varyant 2: 90 saat yonu donus (landscape 720x480)
img_l_cw = cv2.rotate(img_p, cv2.ROTATE_90_CLOCKWISE)
v_land_cw = run(img_l_cw, "(2) 90 saat yonu (landscape)")

# Varyant 3: 90 ters saat (landscape 720x480, ters)
img_l_ccw = cv2.rotate(img_p, cv2.ROTATE_90_COUNTERCLOCKWISE)
v_land_ccw = run(img_l_ccw, "(3) 90 ters saat (landscape)")

# Varyant 4: Center-crop portre -> kare 480x480
crop_y = (h - w) // 2
img_sq = img_p[crop_y:crop_y+w, :]
v_sq = run(img_sq, "(4) center-crop kare 480x480")

# Varyant 5: portre'yi 640x480 landscape'e WARP (oran bozulur)
img_warp = cv2.resize(img_p, (640, 480))
v_warp = run(img_warp, "(5) warp 640x480")

# Varyant 6: portre'yi 480x480'e RESIZE (kare)
img_sq2 = cv2.resize(img_p, (480, 480))
v_sq2 = run(img_sq2, "(6) resize 480x480 (warp)")

# Pairwise farklar (face/pose/hands ortalama mutlak fark)
def diff(a, b, name_a, name_b):
    d = np.abs(a - b).mean()
    print(f"    {name_a} vs {name_b}: mean|delta|={d:.5f}")

print("\n=== PAIRWISE NORMALIZE-SONRASI VECTOR FARKLARI ===")
print("(0'a yakin = ayni dagilim, buyuk = farkli landmark cikiyor)")
diff(v_portrait, v_land_cw,  "portre", "land_cw")
diff(v_portrait, v_land_ccw, "portre", "land_ccw")
diff(v_portrait, v_sq,       "portre", "sq_crop")
diff(v_portrait, v_warp,     "portre", "warp_640x480")
diff(v_portrait, v_sq2,      "portre", "sq_resize")

# Spesifik el bolgesi farki
def hand_diff(a, b):
    L_a = a[POSE_END:LEFT_END]; L_b = b[POSE_END:LEFT_END]
    R_a = a[LEFT_END:RIGHT_END]; R_b = b[LEFT_END:RIGHT_END]
    return float(np.abs(L_a - L_b).mean()), float(np.abs(R_a - R_b).mean())

print("\n=== EL BOLGESI ICIN FARK (en kritik) ===")
for name, v in [("land_cw", v_land_cw), ("land_ccw", v_land_ccw),
                ("sq_crop", v_sq), ("warp", v_warp), ("sq_resize", v_sq2)]:
    Ld, Rd = hand_diff(v_portrait, v)
    print(f"  portre vs {name}: L_diff={Ld:.4f}  R_diff={Rd:.4f}")
