"""
Video → Avatar JSON dönüştürücü
MediaPipe Holistic ile videodan landmark çıkarıp step3_player.html formatında JSON üretir.
step2_recorder.html ile aynı normalizasyon mantığını kullanır.
"""
import cv2
import mediapipe as mp
import json
import sys
import math
import os

def normalize_hand(landmarks):
    """21 hand landmark → wrist-centered, scaled by wrist-to-middle-MCP distance
    MCP kullanıyoruz çünkü: yumrukta parmak ucu bilege çok yakın, scale çok küçük oluyor
    ve tüm koordinatlar abartılıyor. MCP mesafesi fist/open el fark etmez stabil."""
    if not landmarks:
        return [[0,0,0]] * 21
    lms = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    w = lms[0]   # wrist
    m = lms[9]   # middle finger MCP (stabil referans — tip yerine)
    scale = math.sqrt((m[0]-w[0])**2 + (m[1]-w[1])**2 + (m[2]-w[2])**2) or 0.1
    return [
        [(lm[0]-w[0])/scale, (lm[1]-w[1])/scale, (lm[2]-w[2])/scale]
        for lm in lms
    ]

def normalize_pose(pose_landmarks):
    """Pose landmarks → shoulder-centered, scaled by shoulder width, Y flipped"""
    z = [0,0,0]
    if not pose_landmarks:
        return {
            'ns':z,'lear':z,'rear':z,'lmc':z,'rmc':z,
            'ls':z,'le':z,'lw':z,'lpk':z,'lix':z,'ltb':z,
            'rs':z,'re':z,'rw':z,'rpk':z,'rix':z,'rtb':z,
            'lhp':z,'rhp':z,
        }
    lms = pose_landmarks.landmark
    ls, rs = lms[11], lms[12]
    cx = (ls.x + rs.x) / 2
    cy = (ls.y + rs.y) / 2
    scale = math.sqrt((rs.x-ls.x)**2 + (rs.y-ls.y)**2) or 0.1

    def get(idx):
        lm = lms[idx]
        return [
            (lm.x - cx) / scale,
            -(lm.y - cy) / scale,  # Y flip (MediaPipe: down+, Three.js: up+)
            lm.z / scale,
        ]

    return {
        'ns': get(0), 'lear': get(7), 'rear': get(8),
        'lmc': get(9), 'rmc': get(10),
        'ls': get(11), 'le': get(13), 'lw': get(15),
        'lpk': get(17), 'lix': get(19), 'ltb': get(21),
        'rs': get(12), 're': get(14), 'rw': get(16),
        'rpk': get(18), 'rix': get(20), 'rtb': get(22),
        'lhp': get(23), 'rhp': get(24),
    }

def interpolate_missing_hands(frames):
    """Fill missing hand data via linear interpolation (same as step2_recorder)"""
    def has_hand(h):
        return h and any(p[0]!=0 or p[1]!=0 or p[2]!=0 for p in h)

    def lerp_pt(a, b, t):
        return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]

    def lerp21(a, b, t):
        return [lerp_pt(a[i], b[i], t) for i in range(21)]

    for side in ['left_hand', 'right_hand']:
        for i in range(len(frames)):
            if has_hand(frames[i][side]):
                continue
            # Find prev
            prev = -1
            for j in range(i-1, -1, -1):
                if has_hand(frames[j][side]):
                    prev = j; break
            # Find next
            nxt = -1
            for j in range(i+1, len(frames)):
                if has_hand(frames[j][side]):
                    nxt = j; break

            if prev >= 0 and nxt >= 0:
                t = (i - prev) / (nxt - prev)
                frames[i][side] = lerp21(frames[prev][side], frames[nxt][side], t)
            elif prev >= 0:
                frames[i][side] = [list(p) for p in frames[prev][side]]
            elif nxt >= 0:
                frames[i][side] = [list(p) for p in frames[nxt][side]]

    return frames


def process_video(video_path, word, output_dir, target_fps=30):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video açılamadı: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample every N frames to get ~target_fps
    sample_interval = max(1, round(video_fps / target_fps))

    print(f"Video: {video_path}")
    print(f"  FPS: {video_fps}, Frames: {total_frames}, Duration: {total_frames/video_fps:.2f}s")
    print(f"  Sample interval: every {sample_interval} frame(s)")

    frames = []
    frame_idx = 0
    left_detected = 0
    right_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            pose = normalize_pose(results.pose_landmarks)
            left_hand = normalize_hand(results.left_hand_landmarks)
            right_hand = normalize_hand(results.right_hand_landmarks)

            if results.left_hand_landmarks:
                left_detected += 1
            if results.right_hand_landmarks:
                right_detected += 1

            frames.append({
                'pose': pose,
                'left_hand': left_hand,
                'right_hand': right_hand,
            })

        frame_idx += 1

    cap.release()
    holistic.close()

    print(f"  Extracted: {len(frames)} frames")
    print(f"  Left hand detected: {left_detected}/{len(frames)}")
    print(f"  Right hand detected: {right_detected}/{len(frames)}")

    # Interpolate missing hands
    frames = interpolate_missing_hands(frames)

    # Save
    data = {
        'word': word,
        'fps': target_fps,
        'frames': frames,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{word}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"  Saved: {out_path} ({len(frames)} frames)")
    return out_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Kullanım: python video_to_json.py <video_path> [WORD] [output_dir]")
        sys.exit(1)

    video_path = sys.argv[1]
    word = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(video_path))[0].upper()
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'web/avatar/npy'

    process_video(video_path, word, output_dir)
