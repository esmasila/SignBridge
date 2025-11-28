"""
Video'dan Sliding Window ile Veri Toplama
30 saniye video → 100+ örnek
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime


# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Yol
BASE_DATA_PATH = Path(__file__).parent.parent / "data"
BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Parametreler
SEQUENCE_LENGTH = 60  # Her örnek 60 frame
STRIDE = 5            # Her 5 frame'de bir yeni örnek başlat (overlap)


def extract_landmarks(results):
    """Tüm landmark'ları düzleştirilmiş vektör olarak döndür"""
    landmarks = []
    
    # Yüz (468×3)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (468 * 3))
    
    # Pose (33×3)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (33 * 3))
    
    # Sol el (21×3)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (21 * 3))
    
    # Sağ el (21×3)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (21 * 3))
    
    return np.array(landmarks, dtype=np.float32)


def process_video(video_path, label, output_dir):
    """
    Video'dan sliding window ile örnekler çıkar
    
    Args:
        video_path: video dosyası path'i
        label: kelime etiketi
        output_dir: çıktı klasörü
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video: {fps:.1f} FPS, {total_frames} frame ({total_frames/fps:.1f}s)")
    
    # Tüm frame'leri işle
    all_landmarks = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            landmarks = extract_landmarks(results)
            all_landmarks.append(landmarks)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  İşleniyor... {frame_idx}/{total_frames}")
    
    cap.release()
    
    print(f"✅ {len(all_landmarks)} frame işlendi")
    
    # Sliding window ile örnekler oluştur
    sequences = []
    for i in range(0, len(all_landmarks) - SEQUENCE_LENGTH + 1, STRIDE):
        seq = all_landmarks[i:i+SEQUENCE_LENGTH]
        if len(seq) == SEQUENCE_LENGTH:
            sequences.append(np.array(seq))
    
    print(f"✅ {len(sequences)} örnek oluşturuldu (stride={STRIDE})")
    
    # Kaydet
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for idx, seq in enumerate(sequences):
        filename = f"{timestamp}_{idx:04d}.npy"
        filepath = label_dir / filename
        np.save(filepath, seq)
    
    print(f"✅ {len(sequences)} örnek kaydedildi: {label_dir}")
    print(f"   Örnek shape: {sequences[0].shape}\n")


def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print("VİDEO'DAN VERİ TOPLAMA")
    print("=" * 70)
    print("\nKULLANIM:")
    print("  1. Video çek (30-60 saniye, hareketi tekrarla)")
    print("  2. Video'yu bu klasöre koy: tid_sequence/videos/")
    print("  3. Bu scripti çalıştır\n")
    print("=" * 70)
    
    # Video klasörü
    video_dir = Path(__file__).parent.parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Video dosyalarını bul
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
    
    if not video_files:
        print(f"\n❌ Video bulunamadı: {video_dir}")
        print("\nÖnerilen format:")
        print("  tid_sequence/videos/MERHABA.mp4")
        print("  tid_sequence/videos/EVET.mp4")
        print("  ...")
        return
    
    print(f"\n📁 {len(video_files)} video bulundu:\n")
    for vf in video_files:
        print(f"  - {vf.name}")
    
    print("\n" + "=" * 70)
    
    # Her video'yu işle
    for video_file in video_files:
        # Dosya adından label al (örn: MERHABA.mp4 -> MERHABA)
        label = video_file.stem.upper()
        
        print(f"\n🎬 İşleniyor: {video_file.name} → {label}")
        print("-" * 70)
        
        process_video(str(video_file), label, BASE_DATA_PATH)
    
    print("=" * 70)
    print("🎉 TÜM VİDEOLAR İŞLENDİ!")
    print("=" * 70)


if __name__ == "__main__":
    main()
