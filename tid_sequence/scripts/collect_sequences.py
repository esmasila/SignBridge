"""
TİD Sekans Veri Toplama Aracı - Basitleştirilmiş
S tuşu ile otomatik 50 örnek toplar

KULLANIM:
    python collect_sequences.py
    Etiket: MERHABA
    S tuşuna bas → 50 örnek toplamaya başlar (her 3 saniyede bir)
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Yol
BASE_DATA_PATH = Path(__file__).parent.parent / "data"
BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Parametreler
MAX_FRAMES = 60
TARGET_SAMPLES = 50  # Her kelime için 50 örnek
COUNTDOWN_FRAMES = 90  # 3 saniye geri sayım (30 FPS × 3) - sadece ilk kayıt için


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


def draw_landmarks(image, results):
    """Landmark'ları çiz"""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


def save_sequence(sequence, label, counter):
    """Sekansı kaydet"""
    label_path = BASE_DATA_PATH / label
    label_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{counter:03d}.npy"
    filepath = label_path / filename
    
    np.save(filepath, sequence)
    print(f"✅ {counter+1}/{TARGET_SAMPLES} - {filepath.name} - Shape: {sequence.shape}")


def main():
    print("="*70)
    print("TİD SEKANS VERİ TOPLAMA")
    print("="*70)
    print("\nKONTROLLER:")
    print("  S: Otomatik 50 örnek topla")
    print("  E: Yeni etiket")
    print("  Q/ESC: Çıkış\n")
    print("="*70)
    
    label = input("\n📝 Etiket: ").strip().upper()
    if not label:
        print("❌ Etiket boş!")
        return
    
    print(f"✅ Etiket: {label}")
    print(f"📁 Konum: {BASE_DATA_PATH / label}\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        recording = False
        auto_mode = False
        sequence = []
        counter = 0
        countdown = 0
        
        print("🎥 Kamera açıldı! S tuşuna basın...\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            landmarks = extract_landmarks(results)
            
            # Geri sayım
            if auto_mode and not recording and countdown > 0:
                seconds_left = countdown // 30  # Frame'i saniyeye çevir
                cv2.putText(frame, f"BASLIYOR: {seconds_left}", (frame.shape[1]//2-100, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                countdown -= 1
                if countdown == 0:
                    recording = True
                    sequence = []
            
            # Kayıt
            if recording:
                sequence.append(landmarks)
                cv2.putText(frame, f"KAYIT: {len(sequence)}/{MAX_FRAMES}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                if len(sequence) >= MAX_FRAMES:
                    save_sequence(np.array(sequence), label, counter)
                    counter += 1
                    recording = False
                    sequence = []
                    
                    if auto_mode and counter < TARGET_SAMPLES:
                        # İlk kayıttan sonra direkt devam et (countdown = 0)
                        recording = True
                        sequence = []
                    elif auto_mode and counter >= TARGET_SAMPLES:
                        print(f"\n🎉 TAMAMLANDI! {TARGET_SAMPLES} örnek toplandı!\n")
                        auto_mode = False
            
            # Görselleştirme
            draw_landmarks(frame, results)
            
            cv2.putText(frame, f"Etiket: {label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Ornek: {counter}/{TARGET_SAMPLES}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if not recording and countdown == 0:
                mode_text = "OTOMATIK" if auto_mode else "MANUEL"
                cv2.putText(frame, f"{mode_text} | S:Baslat E:Etiket Q:Cikis", 
                           (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('TID Veri Toplama', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if not auto_mode:
                    auto_mode = True
                    countdown = COUNTDOWN_FRAMES
                    print(f"✅ Otomatik mod: {TARGET_SAMPLES} örnek toplanacak!\n")
            
            elif key == ord('e') or key == ord('E'):
                cv2.destroyAllWindows()
                label = input("\n📝 Yeni etiket: ").strip().upper()
                if label:
                    counter = 0
                    auto_mode = False
                    print(f"✅ Yeni etiket: {label}\n")
            
            elif key == ord('q') or key == ord('Q') or key == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Toplam {counter} örnek kaydedildi!")
    print("="*70)


if __name__ == "__main__":
    main()
