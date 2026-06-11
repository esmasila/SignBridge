"""
AUTSL Veri Setini YOLO Formatına Dönüştürme
============================================
226 sınıflık Türk İşaret Dili veri seti
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import mediapipe as mp
import yaml

# Ayarlar
AUTSL_DIR = "autsl_dataset"
OUTPUT_DIR = "autsl_yolo"
FRAMES_PER_VIDEO = 5  # Her videodan kaç frame alınacak
IMG_SIZE = 640

# MediaPipe el tespiti
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)


def get_hand_bbox(image):
    """MediaPipe ile el bounding box'ı bul"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    h, w = image.shape[:2]
    all_x, all_y = [], []
    
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_x.append(lm.x * w)
            all_y.append(lm.y * h)
    
    if not all_x:
        return None
    
    # Bounding box hesapla (padding ekle)
    padding = 30
    x_min = max(0, int(min(all_x)) - padding)
    y_min = max(0, int(min(all_y)) - padding)
    x_max = min(w, int(max(all_x)) + padding)
    y_max = min(h, int(max(all_y)) + padding)
    
    # YOLO formatına dönüştür (normalized)
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    
    return x_center, y_center, width, height


def extract_frames_from_video(video_path, num_frames=5):
    """Videodan eşit aralıklı frame'ler çıkar"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Eşit aralıklı frame indeksleri
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def process_dataset(split="train"):
    """Veri setini işle ve YOLO formatına dönüştür"""
    
    # Labels dosyasını oku
    labels_file = os.path.join(AUTSL_DIR, f"{split}_labels.csv")
    df = pd.read_csv(labels_file, header=None, names=['video', 'label'])
    
    # Çıktı klasörlerini oluştur
    images_dir = os.path.join(OUTPUT_DIR, split, "images")
    labels_dir = os.path.join(OUTPUT_DIR, split, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    video_dir = os.path.join(AUTSL_DIR, split)
    
    processed = 0
    skipped = 0
    
    print(f"\n{split.upper()} seti isleniyor ({len(df)} video)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        video_name = row['video']
        label = row['label']
        
        # Video dosyası yolları (color ve depth var)
        color_video = os.path.join(video_dir, f"{video_name}_color.mp4")
        
        if not os.path.exists(color_video):
            skipped += 1
            continue
        
        # Frame'leri çıkar
        frames = extract_frames_from_video(color_video, FRAMES_PER_VIDEO)
        
        for i, frame in enumerate(frames):
            # El bounding box'ı bul
            bbox = get_hand_bbox(frame)
            
            if bbox is None:
                # El bulunamazsa tam frame kullan
                bbox = (0.5, 0.5, 0.9, 0.9)
            
            # Dosya adları
            img_name = f"{video_name}_f{i}.jpg"
            label_name = f"{video_name}_f{i}.txt"
            
            # Resmi kaydet
            img_path = os.path.join(images_dir, img_name)
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(img_path, frame_resized)
            
            # YOLO label dosyasını kaydet
            label_path = os.path.join(labels_dir, label_name)
            x_center, y_center, width, height = bbox
            with open(label_path, 'w') as f:
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed += 1
    
    print(f"{split}: {processed} frame islendi, {skipped} video atlandi")
    return processed


def create_data_yaml():
    """data.yaml dosyasını oluştur"""
    
    # 226 sınıf için isimler (0-225)
    class_names = [f"sign_{i}" for i in range(226)]
    
    data = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 226,
        'names': class_names
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"\ndata.yaml olusturuldu: {yaml_path}")


def main():
    print("=" * 60)
    print("AUTSL -> YOLO Format Donusumu")
    print("=" * 60)
    print(f"Kaynak: {AUTSL_DIR}")
    print(f"Hedef: {OUTPUT_DIR}")
    print(f"Frame/video: {FRAMES_PER_VIDEO}")
    print(f"Sinif sayisi: 226")
    
    # Çıktı klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Her split'i işle
    total = 0
    for split in ['train', 'val', 'test']:
        count = process_dataset(split)
        total += count
    
    # data.yaml oluştur
    create_data_yaml()
    
    print("\n" + "=" * 60)
    print(f"TAMAMLANDI! Toplam {total} frame olusturuldu.")
    print(f"YOLO veri seti: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
