"""
MediaPipe ile landmark çıkarma ve veri ön işleme
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from tqdm import tqdm

from signbridge.config import (
    MEDIAPIPE_CONFIG,
    NUM_FACE_LANDMARKS,
    NUM_POSE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    TOTAL_LANDMARKS,
    LANDMARK_DIM
)

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """
    MediaPipe Holistic kullanarak görüntülerden landmark çıkarımı yapar
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: MediaPipe konfigürasyonu (None ise varsayılan kullanılır)
        """
        self.config = config or MEDIAPIPE_CONFIG
        
        # MediaPipe Holistic modelini başlat
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.config["min_detection_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"],
            model_complexity=self.config["model_complexity"],
            enable_segmentation=self.config["enable_segmentation"],
            smooth_landmarks=self.config["smooth_landmarks"]
        )
        
        logger.info("MediaPipe Holistic modeli yüklendi")
    
    def extract_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Tek bir görüntüden landmark'ları çıkarır
        
        Args:
            image: BGR formatında OpenCV görüntüsü
            
        Returns:
            landmarks: Shape (TOTAL_LANDMARKS * 3,) numpy array veya None
        """
        # BGR'den RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile işle
        results = self.holistic.process(image_rgb)
        
        # Landmark'ları birleştir
        landmarks = self._combine_landmarks(results)
        
        return landmarks
    
    def _combine_landmarks(self, results) -> Optional[np.ndarray]:
        """
        Yüz, vücut ve el landmark'larını tek bir vektöre birleştirir
        
        Returns:
            Combined landmarks shape: (TOTAL_LANDMARKS * 3,) veya None
        """
        landmarks_list = []
        
        # Yüz landmark'ları (sadece ilk 33 nokta - veri toplama ile uyumlu)
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark[:33]])
            landmarks_list.append(face.flatten())
        else:
            # Yüz bulunamadıysa sıfırlarla doldur (33 * 3 = 99)
            landmarks_list.append(np.zeros(33 * LANDMARK_DIM))
        
        # Pose landmark'ları (33 nokta)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            landmarks_list.append(pose.flatten())
        else:
            landmarks_list.append(np.zeros(NUM_POSE_LANDMARKS * LANDMARK_DIM))
        
        # Sol el landmark'ları (21 nokta)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            landmarks_list.append(left_hand.flatten())
        else:
            landmarks_list.append(np.zeros(NUM_HAND_LANDMARKS * LANDMARK_DIM))
        
        # Sağ el landmark'ları (21 nokta)
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            landmarks_list.append(right_hand.flatten())
        else:
            landmarks_list.append(np.zeros(NUM_HAND_LANDMARKS * LANDMARK_DIM))
        
        # Tüm landmark'ları birleştir
        combined = np.concatenate(landmarks_list)
        
        return combined
    
    def extract_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Video dosyasından tüm karelerin landmark'larını çıkarır
        
        Args:
            video_path: Video dosyası yolu
            
        Returns:
            landmarks_sequence: Liste içinde her kare için landmark array'i
        """
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_from_image(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
        
        cap.release()
        logger.info(f"{video_path} dosyasından {len(landmarks_sequence)} kare işlendi")
        
        return landmarks_sequence
    
    def close(self):
        """MediaPipe kaynaklarını temizle"""
        self.holistic.close()
        logger.info("MediaPipe kapatıldı")


def normalize_landmarks(landmarks: np.ndarray, method: str = "min_max") -> np.ndarray:
    """
    Landmark'ları normalize eder
    
    Args:
        landmarks: Shape (seq_len, feature_dim) veya (feature_dim,)
        method: "min_max", "standard", veya "none"
        
    Returns:
        normalized: Normalize edilmiş landmarks
    """
    if method == "min_max":
        # [0, 1] aralığına normalize et
        min_val = np.min(landmarks, axis=0, keepdims=True)
        max_val = np.max(landmarks, axis=0, keepdims=True)
        normalized = (landmarks - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "standard":
        # Z-score normalizasyonu
        mean = np.mean(landmarks, axis=0, keepdims=True)
        std = np.std(landmarks, axis=0, keepdims=True)
        normalized = (landmarks - mean) / (std + 1e-8)
    
    else:
        normalized = landmarks
    
    return normalized


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    dataset_type: str = "digits",
    max_samples: Optional[int] = None
):
    """
    Veri setindeki tüm görüntüleri işleyerek landmark'ları çıkarır
    
    Args:
        input_dir: Girdi veri klasörü
        output_dir: Çıktı klasörü (.npz dosyaları için)
        dataset_type: "digits", "autsl", veya "bosphorus"
        max_samples: İşlenecek maksimum örnek sayısı (None = hepsi)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    extractor = LandmarkExtractor()
    
    if dataset_type == "digits":
        # Digits veri seti: Dataset/0, Dataset/1, ... Dataset/9
        logger.info(f"Rakam veri seti işleniyor: {input_dir}")
        
        all_landmarks = []
        all_labels = []
        
        for class_idx in range(10):
            class_dir = input_dir / str(class_idx)
            if not class_dir.exists():
                logger.warning(f"Klasör bulunamadı: {class_dir}")
                continue
            
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if max_samples:
                image_files = image_files[:max_samples // 10]
            
            logger.info(f"Sınıf {class_idx} ({len(image_files)} görüntü) işleniyor...")
            
            for img_path in tqdm(image_files, desc=f"Sınıf {class_idx}"):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                landmarks = extractor.extract_from_image(image)
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    all_labels.append(class_idx)
        
        # NumPy array'e çevir ve kaydet
        all_landmarks = np.array(all_landmarks)
        all_labels = np.array(all_labels)
        
        output_file = output_dir / "digits_landmarks.npz"
        np.savez_compressed(
            output_file,
            landmarks=all_landmarks,
            labels=all_labels
        )
        
        logger.info(f"İşleme tamamlandı: {len(all_landmarks)} örnek")
        logger.info(f"Kaydedildi: {output_file}")
    
    elif dataset_type in ["autsl", "bosphorus"]:
        # TODO: AUTSL ve BosphorusSign22k için özel işleme mantığı
        logger.warning(f"{dataset_type} veri seti işleme henüz implemente edilmedi")
        logger.info("Bu veri setleri video tabanlı ve sekans içerir")
        logger.info("Gelecek versiyonda eklenecek")
    
    extractor.close()


def load_preprocessed_data(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    İşlenmiş landmark verilerini yükler
    
    Args:
        npz_path: .npz dosya yolu
        
    Returns:
        landmarks: Shape (N, feature_dim)
        labels: Shape (N,)
    """
    data = np.load(npz_path)
    landmarks = data["landmarks"]
    labels = data["labels"]
    
    logger.info(f"Yüklendi: {npz_path}")
    logger.info(f"Shape: landmarks={landmarks.shape}, labels={labels.shape}")
    
    return landmarks, labels


if __name__ == "__main__":
    # Test kodu
    from signbridge.config import DATASET_PATHS, PROCESSED_DATA_DIR
    
    logging.basicConfig(level=logging.INFO)
    
    # Rakam veri setini işle
    digits_input = DATASET_PATHS["digits"]
    if digits_input.exists():
        preprocess_dataset(
            input_dir=digits_input,
            output_dir=PROCESSED_DATA_DIR,
            dataset_type="digits"
        )
    else:
        logger.error(f"Veri seti bulunamadı: {digits_input}")
        logger.info("Lütfen Sign-Language-Digits-Dataset-master klasörünü data/raw/ içine yerleştirin")
