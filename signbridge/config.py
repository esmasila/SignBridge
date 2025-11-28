"""
SignBridge Konfigürasyon Dosyası
Proje genelinde kullanılacak ayarlar
"""

import os
from pathlib import Path

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Veri dizinleri
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model checkpoint dizini
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Log dizini
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

# MediaPipe Holistic ayarları
MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.3,  # Düşürüldü (0.5→0.3)
    "min_tracking_confidence": 0.3,  # Düşürüldü (0.5→0.3)
    "model_complexity": 0,  # 0: Lite (EN HIZLI, 3x hızlanma), 1: Full, 2: Heavy
    "enable_segmentation": False,
    "smooth_landmarks": False  # Devre dışı (hız için)
}

# Landmark sayıları
NUM_FACE_LANDMARKS = 468
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21  # Her el için
TOTAL_LANDMARKS = NUM_FACE_LANDMARKS + NUM_POSE_LANDMARKS + (2 * NUM_HAND_LANDMARKS)

# Feature boyutu (her landmark x, y, z koordinatları)
LANDMARK_DIM = 3  # x, y, z
TOTAL_FEATURE_DIM = TOTAL_LANDMARKS * LANDMARK_DIM

# Sekans ayarları
SEQUENCE_LENGTH = 30  # Kaç frame birleştirilerek tahmin yapılacak
FPS_TARGET = 25  # Hedef FPS

# Model ayarları
DIGITS_MODEL_CONFIG = {
    "num_classes": 10,
    "input_size": (64, 64),
    "channels": 3,
    "dropout": 0.5
}

SEQUENCE_MODEL_CONFIG = {
    "input_dim": TOTAL_FEATURE_DIM,
    "hidden_dim": 256,
    "num_layers": 2,
    "num_classes": 100,  # Başlangıç, büyük veri setleriyle artacak
    "dropout": 0.3,
    "bidirectional": True
}

# Eğitim ayarları
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "patience": 10,  # Early stopping
    "validation_split": 0.2,
    "test_split": 0.1,
    "num_workers": 4,
    "device": "cuda",  # "cuda" veya "cpu"
}

# Gerçek zamanlı çıkarım ayarları
INFERENCE_CONFIG = {
    "confidence_threshold": 0.6,
    "smoothing_window": 5,  # Son N tahminin ortalaması
    "max_latency_ms": 500,  # Maksimum gecikme (ms)
    "enable_tts": True,  # Text-to-Speech aktif mi?
    "tts_language": "tr",  # Türkçe
    "display_landmarks": True,  # Landmark'ları göster
    "camera_id": 0,  # Varsayılan kamera
    "frame_width": 640,
    "frame_height": 480
}

# Sınıf ID → Türkçe metin haritalama
# TODO: Büyük veri setleri için genişletilecek
CLASS_TO_TURKISH = {
    # Rakamlar (0-9) için
    0: "sıfır",
    1: "bir",
    2: "iki",
    3: "üç",
    4: "dört",
    5: "beş",
    6: "altı",
    7: "yedi",
    8: "sekiz",
    9: "dokuz",
    
    # İleride AUTSL ve BosphorusSign22k için eklenecek
    # Örnek TİD işaretleri:
    # 10: "merhaba",
    # 11: "teşekkür ederim",
    # 12: "günaydın",
    # 13: "iyi akşamlar",
    # 14: "evet",
    # 15: "hayır",
    # ... vb.
}

# Veri seti yolları
DATASET_PATHS = {
    "digits": RAW_DATA_DIR / "Sign-Language-Digits-Dataset-master" / "Dataset",
    "autsl": RAW_DATA_DIR / "AUTSL",
    "bosphorus": RAW_DATA_DIR / "BosphorusSign22k"
}

# Loglama ayarları
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "signbridge.log"),
            "mode": "a"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}

# Gizlilik ayarları
PRIVACY_CONFIG = {
    "save_raw_video": False,  # Ham video kaydedilmez
    "save_frames": False,  # Kareler diske yazılmaz
    "send_to_server": False,  # Hiçbir veri sunucuya gönderilmez
    "in_memory_only": True  # Sadece bellekte işleme
}

# API ayarları
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": True,  # Geliştirme için
    "title": "SignBridge API",
    "description": "Türk İşaret Dili → Türkçe Metin Çeviri API'si",
    "version": "0.1.0"
}
