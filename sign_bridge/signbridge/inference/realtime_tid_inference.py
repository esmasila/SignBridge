"""
Gerçek Zamanlı TİD → Türkçe Çeviri
Webcam ile canlı işaret dili tanıma ve Türkçe metin/ses çıktısı
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
import sys
import time
from collections import deque
import pyttsx3
from typing import Optional

# Proje modüllerini import et
sys.path.append(str(Path(__file__).parent.parent.parent))

from signbridge.config import (
    INFERENCE_CONFIG,
    MEDIAPIPE_CONFIG,
    SEQUENCE_LENGTH,
    CLASS_TO_TURKISH,
    CHECKPOINT_DIR
)
from signbridge.data.preprocess import LandmarkExtractor
from signbridge.data.datasets import RealtimeLandmarkBuffer
from signbridge.models.islr_sequence_model import load_sequence_model
from signbridge.models.cnn_digits import load_digits_model
from signbridge.utils.logging_utils import setup_logger
from signbridge.utils.viz import put_turkish_text, display_prediction


class RealtimeTIDInference:
    """
    Gerçek zamanlı Türk İşaret Dili çevirici
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "cnn",  # "cnn" veya "sequence"
        use_tts: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            model_path: Eğitilmiş model dosyası
            model_type: "cnn" (rakamlar için) veya "sequence" (sekans için)
            use_tts: Text-to-Speech kullan
            confidence_threshold: Minimum güven skoru
        """
        self.logger = setup_logger("realtime_inference")
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Device: {self.device}")
        
        # Model yükle
        self.logger.info(f"Model yükleniyor: {model_path}")
        if model_type == "cnn":
            self.model = load_digits_model(model_path, self.device)
        else:
            self.model = load_sequence_model(model_path, model_type, self.device)
        
        self.model.eval()
        
        # MediaPipe
        self.landmark_extractor = LandmarkExtractor(MEDIAPIPE_CONFIG)
        
        # Landmark buffer (sekans modelleri için)
        if model_type == "sequence":
            self.landmark_buffer = RealtimeLandmarkBuffer(SEQUENCE_LENGTH)
        
        # TTS
        self.use_tts = use_tts
        if use_tts:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                self.logger.info("TTS motoru başlatıldı")
            except Exception as e:
                self.logger.warning(f"TTS başlatılamadı: {e}")
                self.use_tts = False
        
        # Tahmin smoothing
        self.prediction_history = deque(maxlen=INFERENCE_CONFIG["smoothing_window"])
        self.last_spoken_text = ""
        
        # FPS hesaplama
        self.fps_history = deque(maxlen=30)
        
        self.logger.info("Gerçek zamanlı çevirici hazır")
    
    def predict_from_image(self, image: np.ndarray) -> tuple:
        """
        Tek bir görüntüden tahmin yap (CNN için)
        
        Returns:
            prediction_text, confidence
        """
        if self.model_type != "cnn":
            raise ValueError("Bu fonksiyon sadece CNN modelleri için kullanılabilir")
        
        try:
            # Görüntüyü hazırla
            from torchvision import transforms
            from signbridge.config import DIGITS_MODEL_CONFIG
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(DIGITS_MODEL_CONFIG["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # RGB'ye çevir
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Tahmin
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = probabilities.max(1)
                
                # Top 3 tahminleri logla (INFO level - görünsün diye)
                top3_probs, top3_classes = probabilities[0].topk(3)
                debug_info = " | ".join([
                    f"{CLASS_TO_TURKISH.get(c.item(), str(c.item()))}: {p.item():.1%}"
                    for p, c in zip(top3_probs, top3_classes)
                ])
                self.logger.info(f"🎯 Top 3: {debug_info}")
            
            # Sınıf ID'yi Türkçe'ye çevir
            class_id = predicted_class.item()
            confidence_score = confidence.item()
            
            prediction_text = CLASS_TO_TURKISH.get(class_id, f"Bilinmeyen ({class_id})")
            
            return prediction_text, confidence_score
        except Exception as e:
            self.logger.error(f"Tahmin hatası: {e}")
            return "Hata", 0.0
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> tuple:
        """
        Landmark sekansından tahmin yap
        
        Returns:
            prediction_text, confidence
        """
        if self.model_type == "cnn":
            raise ValueError("Bu fonksiyon sekans modelleri için kullanılabilir")
        
        # Landmark'ı buffer'a ekle
        self.landmark_buffer.add(landmarks)
        
        # Buffer henüz dolu değilse bekle
        if not self.landmark_buffer.is_ready():
            return "Bekleniyor...", 0.0
        
        # Sekansı al
        sequence = self.landmark_buffer.get_sequence()
        
        # Tensor'e çevir
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        
        # Tahmin
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(1)
        
        # Sınıf ID'yi Türkçe'ye çevir
        class_id = predicted_class.item()
        confidence_score = confidence.item()
        
        prediction_text = CLASS_TO_TURKISH.get(class_id, f"Bilinmeyen ({class_id})")
        
        return prediction_text, confidence_score
    
    def smooth_predictions(self, prediction: str, confidence: float) -> tuple:
        """
        Tahminleri yumuşat (smoothing)
        
        Returns:
            smoothed_prediction, smoothed_confidence
        """
        if confidence < self.confidence_threshold:
            return "---", 0.0
        
        self.prediction_history.append((prediction, confidence))
        
        # En sık görülen tahmini al
        if len(self.prediction_history) >= 3:
            recent_preds = [p for p, c in self.prediction_history]
            most_common = max(set(recent_preds), key=recent_preds.count)
            avg_confidence = np.mean([c for p, c in self.prediction_history if p == most_common])
            return most_common, avg_confidence
        
        return prediction, confidence
    
    def speak_text(self, text: str):
        """
        Metni sesli oku (TTS)
        """
        if not self.use_tts or text == self.last_spoken_text or text == "---" or text == "Hata":
            return
        
        try:
            # Yeni thread'de çalıştır (engellemesini önlemek için)
            import threading
            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            
            thread = threading.Thread(target=speak, daemon=True)
            thread.start()
            self.last_spoken_text = text
        except Exception as e:
            self.logger.warning(f"TTS hatası: {e}")
    
    def run(self):
        """
        Ana çalıştırma döngüsü - webcam'den canlı çeviri
        """
        self.logger.info("=" * 60)
        self.logger.info("GERÇEK ZAMANLI TÜRK İŞARET DİLİ ÇEVİRİCİ")
        self.logger.info("=" * 60)
        self.logger.info("Çıkmak için 'q' tuşuna basın")
        self.logger.info("TTS açık/kapat için 't' tuşuna basın")
        
        # Kamera aç
        cap = cv2.VideoCapture(INFERENCE_CONFIG["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, INFERENCE_CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INFERENCE_CONFIG["frame_height"])
        
        if not cap.isOpened():
            self.logger.error("Kamera açılamadı!")
            return
        
        self.logger.info(f"Kamera başlatıldı: {INFERENCE_CONFIG['camera_id']}")
        
        # Kameranın ısınması için bekle
        time.sleep(1.0)
        self.logger.info("Kamera hazır")
        
        # MediaPipe drawing utils
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        while True:
            try:
                start_time = time.time()
                
                # Kare oku
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Kare okunamadı")
                    break
                
                # Aynalama (daha doğal görünüm için)
                frame = cv2.flip(frame, 1)
                
                # Tahmin yap
                if self.model_type == "cnn":
                    # CNN için doğrudan görüntüden tahmin
                    prediction, confidence = self.predict_from_image(frame)
                else:
                    # Sekans modelleri için landmark çıkar
                    landmarks = self.landmark_extractor.extract_from_image(frame)
                    if landmarks is not None:
                        prediction, confidence = self.predict_from_landmarks(landmarks)
                    else:
                        prediction, confidence = "Landmark bulunamadı", 0.0
                
                # Smoothing
                smoothed_pred, smoothed_conf = self.smooth_predictions(prediction, confidence)
                
                # TTS
                if smoothed_conf > self.confidence_threshold:
                    self.speak_text(smoothed_pred)
                
                # FPS hesapla
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)
                
                # Görselleştirme
                display_frame = display_prediction(
                    frame, smoothed_pred, smoothed_conf, avg_fps
                )
                
                # Ekstra bilgiler
                info_text = f"Model: {self.model_type.upper()}"
                display_frame = put_turkish_text(
                    display_frame, info_text, (10, display_frame.shape[0] - 60),
                    font_scale=0.6, color=(200, 200, 200), background=False
                )
                
                tts_status = "TTS: ACIK" if self.use_tts else "TTS: KAPALI"
                display_frame = put_turkish_text(
                    display_frame, tts_status, (10, display_frame.shape[0] - 40),
                    font_scale=0.6, color=(0, 255, 255), background=False
                )
                
                # Göster
                cv2.imshow('SignBridge - Turk Isaret Dili Cevirici', display_frame)
                
                # Tuş kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Kullanıcı çıkış yaptı")
                    break
                elif key == ord('t'):
                    self.use_tts = not self.use_tts
                    self.logger.info(f"TTS: {'AÇIK' if self.use_tts else 'KAPALI'}")
            except Exception as e:
                self.logger.error(f"Ana döngüde hata: {e}", exc_info=True)
                continue
        
        # Temizlik
        cap.release()
        cv2.destroyAllWindows()
        self.landmark_extractor.close()
        
        self.logger.info("Program sonlandırıldı")


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerçek Zamanlı TİD Çevirici")
    parser.add_argument(
        "--model",
        type=str,
        default=str(CHECKPOINT_DIR / "digits_cnn_best.pt"),
        help="Model dosyası yolu"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="cnn",
        choices=["cnn", "lstm", "gru", "transformer"],
        help="Model tipi"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="TTS'yi devre dışı bırak"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum güven skoru"
    )
    
    args = parser.parse_args()
    
    # Model dosyası kontrolü
    if not Path(args.model).exists():
        print(f"HATA: Model dosyası bulunamadı: {args.model}")
        print("\nÖnce modeli eğitmeniz gerekiyor:")
        print("  python -m signbridge.training.train_digits")
        return
    
    # Çeviriciyi başlat
    inferencer = RealtimeTIDInference(
        model_path=args.model,
        model_type=args.type,
        use_tts=not args.no_tts,
        confidence_threshold=args.confidence
    )
    
    inferencer.run()


if __name__ == "__main__":
    main()
