"""
AUTSL Model Test - OpenCV Tabanlı
Eğitim verileriyle birebir aynı preprocessing
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import json
import math
from collections import deque
from pathlib import Path
import time

# ==================== MODEL TANIMI ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignTransformerPro(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.35):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Linear(input_size, d_model), 
            nn.LayerNorm(d_model), 
            nn.GELU(), 
            nn.Dropout(dropout)
        )
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model), 
            nn.GELU(), 
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//4), nn.Tanh(), nn.Linear(d_model//4, 1))
            for _ in range(4)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model*5), nn.Dropout(dropout),
            nn.Linear(d_model*5, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(d_model, num_classes)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.input_conv(x)
        x = x + self.conv_block(x.transpose(1,2)).transpose(1,2)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1)
        x = self.transformer(self.pos_encoder(x))
        seq_out = x[:, 1:]
        pooled = [F.softmax(h(seq_out), dim=1) * seq_out for h in self.pool_heads]
        pooled = [p.sum(dim=1) for p in pooled]
        combined = torch.cat(pooled + [seq_out.mean(dim=1)], dim=1)
        return self.classifier(combined)


# ==================== KONFIGÜRASYON ====================

MODEL_DIR = Path(__file__).parent / 'autsl_transformer' / 'model'
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225  # 75 nokta × 3 koordinat

# Threshold değerleri
CONFIDENCE_THRESHOLD = 0.2  # Minimum güven skoru (test için düşürüldü)
MIN_HAND_FRAMES = 10  # Minimum el tespit edilmesi gereken frame sayısı
HAND_VISIBILITY_THRESHOLD = 0.3  # MediaPipe görünürlük eşiği


# ==================== LANDMARK EXTRACTION ====================

def extract_keypoints(results):
    """
    MediaPipe sonuçlarından keypoint çıkar
    EĞİTİM VERİSİYLE BİREBİR AYNI FORMAT
    
    Sıralama:
    1. Pose (33 nokta × 3) = 99 değer
    2. Left Hand (21 nokta × 3) = 63 değer  
    3. Right Hand (21 nokta × 3) = 63 değer
    Toplam: 225 değer
    """
    # Pose landmarks (33 nokta)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 3)
    
    # Left hand landmarks (21 nokta)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)
    
    # Right hand landmarks (21 nokta)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
    
    # Birleştir: pose + left_hand + right_hand = 225 değer
    keypoints = np.concatenate([pose, left_hand, right_hand])
    
    return keypoints.astype(np.float32)


def check_hand_detected(results):
    """El tespit edildi mi kontrol et"""
    left_detected = results.left_hand_landmarks is not None
    right_detected = results.right_hand_landmarks is not None
    return left_detected or right_detected


def get_hand_confidence(results):
    """El landmark görünürlük skoru"""
    confidence = 0.0
    count = 0
    
    if results.left_hand_landmarks:
        # Sol el landmark'larının ortalama görünürlüğü
        # Not: Hand landmarks'da visibility yok, sadece x,y,z var
        # Bu yüzden el tespit edildi mi diye bakıyoruz
        confidence += 1.0
        count += 1
    
    if results.right_hand_landmarks:
        confidence += 1.0
        count += 1
    
    return confidence / max(count, 1)


# ==================== ANA TEST SINIFI ====================

class AUTSLTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {self.device}")
        
        # Model yükle
        self.model = None
        self.label_map = None
        self.norm_mean = None
        self.norm_std = None
        self.load_model()
        
        # MediaPipe Holistic başlat
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,  # Video mode
            model_complexity=2,  # En yüksek doğruluk
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.hand_frame_count = 0
        self.is_recording = False
        
        # Sonuçlar
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.prediction_history = []
        
    def load_model(self):
        """Model ve normaliation dosyalarını yükle"""
        print("\n" + "="*50)
        print("🚀 AUTSL Transformer Model Yükleniyor...")
        print("="*50)
        
        model_path = MODEL_DIR / 'autsl_pro_final.pt'
        mean_path = MODEL_DIR / 'norm_mean.npy'
        std_path = MODEL_DIR / 'norm_std.npy'
        label_path = MODEL_DIR / 'label_map.json'
        
        # Dosya kontrolü
        for path in [model_path, mean_path, std_path, label_path]:
            if not path.exists():
                print(f"❌ Dosya bulunamadı: {path}")
                return False
        
        try:
            # Normalizasyon parametreleri
            self.norm_mean = np.load(mean_path)
            self.norm_std = np.load(std_path)
            print(f"✅ Normalizasyon yüklendi - Mean shape: {self.norm_mean.shape}")
            
            # Label map
            with open(label_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            print(f"✅ {len(self.label_map)} sınıf yüklendi")
            
            # Model
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            print(f"📊 Model Config:")
            print(f"   - Input Size: {config['input_size']}")
            print(f"   - D_Model: {config['d_model']}")
            print(f"   - Heads: {config['nhead']}")
            print(f"   - Layers: {config['num_layers']}")
            print(f"   - Classes: {config['num_classes']}")
            
            self.model = SignTransformerPro(
                input_size=config['input_size'],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            val_acc = checkpoint.get('val_acc', 0)
            test_acc = checkpoint.get('test_acc', 0)
            print(f"✅ Model yüklendi (Val: {val_acc:.2f}%, Test: {test_acc:.2f}%)")
            print("="*50 + "\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def normalize_sequence(self, sequence):
        """Sequence'i normalize et - eğitim verisiyle aynı şekilde"""
        # Shape: (30, 225)
        # norm_mean ve norm_std shape: (1, 1, 225) veya (225,)
        
        mean = self.norm_mean.squeeze()  # (225,)
        std = self.norm_std.squeeze()    # (225,)
        
        # TEST: Normalizasyonu devre dışı bırak
        # return sequence
        
        # KRITIK: Çok düşük std değerlerini düzelt
        # Eğitim verisinde el yok iken std neredeyse 0 hesaplanmış
        # Minimum 0.1 kullan
        std = np.clip(std, 0.1, None)
        
        normalized = (sequence - mean) / std
        
        # Debug: değerleri kontrol et
        print(f"   Sequence min/max: {sequence.min():.3f} / {sequence.max():.3f}")
        print(f"   Normalized min/max: {normalized.min():.3f} / {normalized.max():.3f}")
        
        return normalized
    
    def predict(self, sequence):
        """30 frame'lik sequence için tahmin yap"""
        # Normalize
        normalized = self.normalize_sequence(sequence)
        
        # Tensor'a çevir
        x = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        
        # Tahmin
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            
            # Top-5 tahmin
            top5_probs, top5_indices = probs.topk(5)
            
            confidence = top5_probs[0][0].item()
            predicted_class = top5_indices[0][0].item()
            label = self.label_map.get(predicted_class, f"Sınıf_{predicted_class}")
            
            # Top-5 sonuçları
            top5_results = []
            for i in range(5):
                idx = top5_indices[0][i].item()
                prob = top5_probs[0][i].item()
                lbl = self.label_map.get(idx, f"Sınıf_{idx}")
                top5_results.append((lbl, prob))
        
        return label, confidence, top5_results
    
    def process_frame(self, frame):
        """Tek frame işle"""
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # MediaPipe işle
        results = self.holistic.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        
        # El tespit kontrolü
        hand_detected = check_hand_detected(results)
        
        # Keypoints çıkar
        keypoints = extract_keypoints(results)
        
        return results, keypoints, hand_detected
    
    def draw_landmarks(self, frame, results):
        """Landmark'ları çiz"""
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1)
            )
        
        # Left Hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
            )
        
        # Right Hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
            )
        
        return frame
    
    def reset(self):
        """Buffer ve sayaçları sıfırla"""
        self.frame_buffer.clear()
        self.hand_frame_count = 0
        self.is_recording = False
    
    def run_webcam(self):
        """Webcam ile canlı test"""
        print("\n🎥 Webcam başlatılıyor...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Kamera açıldı")
        print("\n📋 Kullanım:")
        print("   - Ellerinizi kameraya gösterin")
        print("   - El tespit edilince kayıt başlar")
        print("   - 30 frame dolunca tahmin yapılır")
        print("   - 'R' tuşu: Buffer sıfırla")
        print("   - 'Q' tuşu: Çıkış")
        print("   - 'S' tuşu: Ekran görüntüsü")
        print("-" * 50)
        
        fps_time = time.time()
        frame_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS hesapla
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Frame işle
            results, keypoints, hand_detected = self.process_frame(frame)
            
            # Landmark'ları çiz
            frame = self.draw_landmarks(frame, results)
            
            # Kayıt mantığı
            prediction_text = ""
            confidence = 0.0
            
            if hand_detected:
                if not self.is_recording:
                    # Kayıt başlat
                    self.is_recording = True
                    self.frame_buffer.clear()
                    self.hand_frame_count = 0
                    print("🔴 Kayıt başladı...")
                
                # Frame ekle
                self.frame_buffer.append(keypoints)
                self.hand_frame_count += 1
                
            elif self.is_recording:
                # El yok ama kayıt devam ediyor
                self.frame_buffer.append(keypoints)
                
                # Çok az el tespiti varsa iptal et
                if len(self.frame_buffer) > 15 and self.hand_frame_count < 5:
                    print("⚠️ Yetersiz el tespiti, kayıt iptal")
                    self.reset()
            
            # 30 frame doldu - tahmin yap
            if len(self.frame_buffer) >= SEQUENCE_LENGTH:
                print(f"\n📊 Tahmin yapılıyor... (El frame: {self.hand_frame_count}/{SEQUENCE_LENGTH})")
                
                # Yeterli el tespiti var mı?
                if self.hand_frame_count >= MIN_HAND_FRAMES:
                    sequence = np.array(list(self.frame_buffer), dtype=np.float32)
                    label, conf, top5 = self.predict(sequence)
                    
                    # Threshold kontrolü
                    if conf >= CONFIDENCE_THRESHOLD:
                        prediction_text = label
                        confidence = conf
                        self.last_prediction = label
                        self.last_confidence = conf
                        self.prediction_history.append((label, conf))
                        
                        print(f"✅ Tahmin: {label} ({conf*100:.1f}%)")
                        print(f"   Top-5:")
                        for i, (lbl, prob) in enumerate(top5):
                            print(f"   {i+1}. {lbl}: {prob*100:.1f}%")
                    else:
                        print(f"⚠️ Düşük güven: {label} ({conf*100:.1f}%)")
                        prediction_text = f"{label} (?)"
                        confidence = conf
                else:
                    print(f"⚠️ Yetersiz el tespiti: {self.hand_frame_count}/{MIN_HAND_FRAMES}")
                
                # Reset
                self.reset()
            
            # UI çiz
            self.draw_ui(frame, hand_detected, fps, prediction_text, confidence)
            
            # Göster
            cv2.imshow('AUTSL Test - OpenCV', frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
                print("🔄 Buffer sıfırlandı")
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{int(time.time())}.png', frame)
                print("📸 Ekran görüntüsü kaydedildi")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Çıkış yapıldı")
    
    def draw_ui(self, frame, hand_detected, fps, prediction, confidence):
        """UI elementlerini çiz"""
        h, w = frame.shape[:2]
        
        # Üst panel - koyu arka plan
        cv2.rectangle(frame, (0, 0), (w, 120), (40, 40, 40), -1)
        
        # Başlık
        cv2.putText(frame, "AUTSL Test - Turk Isaret Dili", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps}", (w - 120, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Buffer durumu
        buffer_len = len(self.frame_buffer)
        buffer_text = f"Buffer: {buffer_len}/{SEQUENCE_LENGTH}"
        cv2.putText(frame, buffer_text, (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = 200
        bar_x = 200
        bar_y = 60
        progress = buffer_len / SEQUENCE_LENGTH
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 15), (0, 255, 100), -1)
        
        # El durumu
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        hand_text = f"El: {'ALGILANDI' if hand_detected else 'YOK'} ({self.hand_frame_count})"
        cv2.putText(frame, hand_text, (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        
        # Kayıt durumu
        if self.is_recording:
            cv2.circle(frame, (w - 50, 90), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 80, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Alt panel - tahmin sonucu
        if prediction or self.last_prediction:
            cv2.rectangle(frame, (0, h - 100), (w, h), (40, 40, 40), -1)
            
            display_pred = prediction if prediction else self.last_prediction
            display_conf = confidence if confidence > 0 else self.last_confidence
            
            # Tahmin metni
            cv2.putText(frame, f"Tahmin: {display_pred}", (20, h - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 2)
            
            # Güven skoru
            conf_color = (0, 255, 0) if display_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, f"Guven: {display_conf*100:.1f}%", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
            
            # Threshold bilgisi
            cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", (w - 200, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)


# ==================== ANA FONKSİYON ====================

def main():
    print("\n" + "="*60)
    print("   AUTSL Model Test - OpenCV Tabanlı")
    print("   Türk İşaret Dili Tanıma Sistemi")
    print("="*60)
    
    tester = AUTSLTester()
    
    if tester.model is None:
        print("\n❌ Model yüklenemedi. Çıkış yapılıyor...")
        return
    
    tester.run_webcam()


if __name__ == "__main__":
    main()
