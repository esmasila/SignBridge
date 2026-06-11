"""
AUTSL Model Test - Gelişmiş OpenCV Tabanlı
- El algılanmadığında tahmin durur
- Güven eşiği (threshold) kontrolü
- Koordinat merkezleme ve normalizasyon
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


# ==================== KONFİGÜRASYON ====================

MODEL_DIR = Path(__file__).parent / 'autsl_transformer' / 'model'
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225  # 75 nokta × 3 koordinat

# ÖNEMLİ: Güven eşiği
# 226 sınıf için 0.85 çok yüksek, gerçekçi değerler:
# - 0.15-0.25: Normal threshold (226 sınıf için makul)
# - 0.30-0.40: Yüksek güven
# - 0.85: Sadece çok net işaretler için
CONFIDENCE_THRESHOLD = 0.85  # Kullanıcı isteği

# Alternatif: Relative threshold (en yüksek / ikinci en yüksek oranı)
USE_RELATIVE_THRESHOLD = True
RELATIVE_THRESHOLD = 2.0  # 1. tahmin, 2. tahminden en az 2x yüksek olmalı

# El tespiti gereksinimleri
MIN_HAND_FRAMES = 20  # 30 frame'in en az 20'sinde el olmalı (%67)
CONSECUTIVE_NO_HAND_LIMIT = 10  # 10 frame üst üste el yoksa kayıt iptal


# ==================== MERKEZLİ NORMALİZASYON ====================

def center_and_normalize_landmarks(keypoints):
    """
    Koordinatları merkeze al ve normalize et
    
    Bu fonksiyon:
    1. Pose merkezi (burun veya omuz ortası) bulur
    2. Tüm koordinatları bu merkeze göre kaydırır  
    3. Ölçeklendirme yapar
    """
    keypoints = keypoints.copy()
    
    # Pose landmark'ları (0-98 indeks, 33 nokta × 3)
    pose_x = keypoints[0::3][:33]  # x koordinatları
    pose_y = keypoints[1::3][:33]  # y koordinatları
    
    # Merkez: Burun (indeks 0) veya omuz ortası (11, 12)
    # Burun pozisyonu
    nose_x = keypoints[0]  # Burun x
    nose_y = keypoints[1]  # Burun y
    
    # Omuz ortası (sol omuz: 11, sağ omuz: 12)
    left_shoulder_x = keypoints[11 * 3]
    left_shoulder_y = keypoints[11 * 3 + 1]
    right_shoulder_x = keypoints[12 * 3]
    right_shoulder_y = keypoints[12 * 3 + 1]
    
    # Merkez hesapla (omuz ortası daha stabil)
    if left_shoulder_x > 0 and right_shoulder_x > 0:
        center_x = (left_shoulder_x + right_shoulder_x) / 2
        center_y = (left_shoulder_y + right_shoulder_y) / 2
    elif nose_x > 0:
        center_x = nose_x
        center_y = nose_y
    else:
        # Merkez bulunamadı, olduğu gibi döndür
        return keypoints
    
    # Ölçek: Omuz genişliği
    if left_shoulder_x > 0 and right_shoulder_x > 0:
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)
        scale = shoulder_width if shoulder_width > 0.05 else 0.3  # Min 0.05
    else:
        scale = 0.3  # Default ölçek
    
    # Tüm x,y koordinatlarını merkeze göre kaydır ve ölçekle
    for i in range(75):  # 75 nokta (33 pose + 21 left hand + 21 right hand)
        x_idx = i * 3
        y_idx = i * 3 + 1
        # z koordinatı değişmez (derinlik)
        
        if keypoints[x_idx] != 0 or keypoints[y_idx] != 0:
            keypoints[x_idx] = (keypoints[x_idx] - center_x) / scale
            keypoints[y_idx] = (keypoints[y_idx] - center_y) / scale
    
    return keypoints


# ==================== LANDMARK EXTRACTION ====================

def extract_keypoints(results):
    """
    MediaPipe sonuçlarından keypoint çıkar
    Sıralama: Pose (99) + Left Hand (63) + Right Hand (63) = 225
    """
    # Pose (33 nokta × 3)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(99)
    
    # Left Hand (21 nokta × 3)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(63)
    
    # Right Hand (21 nokta × 3)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(63)
    
    keypoints = np.concatenate([pose, left_hand, right_hand]).astype(np.float32)
    return keypoints


def check_hand_detected(results):
    """En az bir el tespit edildi mi?"""
    return (results.left_hand_landmarks is not None or 
            results.right_hand_landmarks is not None)


def get_hand_quality(results):
    """El tespiti kalitesi (0-2 arası, her el için 1 puan)"""
    quality = 0
    if results.left_hand_landmarks:
        quality += 1
    if results.right_hand_landmarks:
        quality += 1
    return quality


# ==================== ANA TEST SINIFI ====================

class AUTSLAdvancedTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {self.device}")
        
        # Model yükle
        self.model = None
        self.label_map = None
        self.norm_mean = None
        self.norm_std = None
        self.load_model()
        
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.reset_state()
        
    def reset_state(self):
        """Tüm state'i sıfırla"""
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.hand_frame_count = 0
        self.consecutive_no_hand = 0
        self.is_recording = False
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.last_top5 = []
        self.prediction_count = 0
        
    def load_model(self):
        """Model yükle"""
        print("\n" + "="*60)
        print("🚀 AUTSL Model Yükleniyor...")
        print("="*60)
        
        # EMA modeli varsa tercih et
        ema_path = MODEL_DIR / 'autsl_ema_model.pt'
        default_path = MODEL_DIR / 'autsl_pro_final.pt'
        
        model_path = ema_path if ema_path.exists() else default_path
        
        if not model_path.exists():
            print(f"❌ Model bulunamadı: {model_path}")
            return False
            
        try:
            # Normalizasyon
            self.norm_mean = np.load(MODEL_DIR / 'norm_mean.npy').squeeze()
            self.norm_std = np.load(MODEL_DIR / 'norm_std.npy').squeeze()
            self.norm_std = np.clip(self.norm_std, 0.1, None)  # Min 0.1
            
            # Label map
            with open(MODEL_DIR / 'label_map.json', 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            
            # Model
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            self.model = SignTransformerPro(**config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model: {model_path.name}")
            print(f"✅ Sınıf sayısı: {len(self.label_map)}")
            print(f"✅ Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            print(f"📊 Güven Eşiği: {CONFIDENCE_THRESHOLD*100:.0f}%")
            print(f"📊 Relative Threshold: {'Aktif' if USE_RELATIVE_THRESHOLD else 'Kapalı'}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Hata: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def normalize_sequence(self, sequence, use_centering=True):
        """
        Sequence normalize et
        use_centering: Her frame'i merkeze göre kaydır
        """
        if use_centering:
            # Her frame'i merkeze al
            centered_sequence = np.zeros_like(sequence)
            for i in range(len(sequence)):
                centered_sequence[i] = center_and_normalize_landmarks(sequence[i])
            sequence = centered_sequence
        
        # Z-score normalizasyonu (eğitim istatistikleriyle)
        normalized = (sequence - self.norm_mean) / self.norm_std
        return normalized
    
    def predict(self, sequence, use_centering=True):
        """Tahmin yap ve filtreleme uygula"""
        normalized = self.normalize_sequence(sequence, use_centering)
        
        x = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            
            # Top-5 al
            top5_probs, top5_indices = probs.topk(5)
            
            top1_prob = top5_probs[0][0].item()
            top2_prob = top5_probs[0][1].item()
            predicted_class = top5_indices[0][0].item()
            label = self.label_map.get(predicted_class, f"Class_{predicted_class}")
            
            # Top-5 listesi
            top5 = []
            for i in range(5):
                idx = top5_indices[0][i].item()
                prob = top5_probs[0][i].item()
                lbl = self.label_map.get(idx, f"Class_{idx}")
                top5.append((lbl, prob))
        
        # Threshold kontrolü
        passed_threshold = False
        reason = ""
        
        # Absolute threshold
        if top1_prob >= self.confidence_threshold:
            passed_threshold = True
            reason = f"Güven ≥ {self.confidence_threshold*100:.0f}%"
        
        # Relative threshold (alternatif)
        if USE_RELATIVE_THRESHOLD and not passed_threshold:
            if top2_prob > 0 and (top1_prob / top2_prob) >= RELATIVE_THRESHOLD:
                passed_threshold = True
                reason = f"Relative: {top1_prob/top2_prob:.1f}x"
        
        return {
            'label': label,
            'confidence': top1_prob,
            'top5': top5,
            'passed_threshold': passed_threshold,
            'reason': reason,
            'top2_ratio': top1_prob / top2_prob if top2_prob > 0 else float('inf')
        }
    
    def process_frame(self, frame):
        """Tek frame işle"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        hand_detected = check_hand_detected(results)
        hand_quality = get_hand_quality(results)
        keypoints = extract_keypoints(results)
        
        return results, keypoints, hand_detected, hand_quality
    
    def draw_landmarks(self, frame, results):
        """Landmark çiz"""
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1)
            )
        
        # Left Hand - Yeşil
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
            )
        
        # Right Hand - Mavi
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
            )
        
        return frame
    
    def run_webcam(self):
        """Ana webcam döngüsü"""
        print("🎥 Kamera başlatılıyor...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("📋 KONTROLLER")
        print("="*60)
        print("  [R] - Buffer sıfırla")
        print("  [C] - Merkezleme aç/kapa")
        print("  [T] - Threshold değiştir") 
        print("  [S] - Ekran görüntüsü")
        print("  [Q] - Çıkış")
        print("="*60)
        print("\n⏳ El gösterene kadar bekleniyor...")
        
        fps_time = time.time()
        frame_count = 0
        fps = 0
        use_centering = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Frame işle
            results, keypoints, hand_detected, hand_quality = self.process_frame(frame)
            
            # Landmark çiz
            frame = self.draw_landmarks(frame, results)
            
            # State machine
            prediction_result = None
            status_msg = ""
            
            if hand_detected:
                self.consecutive_no_hand = 0
                
                if not self.is_recording:
                    # Kayıt başlat
                    self.is_recording = True
                    self.frame_buffer.clear()
                    self.hand_frame_count = 0
                    print("\n🔴 Kayıt başladı - İşaret yapın...")
                    status_msg = "KAYIT BAŞLADI"
                
                # Frame ekle
                self.frame_buffer.append(keypoints)
                self.hand_frame_count += 1
                status_msg = f"Kayıt: {len(self.frame_buffer)}/30"
                
            else:
                # El yok
                if self.is_recording:
                    self.consecutive_no_hand += 1
                    self.frame_buffer.append(keypoints)  # Yine de ekle
                    
                    # Çok uzun süre el yoksa iptal
                    if self.consecutive_no_hand >= CONSECUTIVE_NO_HAND_LIMIT:
                        print("⚠️ El çok uzun süre algılanmadı - kayıt iptal")
                        self.reset_state()
                        status_msg = "KAYIT İPTAL"
                else:
                    status_msg = "Ellerinizi gösterin..."
            
            # 30 frame doldu
            if len(self.frame_buffer) >= SEQUENCE_LENGTH and self.is_recording:
                # El tespiti kontrolü
                if self.hand_frame_count < MIN_HAND_FRAMES:
                    print(f"⚠️ Yetersiz el tespiti: {self.hand_frame_count}/{MIN_HAND_FRAMES}")
                    self.reset_state()
                else:
                    # Tahmin yap
                    sequence = np.array(list(self.frame_buffer), dtype=np.float32)
                    prediction_result = self.predict(sequence, use_centering)
                    
                    self.last_prediction = prediction_result['label']
                    self.last_confidence = prediction_result['confidence']
                    self.last_top5 = prediction_result['top5']
                    self.prediction_count += 1
                    
                    if prediction_result['passed_threshold']:
                        print(f"\n✅ [{self.prediction_count}] {prediction_result['label']}")
                        print(f"   Güven: {prediction_result['confidence']*100:.1f}%")
                        print(f"   Reason: {prediction_result['reason']}")
                    else:
                        print(f"\n⚠️ [{self.prediction_count}] {prediction_result['label']} (düşük güven)")
                        print(f"   Güven: {prediction_result['confidence']*100:.1f}% < {self.confidence_threshold*100:.0f}%")
                    
                    print(f"   Top-3: {', '.join([f'{l}: {p*100:.1f}%' for l, p in prediction_result['top5'][:3]])}")
                    
                    self.reset_state()
            
            # UI çiz
            self.draw_ui(frame, hand_detected, hand_quality, fps, status_msg, use_centering)
            
            cv2.imshow('AUTSL Advanced Test', frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_state()
                print("🔄 Reset")
            elif key == ord('c'):
                use_centering = not use_centering
                print(f"📍 Merkezleme: {'Aktif' if use_centering else 'Kapalı'}")
            elif key == ord('t'):
                self.confidence_threshold = 0.25 if self.confidence_threshold > 0.5 else 0.85
                print(f"📊 Threshold: {self.confidence_threshold*100:.0f}%")
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{int(time.time())}.png', frame)
                print("📸 Kaydedildi")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_ui(self, frame, hand_detected, hand_quality, fps, status, use_centering):
        """UI elementleri"""
        h, w = frame.shape[:2]
        
        # Üst panel
        cv2.rectangle(frame, (0, 0), (w, 140), (30, 30, 30), -1)
        
        # Başlık
        cv2.putText(frame, "AUTSL Test - Turk Isaret Dili", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        
        # FPS ve ayarlar
        cv2.putText(frame, f"FPS: {fps}", (w - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Threshold: {self.confidence_threshold*100:.0f}%", (w - 150, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, f"Center: {'ON' if use_centering else 'OFF'}", (w - 100, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Buffer durumu
        buffer_len = len(self.frame_buffer)
        cv2.putText(frame, f"Buffer: {buffer_len}/30", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bar
        bar_w = 250
        bar_x = 180
        bar_y = 60
        progress = buffer_len / SEQUENCE_LENGTH
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 15), 
                      (0, 255, 100) if progress < 1 else (0, 200, 255), -1)
        
        # El durumu
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 200)
        hand_text = f"El: {'✓' if hand_detected else '✗'} ({self.hand_frame_count})"
        cv2.putText(frame, hand_text, (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        
        # El kalitesi
        quality_text = f"Kalite: {'●' * hand_quality}{'○' * (2 - hand_quality)}"
        cv2.putText(frame, quality_text, (180, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Status
        if status:
            cv2.putText(frame, status, (350, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Kayıt göstergesi
        if self.is_recording:
            cv2.circle(frame, (w - 40, 110), 12, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 75, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alt panel - tahmin sonuçları
        if self.last_prediction:
            cv2.rectangle(frame, (0, h - 150), (w, h), (30, 30, 30), -1)
            
            # Tahmin
            conf_color = (0, 255, 100) if self.last_confidence >= self.confidence_threshold else (0, 150, 255)
            cv2.putText(frame, f"Tahmin: {self.last_prediction}", (20, h - 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 2)
            
            # Güven
            cv2.putText(frame, f"Guven: {self.last_confidence*100:.1f}%", (20, h - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
            
            # Top-5
            if self.last_top5:
                top5_text = " | ".join([f"{l}: {p*100:.0f}%" for l, p in self.last_top5[:5]])
                cv2.putText(frame, f"Top-5: {top5_text}", (20, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("   AUTSL Advanced Test - Gelişmiş OpenCV")
    print("   - El algılama kontrolü")
    print("   - Güven eşiği (threshold)")
    print("   - Koordinat merkezleme")
    print("="*60)
    
    tester = AUTSLAdvancedTester()
    
    if tester.model is None:
        print("\n❌ Model yüklenemedi!")
        return
    
    tester.run_webcam()


if __name__ == "__main__":
    main()
