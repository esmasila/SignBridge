"""
AUTSL Transformer Inference v6
MediaPipe + Transformer Model ile İşaret Dili Tanıma
718 feature: raw(225) + velocity(225) + acceleration(225) + distances(43)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import math
from collections import deque


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


# ==================== PREDICTOR CLASS ====================

# Inter-landmark distance pairs (Colab ile birebir aynı)
DISTANCE_PAIRS = [
    (32, 11), (32, 14), (32, 18),
    (53, 11), (53, 14), (53, 18),
    (36, 11), (36, 14),
    (57, 11), (57, 18),
    (40, 11), (61, 11),
    (32, 53), (36, 57), (40, 61), (44, 65), (48, 69),
    (36, 48), (57, 69), (36, 44), (57, 65),
    (33, 36), (54, 57), (33, 40), (54, 61),
    (32, 0), (53, 0), (32, 2), (53, 2), (32, 3), (53, 3),
    (36, 32), (40, 32), (44, 32), (48, 32),
    (57, 53), (61, 53), (65, 53), (69, 53),
    (36, 0), (57, 0), (32, 6), (53, 6),
]
NUM_DISTANCES = len(DISTANCE_PAIRS)


def compute_distances(seq):
    """seq: (T, 225) -> (T, NUM_DISTANCES)"""
    T = seq.shape[0]
    dists = np.zeros((T, NUM_DISTANCES), dtype=np.float32)
    for i, (a, b) in enumerate(DISTANCE_PAIRS):
        if a * 3 + 2 < seq.shape[1] and b * 3 + 2 < seq.shape[1]:
            ax, ay = seq[:, a * 3], seq[:, a * 3 + 1]
            bx, by = seq[:, b * 3], seq[:, b * 3 + 1]
            dists[:, i] = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + 1e-8)
    return dists


def build_features(sequence):
    """Raw 225 landmarks -> 718 features (velocity + acceleration + distances)"""
    parts = [sequence]  # raw 225
    v = np.zeros_like(sequence)
    v[1:] = sequence[1:] - sequence[:-1]
    parts.append(v)
    a = np.zeros_like(sequence)
    a[2:] = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]
    parts.append(a)
    d = compute_distances(sequence)
    parts.append(d)
    return np.concatenate(parts, axis=-1).astype(np.float32)


class AUTSLPredictor:
    """AUTSL 226 sınıf işaret dili tahmin sınıfı"""
    
    def __init__(self, model_dir=None, device='cpu'):
        """
        Args:
            model_dir: Model dosyalarının bulunduğu klasör
            device: 'cuda' veya 'cpu'
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / 'model'
        else:
            model_dir = Path(model_dir)
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"🖥️ Device: {self.device}")
        
        # Model dosyalarını yükle
        model_path = model_dir / 'best_model.pt'
        if not model_path.exists():
            model_path = model_dir / 'autsl_pro_final.pt'
        mean_path = model_dir / 'norm_mean.npy'
        std_path = model_dir / 'norm_std.npy'
        label_path = model_dir / 'label_map.json'
        feat_config_path = model_dir / 'feature_config.json'
        
        # Dosya kontrolü
        for p in [model_path, mean_path, std_path, label_path]:
            if not p.exists():
                raise FileNotFoundError(f"❌ Dosya bulunamadı: {p}")
        
        # Feature config (v6)
        self.use_v6_features = False
        if feat_config_path.exists():
            with open(feat_config_path, 'r') as f:
                self.feat_config = json.load(f)
            self.use_v6_features = True
            print(f"✅ Feature config yüklendi (size={self.feat_config.get('feature_size', '?')})")
        else:
            self.feat_config = None
        
        # Normalizasyon parametreleri
        self.mean = np.load(mean_path).flatten()
        self.std = np.maximum(np.load(std_path).flatten(), 1e-6)
        print(f"✅ Normalizasyon yüklendi (shape={self.mean.shape})")
        
        # Label map
        with open(label_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        # Key'leri int'e çevir
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        print(f"✅ {len(self.label_map)} sınıf yüklendi")
        
        # Model yükle
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Config: feature_config.json veya checkpoint'tan
        if self.feat_config:
            input_size = self.feat_config.get('feature_size', 718)
            d_model = self.feat_config.get('d_model', 384)
            nhead = self.feat_config.get('nhead', 12)
            num_layers = self.feat_config.get('num_layers', 6)
            num_classes = 226
            dropout = self.feat_config.get('dropout', 0.25)
        elif 'config' in checkpoint:
            config = checkpoint['config']
            input_size = config['input_size']
            d_model = config['d_model']
            nhead = config['nhead']
            num_layers = config['num_layers']
            num_classes = config['num_classes']
            dropout = config.get('dropout', 0.35)
        else:
            input_size = 718
            d_model = 384
            nhead = 12
            num_layers = 6
            num_classes = 226
            dropout = 0.25
        
        self.model = SignTransformerPro(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        val_acc = checkpoint.get('val_acc', 'N/A')
        if isinstance(val_acc, (int, float)):
            print(f"✅ Model yüklendi (Val: {val_acc:.2f}%, input={input_size})")
        else:
            print(f"✅ Model yüklendi (Val: {val_acc}, input={input_size})")
        
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"✅ MediaPipe Holistic başlatıldı")
        
        # Frame buffer (30 frame)
        self.frame_buffer = deque(maxlen=30)
        self.seq_length = 30
        self.raw_feature_size = 225  # MediaPipe raw output
        self.feature_size = input_size  # 718 for v6
        
        print(f"\n🎯 AUTSL Predictor hazır! (226 sınıf, {self.feature_size} feature)")
    
    def extract_landmarks(self, results):
        """MediaPipe sonuçlarından landmark çıkar (225 feature)"""
        landmarks = []
        
        # Pose (33 × 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)
        
        # Left Hand (21 × 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right Hand (21 × 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        return np.array(landmarks, dtype=np.float32)
    
    def process_frame(self, frame):
        """Tek frame işle ve buffer'a ekle"""
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe işle
        results = self.holistic.process(rgb_frame)
        
        # Landmark çıkar
        landmarks = self.extract_landmarks(results)
        
        # Buffer'a ekle
        self.frame_buffer.append(landmarks)
        
        return results
    
    def predict(self):
        """Buffer'daki frame'lerden tahmin yap"""
        if len(self.frame_buffer) < self.seq_length:
            return None, 0.0, len(self.frame_buffer)
        
        # Buffer'ı numpy array'e çevir (30, 225) raw
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)
        
        # v6: Build features (225 -> 718)
        if self.use_v6_features:
            sequence = build_features(sequence)  # (30, 718)
        
        # Normalizasyon
        sequence = (sequence - self.mean) / self.std
        
        # Tensor'a çevir
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, 30, 225)
        
        # Tahmin
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            class_id = predicted.item()
            conf = confidence.item()
            
            # Label al
            label = self.label_map.get(class_id, f"Sınıf_{class_id}")
        
        return label, conf, self.seq_length
    
    def draw_landmarks(self, frame, results):
        """Frame üzerine landmark çiz"""
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Eller
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return frame
    
    def reset_buffer(self):
        """Buffer'ı temizle"""
        self.frame_buffer.clear()
    
    def close(self):
        """Kaynakları serbest bırak"""
        self.holistic.close()


# ==================== WEBCAM DEMO ====================

def run_webcam_demo():
    """Webcam ile gerçek zamanlı demo"""
    print("\n" + "="*50)
    print("🎥 AUTSL Webcam Demo")
    print("="*50)
    print("Kontroller:")
    print("  [SPACE] - Buffer temizle")
    print("  [Q]     - Çıkış")
    print("="*50 + "\n")
    
    # Predictor oluştur
    predictor = AUTSLPredictor()
    
    # Webcam aç
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    print("✅ Webcam açıldı. İşaret yapmaya başlayın...")
    
    last_prediction = ""
    last_confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ayna efekti
        frame = cv2.flip(frame, 1)
        
        # Frame işle
        results = predictor.process_frame(frame)
        
        # Landmark çiz
        frame = predictor.draw_landmarks(frame, results)
        
        # Tahmin
        label, confidence, buffer_len = predictor.predict()
        
        if label is not None and confidence > 0.3:  # Threshold
            last_prediction = label
            last_confidence = confidence
        
        # UI
        # Buffer durumu
        cv2.rectangle(frame, (10, 10), (210, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Buffer: {buffer_len}/{predictor.seq_length}", 
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Tahmin göster
        if last_prediction:
            color = (0, 255, 0) if last_confidence > 0.5 else (0, 255, 255)
            cv2.rectangle(frame, (10, 60), (500, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"{last_prediction} ({last_confidence*100:.1f}%)", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Göster
        cv2.imshow('AUTSL Sign Language Recognition', frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            predictor.reset_buffer()
            last_prediction = ""
            last_confidence = 0.0
            print("🔄 Buffer temizlendi")
    
    cap.release()
    cv2.destroyAllWindows()
    predictor.close()
    print("\n👋 Demo kapatıldı.")


if __name__ == '__main__':
    run_webcam_demo()
