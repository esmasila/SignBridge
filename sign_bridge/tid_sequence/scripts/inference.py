"""
Real-time TID Sequence Inference
Webcam'den canlı tahmin yap
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
import sys
import json

# Kendi modüllerimizi import et
sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import LSTMClassifier, GRUClassifier


class TIDSequencePredictor:
    """Real-time TID sequence prediction"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: model checkpoint (.pt file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Model info
        model_type = checkpoint['model_type']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        num_classes = checkpoint['num_classes']
        self.label_map = checkpoint['label_map']
        
        # Reverse label map (idx -> name)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        
        print(f"Model: {model_type.upper()}")
        print(f"Classes: {list(self.label_map.keys())}")
        print(f"Device: {self.device}\n")
        
        # Create model
        if model_type == 'lstm':
            self.model = LSTMClassifier(
                input_size=1629,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes
            )
        elif model_type == 'gru':
            self.model = GRUClassifier(
                input_size=1629,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("✅ Model loaded!\n")
    
    def extract_landmarks(self, results):
        """MediaPipe sonuçlarından landmark'ları çıkar"""
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
    
    def predict_sequence(self, sequence):
        """
        Bir sekansı tahmin et
        
        Args:
            sequence: numpy array (60, 1629)
        Returns:
            predicted_label: str
            confidence: float (0-1)
            all_probs: dict {label: prob}
        """
        # Normalize
        seq_min = sequence.min()
        seq_max = sequence.max()
        if seq_max - seq_min > 0:
            sequence = (sequence - seq_min) / (seq_max - seq_min)
        
        # To tensor
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, 60, 1629)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(seq_tensor)  # (1, num_classes)
            probs = torch.softmax(outputs, dim=1)[0]  # (num_classes,)
        
        # Get results
        confidence, pred_idx = torch.max(probs, 0)
        predicted_label = self.idx_to_label[pred_idx.item()]
        
        # All probabilities
        all_probs = {self.idx_to_label[i]: probs[i].item() for i in range(len(probs))}
        
        return predicted_label, confidence.item(), all_probs
    
    def draw_landmarks(self, image, results):
        """Landmark'ları görselleştir"""
        # Yüz
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1)
            )
        
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2),
                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2)
            )
        
        # Eller
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2),
                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )
    
    def run_live(self, max_frames=60):
        """
        Canlı kameradan tahmin yap
        
        Args:
            max_frames: kaç frame toplanacak (default 60)
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        sequence = []
        recording = False
        predicted_label = ""
        confidence = 0.0
        
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            print("🎥 Camera opened!")
            print("Controls:")
            print("  SPACE: Start/Stop recording")
            print("  R: Reset sequence")
            print("  Q/ESC: Quit\n")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                landmarks = self.extract_landmarks(results)
                
                # Recording
                if recording:
                    sequence.append(landmarks)
                    
                    if len(sequence) >= max_frames:
                        # Predict
                        seq_array = np.array(sequence[:max_frames])  # (60, 1629)
                        predicted_label, confidence, all_probs = self.predict_sequence(seq_array)
                        
                        print(f"\n🎯 Prediction: {predicted_label} ({confidence*100:.1f}%)")
                        print("All probabilities:")
                        for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {label}: {prob*100:.1f}%")
                        
                        recording = False
                
                # Visualization
                self.draw_landmarks(frame, results)
                
                # UI
                status = "RECORDING" if recording else "READY"
                color = (0, 0, 255) if recording else (0, 255, 0)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Frames: {len(sequence)}/{max_frames}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if predicted_label:
                    # Tahmin sonucu
                    cv2.putText(frame, f"Result: {predicted_label}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(frame, "SPACE:Record R:Reset Q:Quit", (10, frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('TID Sequence Prediction', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space
                    if not recording:
                        recording = True
                        sequence = []
                        predicted_label = ""
                        print("🔴 Recording started...")
                    else:
                        recording = False
                        print("⏸ Recording stopped")
                
                elif key == ord('r') or key == ord('R'):
                    sequence = []
                    predicted_label = ""
                    confidence = 0.0
                    print("🔄 Reset")
                
                elif key == ord('q') or key == ord('Q') or key == 27:
                    break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Checkpoint path
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "lstm_best.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Create predictor
    predictor = TIDSequencePredictor(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    
    # Run live
    predictor.run_live(max_frames=60)
