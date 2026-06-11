#!/usr/bin/env python3
"""
AUTSL Görüntüleri ile Transformer Model Testi
==============================================
Bu script AUTSL YOLO görüntülerini kullanarak modeli test eder.
Her sample için 5 frame var, padding ile 30'a tamamlanır.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import json
import yaml
import math
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("c:/Projects/sign_bridge")
MODEL_DIR = BASE_DIR / "autsl_transformer" / "model"
YOLO_DIR = BASE_DIR / "autsl_yolo"
TEST_IMAGES_DIR = YOLO_DIR / "test" / "images"
TEST_LABELS_DIR = YOLO_DIR / "test" / "labels"

# Model Parameters
SEQUENCE_LENGTH = 30
INPUT_DIM = 225  # 75 landmarks * 3 coords
NUM_CLASSES = 226

# ============== MODEL DEFINITION ==============
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

# ============== MEDIAPIPE SETUP ==============
mp_holistic = mp.solutions.holistic

def extract_landmarks(image_bgr):
    """Extract landmarks from an image using MediaPipe."""
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3
    ) as holistic:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        features = []
        
        # Pose landmarks (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 99)
        
        # Left hand landmarks (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
        
        # Right hand landmarks (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
        
        return np.array(features, dtype=np.float32)

def load_class_names():
    """Load class names from data.yaml."""
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def get_sample_label(sample_name):
    """Get the class label for a sample from YOLO label file."""
    # Any frame label will do (e.g., f0)
    label_file = TEST_LABELS_DIR / f"{sample_name}_f0.txt"
    if not label_file.exists():
        return None
    
    with open(label_file, 'r') as f:
        line = f.readline().strip()
        if line:
            class_id = int(line.split()[0])
            return class_id
    return None

def get_unique_samples(limit=20):
    """Get unique sample names from test images."""
    samples = defaultdict(list)
    
    for img_file in TEST_IMAGES_DIR.glob("*.jpg"):
        # Parse filename: signerX_sampleY_fZ.jpg
        name = img_file.stem
        parts = name.rsplit('_f', 1)
        if len(parts) == 2:
            sample_name = parts[0]
            frame_num = int(parts[1])
            samples[sample_name].append((frame_num, img_file))
    
    # Sort frames for each sample
    for sample_name in samples:
        samples[sample_name].sort(key=lambda x: x[0])
    
    # Return first 'limit' samples
    return dict(list(samples.items())[:limit])

def process_sample(sample_name, frame_files, norm_mean, norm_std):
    """Process a sample and extract normalized features."""
    frames = []
    
    for frame_num, img_path in frame_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not read {img_path}")
            frames.append(np.zeros(INPUT_DIM, dtype=np.float32))
            continue
        
        landmarks = extract_landmarks(img)
        frames.append(landmarks)
    
    # Convert to array
    sequence = np.array(frames)  # (num_frames, 225)
    
    # Pad or truncate to SEQUENCE_LENGTH
    if len(sequence) < SEQUENCE_LENGTH:
        # Repeat frames to fill
        while len(sequence) < SEQUENCE_LENGTH:
            sequence = np.concatenate([sequence, sequence], axis=0)
        sequence = sequence[:SEQUENCE_LENGTH]
    else:
        sequence = sequence[:SEQUENCE_LENGTH]
    
    # Normalize
    sequence = (sequence - norm_mean) / norm_std
    
    return sequence

def main():
    print("=" * 60)
    print("AUTSL Görüntüleri ile Transformer Model Testi")
    print("=" * 60)
    
    # Check paths
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return
    
    if not TEST_IMAGES_DIR.exists():
        print(f"ERROR: Test images not found: {TEST_IMAGES_DIR}")
        return
    
    # Load class names
    print("\n[1] Loading class names...")
    class_names = load_class_names()
    print(f"    Loaded {len(class_names)} class names")
    
    # Load normalization parameters
    print("\n[2] Loading normalization parameters...")
    norm_mean = np.load(MODEL_DIR / "norm_mean.npy").squeeze()
    norm_std = np.load(MODEL_DIR / "norm_std.npy").squeeze()
    norm_std = np.clip(norm_std, 0.1, None)  # Clip to avoid division by zero
    print(f"    Mean shape: {norm_mean.shape}, Std shape: {norm_std.shape}")
    
    # Load model
    print("\n[3] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    # Try EMA model first
    ema_path = MODEL_DIR / "autsl_ema_model.pt"
    if ema_path.exists():
        print(f"    Loading EMA model: {ema_path}")
        checkpoint = torch.load(ema_path, map_location=device)
    else:
        checkpoint_path = MODEL_DIR / "autsl_pro_final.pt"
        print(f"    Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    d_model = config.get('d_model', 384)
    nhead = config.get('nhead', 12)
    num_layers = config.get('num_layers', 6)
    
    print(f"    Config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    model = SignTransformerPro(
        input_size=INPUT_DIM,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        dropout=0.0
    ).to(device)
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    model.eval()
    print("    Model loaded successfully!")
    
    # Get test samples
    print("\n[4] Finding test samples...")
    samples = get_unique_samples(limit=20)
    print(f"    Found {len(samples)} unique samples")
    
    # Process each sample
    print("\n[5] Processing samples...")
    print("-" * 60)
    
    correct = 0
    total = 0
    results = []
    
    for sample_name, frame_files in samples.items():
        # Get ground truth
        true_label = get_sample_label(sample_name)
        if true_label is None:
            print(f"  Skipping {sample_name}: no label found")
            continue
        
        true_class = class_names[true_label]
        
        # Process sample
        print(f"\n  Processing: {sample_name} ({len(frame_files)} frames)")
        print(f"    True label: {true_label} = {true_class}")
        
        try:
            sequence = process_sample(sample_name, frame_files, norm_mean, norm_std)
            
            # Predict
            with torch.no_grad():
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                
                top_prob, top_idx = torch.max(probs, dim=1)
                pred_label = top_idx.item()
                pred_prob = top_prob.item()
                pred_class = class_names[pred_label]
            
            # Check if correct
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"    Prediction: {pred_label} = {pred_class} ({pred_prob*100:.1f}%)")
            print(f"    {status}")
            
            # Get top-5
            top5_probs, top5_idx = torch.topk(probs, 5, dim=1)
            print(f"    Top-5 predictions:")
            for i in range(5):
                idx = top5_idx[0, i].item()
                prob = top5_probs[0, i].item()
                name = class_names[idx]
                marker = " <-- TRUE" if idx == true_label else ""
                print(f"      {i+1}. {name}: {prob*100:.1f}%{marker}")
            
            results.append({
                'sample': sample_name,
                'true': true_class,
                'pred': pred_class,
                'correct': is_correct,
                'confidence': pred_prob
            })
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if total > 0:
        accuracy = correct / total * 100
        print(f"Correct: {correct}/{total} = {accuracy:.1f}%")
        
        avg_conf = np.mean([r['confidence'] for r in results])
        print(f"Average confidence: {avg_conf*100:.1f}%")
    else:
        print("No samples processed!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
