"""
GRU modelini TFLite formatına çevir
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os

# Model tanımı (pipeline.py'den kopyala)
class GRUModel(nn.Module):
    def __init__(self, input_size=1629, hidden=256, layers=2, classes=100, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, classes)
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])

# Checkpoint yükle
ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

num_classes = ckpt.get('num_classes', 100)
print(f"Sinif sayisi: {num_classes}")

model = GRUModel(classes=num_classes, dropout=0.0)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ONNX'e çevir
dummy = torch.randn(1, 30, 1629)  # batch=1, seq_len=30, features=1629
onnx_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model.onnx")

torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {1: 'seq_len'}},
    opset_version=13
)
print(f"ONNX kaydedildi: {onnx_path}")

# ONNX -> TFLite
try:
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    tf_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model_tf")
    tf_rep.export_graph(tf_path)
    print(f"TF model kaydedildi: {tf_path}")

    # TFLite'a çevir
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite kaydedildi: {tflite_path} ({len(tflite_model)/1024:.0f} KB)")

except ImportError:
    print("\nonnx-tf veya tensorflow kurulu degil. Alternatif yol deneniyor...")

    # Alternatif: PyTorch -> TFLite via ai_edge_torch
    try:
        import ai_edge_torch

        sample = torch.randn(1, 30, 1629)
        edge_model = ai_edge_torch.convert(model, (sample,))

        tflite_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model.tflite")
        edge_model.export(tflite_path)
        print(f"TFLite kaydedildi: {tflite_path}")
    except ImportError:
        print("\nai_edge_torch da kurulu degil.")
        print("Lutfen su komutu calistirin:")
        print("  pip install onnx onnx-tf tensorflow")
        print("  veya")
        print("  pip install ai-edge-torch")
        print("\nSonra bu scripti tekrar calistirin.")

print("\nBitti!")
