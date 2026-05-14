"""
best_model_v2.pt -> model_v2.onnx (input_size=1755)
Faz B: Telefon uzerinde onnxruntime ile calistirmak icin.
"""
import torch
import torch.nn as nn
import os

INPUT_SIZE = 1755   # v2: 1629 landmark + 126 velocity
HIDDEN     = 256
LAYERS     = 2

class GRUModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN, layers=LAYERS, classes=100, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, classes),
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])

BASE = os.path.dirname(__file__)
# v2 weights
CKPT = os.path.join(BASE, "checkpoints", "best_model_v2.pt")
if not os.path.exists(CKPT):
    # fallback: best_model.pt (v2 oldu ama v2 yoksa)
    CKPT = os.path.join(BASE, "checkpoints", "best_model.pt")

print(f"Yuklenen checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

num_classes = ckpt.get("num_classes", 100)
print(f"Sinif sayisi: {num_classes}")

model = GRUModel(classes=num_classes, dropout=0.0)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dummy = torch.randn(1, 30, INPUT_SIZE)

onnx_path = os.path.join(BASE, "checkpoints", "model_v2.onnx")
# NOT: dynamic_axes KULLANMIYORUZ.
# PyTorch uyarisi: "Exporting a model to ONNX with a batch_size other than 1,
# with a variable length with GRU can cause an error when running the ONNX
# model with a different batch size."
# Telefonda her zaman [1, 30, 1755] ile calistiracagiz, bu yuzden sabit shape en guvenli yol.
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=14,
    do_constant_folding=True,
)

size_kb = os.path.getsize(onnx_path) / 1024
print(f"\nOK: {onnx_path} ({size_kb:.0f} KB)")

# Hizli sagilik kontrolu
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})
    print(f"ONNX inference OK. output shape: {out[0].shape}  sinif sayisi: {out[0].shape[1]}")
    print(f"Top-1 (rastgele input): class={out[0].argmax()}")
except ImportError:
    print("(onnxruntime yukletilmemis, dogrulama atlandi — pip install onnxruntime)")
