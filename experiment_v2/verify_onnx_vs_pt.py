"""
Hizli teshis: best_model.pt ile model_v2.onnx ayni logits'i uretiyor mu?
Eger uretmiyorsa ONNX export bozuk. Uretiyorsa sorun mobil tarafta.
"""
import os, sys, numpy as np, torch, torch.nn as nn, onnxruntime as ort

INPUT_SIZE = 1755
HIDDEN = 256
LAYERS = 2

class GRUModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN, layers=LAYERS, classes=100, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True,
                          dropout=0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, classes),
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])

BASE = os.path.dirname(__file__)
PT   = os.path.join(BASE, "checkpoints", "best_model.pt")
ONNX = os.path.join(BASE, "checkpoints", "model_v2.onnx")

ckpt = torch.load(PT, map_location="cpu", weights_only=False)
label_map = ckpt["label_map"]
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = ckpt["num_classes"]
print(f"PT loaded. classes={num_classes}")

pt_model = GRUModel(classes=num_classes, dropout=0.0)
pt_model.load_state_dict(ckpt["model_state_dict"])
pt_model.eval()

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
print(f"ONNX loaded. input={sess.get_inputs()[0].shape} {sess.get_inputs()[0].type}")
print(f"          output={sess.get_outputs()[0].shape}")

# 3 farkli input ile karsilastir
def top3(logits):
    lg = logits.squeeze()
    probs = torch.softmax(torch.tensor(lg), dim=-1).numpy()
    order = np.argsort(probs)[::-1]
    return [(idx_to_label[int(i)], float(probs[i])) for i in order[:3]]

np.random.seed(42)
test_inputs = {
    "zeros":   np.zeros((1, 30, INPUT_SIZE), dtype=np.float32),
    "random":  np.random.randn(1, 30, INPUT_SIZE).astype(np.float32),
    "ones":    np.ones((1, 30, INPUT_SIZE), dtype=np.float32) * 0.5,
}

for name, inp in test_inputs.items():
    with torch.no_grad():
        pt_out = pt_model(torch.tensor(inp)).numpy()
    onnx_out = sess.run(None, {"input": inp})[0]

    abs_diff = float(np.abs(pt_out - onnx_out).max())
    print(f"\n=== {name} ===")
    print(f"  max|PT-ONNX| = {abs_diff:.6f}")
    print(f"  PT   top3: {top3(pt_out)}")
    print(f"  ONNX top3: {top3(onnx_out)}")
