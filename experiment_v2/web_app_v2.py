"""
experiment_v2 Web Arayuzu — V2 (Normalizasyon + Velocity + Fast-Track)

Bu KOPYA, calisan web_app.py'yi BOZMADAN v2 modelini test etmek icin.
  - Model: best_model_v2.pt  (finetune.py ile egitilir)
  - Input: 1755 (1629 normalize + 126 el velocity)
  - Normalize: omuz-merkezli + bilek-merkezli (2 katman)
  - Velocity: frame farki ile hareket yonu/hizi
  - Fast-track: yuksek guvende hemen ekle
  - Port: 5052

Orijinal sistem (web_app.py + best_model.pt) tamamen dokunulmadan kaldi.

Baslat:  python experiment_v2/web_app_v2.py
Adres:   http://localhost:5052
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import base64
import time
import json
from pathlib import Path
from collections import deque
from flask import Flask, render_template_string, Response, request, jsonify
from flask_socketio import SocketIO, emit

BASE    = Path(__file__).parent
CKPT    = BASE / "checkpoints" / "best_model_v2.pt"   # V2 modeli
SEQ_LEN = 30

# ---- Boyutlar (finetune.py ile AYNI) ----
FACE_END  = 468 * 3          # 1404
POSE_END  = FACE_END + 33*3  # 1503
LEFT_END  = POSE_END + 21*3  # 1566
RIGHT_END = LEFT_END + 21*3  # 1629
HAND_VEL  = 21*3 + 21*3      # 126
INPUT_SIZE = RIGHT_END + HAND_VEL  # 1755

# ---- Model ----
class GRUModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=256, layers=2, classes=3, dropout=0.3):
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

if not CKPT.exists():
    raise FileNotFoundError(
        f"V2 modeli bulunamadi: {CKPT}\n"
        f"Once egit: python experiment_v2/finetune.py"
    )

ckpt         = torch.load(CKPT, map_location="cpu")
label_map    = ckpt["label_map"]
num_classes  = ckpt["num_classes"]
idx_to_label = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GRUModel(classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"V2 Model yuklendi | {num_classes} kelime | {device} | input={INPUT_SIZE}")

# ---- MediaPipe ----
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_landmarks(results):
    lm = []
    if results.face_landmarks:
        for p in results.face_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 468 * 3)
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 33 * 3)
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    return np.array(lm, dtype=np.float32)

# ---- 2 Katmanli Normalizasyon (finetune.py ile BIREBIR AYNI!) ----
def normalize_landmarks(lm):
    result = lm.copy()

    # KATMAN 1: Omuz merkezli
    pose = result[FACE_END:POSE_END].reshape(33, 3)
    ls, rs = pose[11].copy(), pose[12].copy()
    if not (np.allclose(ls, 0) and np.allclose(rs, 0)):
        center = rs.copy() if np.allclose(ls, 0) else (
                 ls.copy() if np.allclose(rs, 0) else (ls + rs) / 2.0)
        scale = float(np.linalg.norm(rs - ls))
        if scale >= 1e-4:
            all_lm = result.reshape(-1, 3)
            all_lm = (all_lm - center) / scale
            result = all_lm.flatten()

    # KATMAN 2: Sol el bilek merkezli
    lh = result[POSE_END:LEFT_END].reshape(21, 3)
    if not np.allclose(lh[0], 0):
        wrist      = lh[0].copy()
        palm_scale = np.linalg.norm(lh[9] - lh[0])
        if palm_scale > 1e-4:
            lh[1:] = (lh[1:] - wrist) / palm_scale
            result[POSE_END:LEFT_END] = lh.flatten()

    # KATMAN 2: Sag el bilek merkezli
    rh = result[LEFT_END:RIGHT_END].reshape(21, 3)
    if not np.allclose(rh[0], 0):
        wrist      = rh[0].copy()
        palm_scale = np.linalg.norm(rh[9] - rh[0])
        if palm_scale > 1e-4:
            rh[1:] = (rh[1:] - wrist) / palm_scale
            result[LEFT_END:RIGHT_END] = rh.flatten()

    return result

# ---- Velocity (finetune.py ile BIREBIR AYNI!) ----
def add_velocity(seq):
    """(T, 1629) -> (T, 1755): her frame'e el velocity (126) eklenir."""
    hands = seq[:, POSE_END:]
    vel   = np.zeros_like(hands)
    vel[1:] = hands[1:] - hands[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)

# ---- Uygulama durumu ----
state = {
    "buffer":        deque(maxlen=SEQ_LEN),    # normalize EDILMIS landmark
    "smooth_counts": {k: 0 for k in label_map},
    "sentence":      [],
    "current_word":  "",
    "current_conf":  0.0,
    "last_add_time": 0.0,
    "last_hand_time": 0.0,
    "running":       False,
    "cap":           None,
    "holistic":      None,
}

# ---- Fast-track parametreleri ----
CONF_THRESHOLD  = 0.82   # normal esik
CONF_FAST_TRACK = 0.94   # bu ustunde hizli ekle
SMOOTH_NEEDED   = 4      # normal yol: 4 frame ust uste
SMOOTH_FAST     = 2      # fast yol: 2 frame yeter
MARGIN_FAST     = 0.30   # top1 - top2 farki fast icin
COOLDOWN_SEC    = 0.5    # kelimeler arasi kisa bekleme
HAND_GONE_SEC   = 2.0

# ---- Flask ----
app = Flask(__name__)
app.config["SECRET_KEY"] = "signbridge_v2_copy"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignBridge V2 — Deney</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #0a1a0f; color: #e0e0e0; min-height: 100vh; }
  header {
    background: linear-gradient(135deg, #0f2a1a, #102e1f);
    padding: 16px 32px; display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #1a4a2a;
  }
  header h1 { font-size: 1.5rem; color: #6ee7b7; }
  header .tag { background:#10b981;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:700; }
  header span.sub { font-size: 0.85rem; color: #6b7280; }
  .pill {
    background: #0f2a1a; border: 1px solid #1a4a2a; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; color: #6ee7b7;
  }
  .main { display: grid; grid-template-columns: 1fr 380px; gap: 20px; padding: 20px; max-width: 1300px; margin: 0 auto; }
  .camera-section { display: flex; flex-direction: column; gap: 16px; }
  .camera-wrapper { position: relative; background: #111; border-radius: 16px; overflow: hidden; border: 1px solid #1a4a2a; aspect-ratio: 4/3; }
  #cameraFeed { width: 100%; height: 100%; object-fit: cover; display: block; }
  .camera-placeholder { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: #4b5563; }
  .pred-overlay {
    position: absolute; top: 0; left: 0; right: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0.75) 0%, transparent 100%);
    padding: 16px 20px; display: flex; align-items: center; justify-content: space-between;
  }
  #predWord { font-size: 2rem; font-weight: 700; color: #6ee7b7; letter-spacing: 2px; }
  #predConf { font-size: 1.1rem; color: #a7f3d0; font-weight: 600; }
  .conf-bar-wrap { position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: rgba(255,255,255,0.1); }
  #confBar { height: 100%; background: linear-gradient(90deg, #10b981, #06b6d4); transition: width 0.15s ease; width: 0%; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; }
  button { padding: 10px 20px; border: none; border-radius: 10px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  #btnStart { background: linear-gradient(135deg, #10b981, #059669); color: white; flex: 1; }
  #btnStart:hover { opacity: 0.85; transform: translateY(-1px); }
  #btnStart.active { background: linear-gradient(135deg, #dc2626, #b91c1c); }
  #btnClear { background: #0f2a1a; color: #a7f3d0; border: 1px solid #1a4a2a; }
  .side-panel { display: flex; flex-direction: column; gap: 16px; }
  .card { background: #0f2a1a; border: 1px solid #1a4a2a; border-radius: 16px; padding: 20px; }
  .card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; margin-bottom: 14px; }
  #sentenceBox { min-height: 90px; background: #0a1a0f; border-radius: 10px; padding: 14px; font-size: 1.15rem; line-height: 1.6; color: #e2e8f0; border: 1px solid #1a4a2a; word-break: break-word; }
  #sentenceBox.empty { color: #4b5563; font-style: italic; }
  #wordList { display: flex; flex-wrap: wrap; gap: 6px; }
  .word-chip { background: #0a1a0f; border: 1px solid #1a4a2a; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; color: #a7f3d0; transition: all 0.3s; }
  .word-chip.active { background: #10b981; border-color: #10b981; color: white; transform: scale(1.05); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4b5563; display: inline-block; margin-right: 6px; }
  .status-dot.active { background: #10b981; box-shadow: 0 0 8px #10b981; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
</head>
<body>
<header>
  <h1>SignBridge</h1>
  <span class="tag">V2 DENEY</span>
  <span class="sub">Normalize + Velocity + Fast-Track</span>
  <div class="pill" id="modelInfo">Yukleniyor...</div>
  <div style="margin-left:auto; display:flex; align-items:center; font-size:0.85rem; color:#6b7280;">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">Hazir</span>
  </div>
</header>

<div class="main">
  <div class="camera-section">
    <div class="camera-wrapper">
      <div class="camera-placeholder" id="placeholder">
        <p>Kamerayi baslatmak icin butona bas</p>
      </div>
      <img id="cameraFeed" style="display:none;" alt="Kamera">
      <div class="pred-overlay" style="display:none;" id="predOverlay">
        <span id="predWord">—</span>
        <span id="predConf">%0</span>
      </div>
      <div class="conf-bar-wrap"><div id="confBar"></div></div>
    </div>

    <div class="controls">
      <button id="btnStart" onclick="toggleCamera()">Kamerayi Baslat</button>
      <button id="btnClear" onclick="clearSentence()">Temizle</button>
    </div>
  </div>

  <div class="side-panel">
    <div class="card">
      <div class="card-title">Isaret Kelimeleri</div>
      <div id="sentenceBox" class="empty">Henuz kelime algilanmadi...</div>
    </div>
    <div class="card">
      <div class="card-title">Bilinen Kelimeler (<span id="wordCount">0</span>)</div>
      <div id="wordList"></div>
    </div>
  </div>
</div>

<script>
const socket = io();
let isRunning = false;
let sentence = [];

socket.emit("get_info");
socket.on("info", (data) => {
  document.getElementById("modelInfo").textContent = data.num_classes + " kelime | V2";
  document.getElementById("wordCount").textContent = data.num_classes;
  const wl = document.getElementById("wordList");
  wl.innerHTML = "";
  data.words.forEach(w => {
    const chip = document.createElement("div");
    chip.className = "word-chip";
    chip.id = "chip_" + w;
    chip.textContent = w;
    wl.appendChild(chip);
  });
});

socket.on("prediction", (data) => {
  document.getElementById("predWord").textContent = data.word || "—";
  document.getElementById("predConf").textContent = "%" + Math.round(data.conf * 100);
  document.getElementById("confBar").style.width = (data.conf * 100) + "%";

  document.querySelectorAll(".word-chip").forEach(c => c.classList.remove("active"));
  const chip = document.getElementById("chip_" + data.word);
  if (chip && data.conf >= 0.75) chip.classList.add("active");

  sentence = data.sentence;
  updateSentenceBox();
});

socket.on("frame", (data) => {
  document.getElementById("cameraFeed").src = "data:image/jpeg;base64," + data.img;
});

function updateSentenceBox() {
  const box = document.getElementById("sentenceBox");
  if (sentence.length === 0) {
    box.textContent = "Henuz kelime algilanmadi...";
    box.classList.add("empty");
  } else {
    box.textContent = sentence.join(" ");
    box.classList.remove("empty");
  }
}

function toggleCamera() {
  if (!isRunning) {
    socket.emit("start_camera");
    isRunning = true;
    document.getElementById("btnStart").textContent = "Kamerayi Durdur";
    document.getElementById("btnStart").classList.add("active");
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("cameraFeed").style.display = "block";
    document.getElementById("predOverlay").style.display = "flex";
    document.getElementById("statusDot").classList.add("active");
    document.getElementById("statusText").textContent = "Canli";
  } else {
    socket.emit("stop_camera");
    isRunning = false;
    document.getElementById("btnStart").textContent = "Kamerayi Baslat";
    document.getElementById("btnStart").classList.remove("active");
    document.getElementById("placeholder").style.display = "flex";
    document.getElementById("cameraFeed").style.display = "none";
    document.getElementById("predOverlay").style.display = "none";
    document.getElementById("statusDot").classList.remove("active");
    document.getElementById("statusText").textContent = "Durduruldu";
  }
}

function clearSentence() {
  socket.emit("clear_sentence");
  sentence = [];
  updateSentenceBox();
}

document.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") clearSentence();
});
</script>
</body>
</html>
"""

# ---- SocketIO olaylar ----
@socketio.on("connect")
def handle_connect():
    state["running"] = False
    if state["cap"] is not None:
        try: state["cap"].release()
        except: pass
        state["cap"] = None
    state["sentence"] = []
    state["buffer"] = deque(maxlen=SEQ_LEN)
    state["smooth_counts"] = {k: 0 for k in label_map}
    state["current_word"] = ""
    state["current_conf"] = 0.0
    state["last_add_time"] = 0.0

@socketio.on("get_info")
def handle_info():
    emit("info", {
        "num_classes": num_classes,
        "words": list(label_map.keys()),
    })

@socketio.on("clear_sentence")
def handle_clear():
    state["sentence"] = []
    state["smooth_counts"] = {k: 0 for k in label_map}

@socketio.on("start_camera")
def handle_start():
    if state["running"]:
        return
    state["running"] = True
    state["buffer"]  = deque(maxlen=SEQ_LEN)
    state["smooth_counts"] = {k: 0 for k in label_map}

    cap = cv2.VideoCapture(0)
    state["cap"] = cap
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,            # daha guclu (yavas ama hassas)
        min_detection_confidence=0.3,  # dusuk esik -> kapali/ortuk elleri de yakala
        min_tracking_confidence=0.3,
        smooth_landmarks=True,
        refine_face_landmarks=False
    )
    state["holistic"] = holistic

    import threading
    def camera_loop():
        mp_draw = mp.solutions.drawing_utils
        try:
            while state["running"]:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS)

                lm = extract_landmarks(results)
                lm_norm = normalize_landmarks(lm)     # KATMAN 1+2
                state["buffer"].append(lm_norm)

                hand_visible = (results.left_hand_landmarks is not None or
                                results.right_hand_landmarks is not None)

                now_t = time.time()
                if hand_visible:
                    state["last_hand_time"] = now_t

                pred_word = state["current_word"]
                pred_conf = 0.0
                top2_margin = 0.0
                added     = False

                if len(state["buffer"]) == SEQ_LEN:
                    seq = np.array(state["buffer"], dtype=np.float32)  # (30, 1629)
                    seq = add_velocity(seq)                             # (30, 1755)
                    seq = np.clip(seq, -5, 5)
                    inp = torch.tensor(seq).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out  = model(inp)
                        prob = torch.softmax(out, dim=1)[0]
                        top_vals, top_idx = prob.topk(2)
                        pred_conf   = top_vals[0].item()
                        top2_margin = (top_vals[0] - top_vals[1]).item()
                        pred_word   = idx_to_label[top_idx[0].item()]

                    state["current_word"] = pred_word
                    state["current_conf"] = pred_conf

                    now = time.time()
                    in_cooldown = (now - state["last_add_time"]) < COOLDOWN_SEC

                    # Fast-track: yuksek guven + net margin → hizli ekle
                    is_fast = (pred_conf >= CONF_FAST_TRACK and top2_margin >= MARGIN_FAST)
                    needed  = SMOOTH_FAST if is_fast else SMOOTH_NEEDED
                    threshold = CONF_THRESHOLD

                    if pred_conf >= threshold and hand_visible and not in_cooldown:
                        state["smooth_counts"][pred_word] += 1
                        for k in state["smooth_counts"]:
                            if k != pred_word:
                                state["smooth_counts"][k] = max(0, state["smooth_counts"][k] - 1)

                        if state["smooth_counts"][pred_word] >= needed:
                            if not (state["sentence"] and state["sentence"][-1] == pred_word):
                                if len(state["sentence"]) >= 20:
                                    state["sentence"].pop(0)
                                state["sentence"].append(pred_word)
                                added = True
                            state["last_add_time"] = now
                            state["smooth_counts"] = {k: 0 for k in label_map}
                    else:
                        if not hand_visible or in_cooldown:
                            state["smooth_counts"] = {k: 0 for k in label_map}

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_b64 = base64.b64encode(buf).decode("utf-8")

                socketio.emit("frame", {"img": img_b64})
                socketio.emit("prediction", {
                    "word":     pred_word if hand_visible else "",
                    "conf":     pred_conf if hand_visible else 0.0,
                    "added":    added,
                    "sentence": list(state["sentence"]),
                    "hand":     hand_visible,
                })
        finally:
            cap.release()
            holistic.close()

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

@socketio.on("stop_camera")
def handle_stop():
    state["running"] = False

@socketio.on("disconnect")
def handle_disconnect():
    state["running"] = False

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SignBridge Web Arayuzu — V2 DENEY")
    print("="*50)
    print(f"Model: {CKPT.name} (input={INPUT_SIZE})")
    print(f"Kelimeler ({num_classes}): {', '.join(label_map.keys())}")
    print(f"Adres: http://localhost:5052")
    print("ORIJINAL: http://localhost:5050 (web_app.py) dokunulmadi")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5052, debug=False)
