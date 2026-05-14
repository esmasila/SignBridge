"""
SignBridge - YUZSUZ DENEY (Port 5051)
Yuz landmarksiz model: 1629 -> 225 ozellik (sadece pose + eller)
Ana sistemi bozmadan bu dosyayla test edebilirsin.

Once egit: python experiment_v2/train_noface.py
Sonra baslat: python experiment_v2/web_app_noface.py
Tarayici: http://localhost:5051
"""

from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import base64
import time
import json
import threading
from pathlib import Path
from collections import deque
from flask import Flask, render_template_string, Response, request, jsonify, redirect
from flask_socketio import SocketIO, emit

BASE    = Path(__file__).parent
CKPT    = BASE / "checkpoints" / "best_model_noface.pt"
PORT    = 5051

SEQ_LEN          = 30
PROB_AVG_WINDOW  = 2
MARGIN_MIN       = 0.12
TTA_K            = 1

# Yuz: 468*3=1404 → atilir. Kalan: pose(99) + sol el(63) + sag el(63) = 225
FACE_DIM   = 468 * 3   # 1404
INPUT_SIZE = 225

if not CKPT.exists():
    print(f"\n[HATA] {CKPT} bulunamadi!")
    print("Once egit: python experiment_v2/train_noface.py\n")
    import sys; sys.exit(1)


# ---- Model (yuzsuz: 225 giris) ----
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


ckpt         = torch.load(CKPT, map_location="cpu")
label_map    = ckpt["label_map"]
num_classes  = ckpt["num_classes"]
idx_to_label = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GRUModel(classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
val_acc = ckpt.get("val_acc", 0)
print(f"[YUZSUZ] Model yuklendi | {num_classes} kelime | val_acc={val_acc:.2f}% | {device}")


# ---- MediaPipe Hands Fallback ----
mp_hands       = mp.solutions.hands
hands_fallback = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.4, min_tracking_confidence=0.4
)

def hand_recovery(results, rgb_frame):
    has_left  = results.left_hand_landmarks is not None
    has_right = results.right_hand_landmarks is not None
    if has_left and has_right:
        return None
    hand_res = hands_fallback.process(rgb_frame)
    if not hand_res.multi_hand_landmarks:
        return None
    recovered = {"left": None, "right": None}
    for lm, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
        label = hd.classification[0].label
        if label == "Left":
            if not has_right and recovered["right"] is None:
                recovered["right"] = lm.landmark
        else:
            if not has_left and recovered["left"] is None:
                recovered["left"] = lm.landmark
    return recovered


# ---- TTA ----
def predict_with_tta(seq_np):
    probs = []
    with torch.no_grad():
        inp = torch.tensor(seq_np[np.newaxis]).to(device)
        out = model(inp)
        probs.append(torch.softmax(out, dim=1)[0].cpu().numpy())
        for _ in range(TTA_K):
            aug = seq_np.copy()
            aug += np.random.normal(0, 0.004, aug.shape).astype(np.float32)
            scale = np.random.uniform(0.95, 1.05)
            aug = np.clip(aug * scale, 0, 1)
            inp_aug = torch.tensor(aug[np.newaxis]).to(device)
            out_aug = model(inp_aug)
            probs.append(torch.softmax(out_aug, dim=1)[0].cpu().numpy())
    return np.mean(probs, axis=0)


# ---- MediaPipe ----
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_landmarks(results, recovered_hands=None):
    """Yuz ATILDI. Sadece pose + eller = 225 ozellik."""
    lm = []
    # Pose (33*3=99)
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 33 * 3)
    # Sol el (21*3=63)
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    elif recovered_hands and recovered_hands.get("left") is not None:
        for p in recovered_hands["left"]: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    # Sag el (21*3=63)
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    elif recovered_hands and recovered_hands.get("right") is not None:
        for p in recovered_hands["right"]: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    return np.array(lm, dtype=np.float32)  # (225,)


# ---- Uygulama durumu ----
state = {
    "buffer":        deque(maxlen=SEQ_LEN),
    "prob_history":  deque(maxlen=PROB_AVG_WINDOW),
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

CONF_THRESHOLD = 0.85
SMOOTH_NEEDED  = 6
COOLDOWN_SEC   = 0.6
HAND_GONE_SEC  = 2.0

# ---- Flask ----
app = Flask(__name__)
app.config["SECRET_KEY"] = "signbridge_noface"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignBridge DENEY - Yüzsüz Model</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
  header {
    background: linear-gradient(135deg, #1a2e1a, #163e16);
    padding: 16px 32px; display: flex; align-items: center; gap: 16px;
    border-bottom: 2px solid #2a4a2a;
  }
  header h1 { font-size: 1.5rem; color: #78fa87; }
  .badge {
    background: #2d4a1e; border: 1px solid #4a7a2a; border-radius: 20px;
    padding: 4px 14px; font-size: 0.8rem; color: #a3e635; font-weight: 700;
  }
  .pill { background: #1e293b; border: 1px solid #334155; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; color: #94a3b8; }
  .main { display: grid; grid-template-columns: 1fr 380px; gap: 20px;
    padding: 20px; max-width: 1300px; margin: 0 auto; }
  .camera-section { display: flex; flex-direction: column; gap: 16px; }
  .camera-wrapper { position: relative; background: #111; border-radius: 16px;
    overflow: hidden; border: 2px solid #2d4a1e; aspect-ratio: 4/3; }
  #cameraFeed { width: 100%; height: 100%; object-fit: cover; display: block; }
  .camera-placeholder { position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px; color: #4b5563; }
  .pred-overlay { position: absolute; top: 0; left: 0; right: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0.75) 0%, transparent 100%);
    padding: 16px 20px; display: flex; align-items: center; justify-content: space-between; }
  #predWord { font-size: 2rem; font-weight: 700; color: #78fa87; letter-spacing: 2px; }
  #predConf { font-size: 1.1rem; color: #6ee7b7; font-weight: 600; }
  .conf-bar-wrap { position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: rgba(255,255,255,0.1); }
  #confBar { height: 100%; background: linear-gradient(90deg, #16a34a, #06b6d4); transition: width 0.15s ease; width: 0%; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; }
  button { padding: 10px 20px; border: none; border-radius: 10px; font-size: 0.9rem;
    font-weight: 600; cursor: pointer; transition: all 0.2s; }
  #btnStart { background: linear-gradient(135deg, #16a34a, #15803d); color: white; flex: 1; }
  #btnStart:hover { opacity: 0.85; transform: translateY(-1px); }
  #btnStart.active { background: linear-gradient(135deg, #dc2626, #b91c1c); }
  #btnClear { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }
  .side-panel { display: flex; flex-direction: column; gap: 16px; }
  .card { background: #1a1a2e; border: 1px solid #1e293b; border-radius: 16px; padding: 20px; }
  .card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;
    color: #6b7280; margin-bottom: 14px; }
  #sentenceBox { min-height: 90px; background: #0f172a; border-radius: 10px; padding: 14px;
    font-size: 1.15rem; line-height: 1.6; color: #e2e8f0; border: 1px solid #1e293b; word-break: break-word; }
  #sentenceBox.empty { color: #4b5563; font-style: italic; }
  #wordList { display: flex; flex-wrap: wrap; gap: 6px; }
  .word-chip { background: #0f172a; border: 1px solid #1e293b; border-radius: 20px;
    padding: 4px 12px; font-size: 0.78rem; color: #94a3b8; transition: all 0.3s; }
  .word-chip.active { background: #16a34a; border-color: #16a34a; color: white; transform: scale(1.05); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4b5563;
    display: inline-block; margin-right: 6px; }
  .status-dot.active { background: #10b981; box-shadow: 0 0 8px #10b981; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  #cooldownBar { height: 3px; background: #06b6d4; border-radius: 2px; margin-top: 8px;
    width: 0%; transition: width 0.1s linear; }
</style>
</head>
<body>
<header>
  <h1>SignBridge</h1>
  <div class="badge">🧪 DENEY — Yüzsüz Model</div>
  <span style="color:#6b7280;font-size:0.85rem;">Port 5051 | 225 özellik (pose+eller)</span>
  <div class="pill" id="modelInfo">Yükleniyor...</div>
  <div style="margin-left:auto;display:flex;align-items:center;font-size:0.85rem;color:#6b7280;">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">Hazır</span>
  </div>
</header>

<div class="main">
  <div class="camera-section">
    <div class="camera-wrapper">
      <div class="camera-placeholder" id="placeholder">
        <p style="color:#a3e635;font-size:1.1rem;">🧪 Yüzsüz Deney Modu</p>
        <p>Kamerayı başlatmak için butona bas</p>
      </div>
      <img id="cameraFeed" style="display:none;" alt="Kamera">
      <div class="pred-overlay" style="display:none;" id="predOverlay">
        <span id="predWord">—</span>
        <span id="predConf">%0</span>
      </div>
      <div class="conf-bar-wrap"><div id="confBar"></div></div>
    </div>
    <div class="controls">
      <button id="btnStart" onclick="toggleCamera()">Kamerayı Başlat</button>
      <button id="btnAdd" onclick="manualAdd()" style="background:#16a34a;color:#fff;font-size:1.1rem;padding:12px 28px;">
        ✋ Kelime Ekle (Space)
      </button>
      <button id="btnClear" onclick="clearSentence()">Temizle</button>
      <label style="display:flex;align-items:center;gap:6px;color:#94a3b8;font-size:0.8rem;cursor:pointer;">
        <input type="checkbox" id="autoMode" checked> Otomatik Ekle
      </label>
    </div>
  </div>

  <div class="side-panel">
    <div class="card" style="border:2px solid #4a7a2a;">
      <div class="card-title" style="color:#a3e635;">🧪 DENEY BİLGİSİ</div>
      <div style="font-size:0.8rem;color:#94a3b8;line-height:1.7;">
        <div>Giriş boyutu: <span style="color:#a3e635;">225 özellik</span></div>
        <div>Yüz landmark: <span style="color:#ef4444;">ATILDI (1404 özellik)</span></div>
        <div>Pose: <span style="color:#a3e635;">✓ (99)</span> | Eller: <span style="color:#a3e635;">✓ (126)</span></div>
        <div id="valAccInfo" style="margin-top:4px;"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">İşaret Kelimeleri</div>
      <div id="sentenceBox" class="empty">Henüz kelime algılanmadı...</div>
      <div id="cooldownBar"></div>
    </div>
    <div class="card">
      <div class="card-title">DEBUG</div>
      <div id="debugPanel" style="font-family:monospace;font-size:0.8rem;color:#94a3b8;line-height:1.7;">
        <div>El: <span id="dbgHand" style="color:#6b7280;">-</span></div>
        <div>Buffer: <span id="dbgBuffer">0/30</span></div>
        <div>Margin: <span id="dbgMargin">-</span> <span id="dbgMarginOK"></span></div>
        <div style="margin-top:6px;color:#f59e0b;">TOP 5:</div>
        <div id="dbgTop5" style="padding-left:8px;"></div>
      </div>
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
let lastPredWord = '';
let lastPredConf = 0;

function manualAdd() {
  if (lastPredWord && lastPredConf >= 0.70) {
    if (sentence.length > 0 && sentence[sentence.length-1] === lastPredWord) return;
    sentence.push(lastPredWord);
    updateSentenceBox();
    socket.emit("force_add");
  }
}

socket.emit("get_info");
socket.on("info", (data) => {
  document.getElementById("modelInfo").textContent = data.num_classes + " kelime";
  document.getElementById("wordCount").textContent = data.num_classes;
  document.getElementById("valAccInfo").innerHTML = "Val acc: <span style='color:#a3e635;font-weight:700'>" + data.val_acc.toFixed(1) + "%</span>";
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
  lastPredWord = data.word;
  lastPredConf = data.conf;
  document.getElementById("predWord").textContent = data.word || "—";
  document.getElementById("predConf").textContent = "%" + Math.round(data.conf * 100);
  document.getElementById("confBar").style.width = (data.conf * 100) + "%";
  document.querySelectorAll(".word-chip").forEach(c => c.classList.remove("active"));
  const chip = document.getElementById("chip_" + data.word);
  if (chip && data.conf >= 0.80) chip.classList.add("active");
  sentence = data.sentence;
  updateSentenceBox();
  // Debug
  document.getElementById("dbgHand").textContent = data.hand ? "✓ GORUNUR" : "✗ YOK";
  document.getElementById("dbgHand").style.color = data.hand ? "#22c55e" : "#ef4444";
  document.getElementById("dbgBuffer").textContent = (data.buffer_size||0) + "/" + (data.buffer_max||30);
  const marg = data.margin || 0;
  document.getElementById("dbgMargin").textContent = marg.toFixed(3);
  document.getElementById("dbgMarginOK").textContent = data.margin_ok ? "✓" : "✗ KARISIK!";
  document.getElementById("dbgMarginOK").style.color = data.margin_ok ? "#22c55e" : "#ef4444";
  const top5 = data.top5 || [];
  document.getElementById("dbgTop5").innerHTML = top5.map((p, i) => {
    const pct = (p.conf * 100).toFixed(1);
    const color = i === 0 ? "#a3e635" : (i === 1 ? "#f59e0b" : "#6b7280");
    return `<div style="color:${color}">${i+1}. ${p.word}: ${pct}%</div>`;
  }).join("") || "<div style='color:#6b7280'>-</div>";
  if (data.added) {
    const bar = document.getElementById("cooldownBar");
    bar.style.width = "100%";
    setTimeout(() => { bar.style.width = "0%"; }, 1200);
  }
  if (data.done && sentence.length > 0) {
    sentence = [];
    updateSentenceBox();
  }
});

function updateSentenceBox() {
  const box = document.getElementById("sentenceBox");
  if (sentence.length === 0) {
    box.textContent = "Henüz kelime algılanmadı...";
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
    document.getElementById("btnStart").textContent = "Kamerayı Durdur";
    document.getElementById("btnStart").classList.add("active");
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("cameraFeed").style.display = "block";
    document.getElementById("predOverlay").style.display = "flex";
    document.getElementById("statusDot").classList.add("active");
    document.getElementById("statusText").textContent = "Canlı";
  } else {
    socket.emit("stop_camera");
    isRunning = false;
    document.getElementById("btnStart").textContent = "Kamerayı Başlat";
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

socket.on("frame", (data) => {
  document.getElementById("cameraFeed").src = "data:image/jpeg;base64," + data.img;
});

document.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") clearSentence();
  if (e.key === " ") { e.preventDefault(); manualAdd(); }
});
</script>
</body>
</html>"""

# ---- SocketIO ----
@socketio.on("connect")
def handle_connect():
    state["running"] = False
    if state["cap"] is not None:
        try: state["cap"].release()
        except: pass
        state["cap"] = None
    state["sentence"] = []
    state["buffer"] = deque(maxlen=SEQ_LEN)
    state["prob_history"] = deque(maxlen=PROB_AVG_WINDOW)
    state["smooth_counts"] = {k: 0 for k in label_map}
    state["current_word"] = ""
    state["current_conf"] = 0.0
    state["last_add_time"] = 0.0

@socketio.on("get_info")
def handle_info():
    emit("info", {
        "num_classes": num_classes,
        "words": list(label_map.keys()),
        "val_acc": ckpt.get("val_acc", 0.0),
    })

@socketio.on("clear_sentence")
def handle_clear():
    state["sentence"] = []
    state["smooth_counts"] = {k: 0 for k in label_map}

@socketio.on("force_add")
def handle_force_add():
    word = state["current_word"]
    conf = state["current_conf"]
    if word and conf >= CONF_THRESHOLD:
        state["sentence"].append(word)
        state["last_add_time"] = time.time()
        state["smooth_counts"] = {k: 0 for k in label_map}

@socketio.on("start_camera")
def handle_start():
    if state["running"]:
        return
    state["running"] = True
    state["buffer"]  = deque(maxlen=SEQ_LEN)
    state["prob_history"] = deque(maxlen=PROB_AVG_WINDOW)
    state["smooth_counts"] = {k: 0 for k in label_map}

    cap = cv2.VideoCapture(0)
    state["cap"] = cap
    holistic = mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    state["holistic"] = holistic

    def camera_loop():
        mp_draw = mp.solutions.drawing_utils
        frame_counter = 0
        try:
            while state["running"]:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_counter += 1
                frame = cv2.flip(frame, 1)

                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Hand recovery
                recovered = None
                if (results.left_hand_landmarks is None or
                    results.right_hand_landmarks is None):
                    recovered = hand_recovery(results, rgb)

                # Landmark ciz (yuz cizme — sadece eller ve pose)
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS)

                if recovered:
                    for side in ("left", "right"):
                        lms = recovered.get(side)
                        if lms is None: continue
                        hc, wc = frame.shape[:2]
                        pts = [(int(p.x*wc), int(p.y*hc)) for p in lms]
                        for (px, py) in pts:
                            cv2.circle(frame, (px, py), 4, (0, 140, 255), -1)

                # "YUZSUZ" etiketi
                cv2.putText(frame, "YUZSUZ", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2)

                lm = extract_landmarks(results, recovered_hands=recovered)
                state["buffer"].append(lm)

                rec_left  = recovered and recovered.get("left")  is not None
                rec_right = recovered and recovered.get("right") is not None
                hand_visible = (results.left_hand_landmarks is not None or
                                results.right_hand_landmarks is not None or
                                rec_left or rec_right)

                now_t = time.time()
                sentence_done = False

                if hand_visible:
                    state["last_hand_time"] = now_t
                else:
                    if (state["sentence"] and state["last_hand_time"] > 0 and
                        (now_t - state["last_hand_time"]) >= HAND_GONE_SEC):
                        sentence_done = True
                        state["last_hand_time"] = 0.0

                pred_word = state["current_word"]
                pred_conf = 0.0
                added     = False
                pred_margin = 0.0
                top5_debug  = []

                if len(state["buffer"]) == SEQ_LEN:
                    seq = np.clip(np.array(state["buffer"], dtype=np.float32), 0, 1)
                    prob = predict_with_tta(seq)
                    state["prob_history"].append(prob)
                    avg_prob = np.mean(np.stack(state["prob_history"]), axis=0)

                    top5_idx = np.argsort(avg_prob)[-5:][::-1]
                    top5_debug = [
                        {"word": idx_to_label[int(i)], "conf": float(avg_prob[i])}
                        for i in top5_idx
                    ]
                    top1_conf = top5_debug[0]["conf"]
                    top2_conf = top5_debug[1]["conf"]
                    pred_margin = top1_conf - top2_conf
                    pred_conf   = top1_conf
                    pred_word   = top5_debug[0]["word"]
                    state["current_word"] = pred_word
                    state["current_conf"] = pred_conf
                    state["_top5_debug"]  = top5_debug
                    state["_margin_debug"]= pred_margin

                    now = time.time()
                    in_cooldown   = (now - state["last_add_time"]) < COOLDOWN_SEC
                    is_confident  = (pred_conf >= CONF_THRESHOLD and pred_margin >= MARGIN_MIN)

                    if is_confident and hand_visible and not in_cooldown:
                        state["smooth_counts"][pred_word] += 1
                        for k in state["smooth_counts"]:
                            if k != pred_word:
                                state["smooth_counts"][k] = max(0, state["smooth_counts"][k] - 1)
                        if state["smooth_counts"][pred_word] >= SMOOTH_NEEDED:
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
                    "done":     sentence_done,
                    "buffer_size": len(state["buffer"]),
                    "buffer_max":  SEQ_LEN,
                    "top5":     top5_debug,
                    "margin":   pred_margin,
                    "margin_ok": pred_margin >= MARGIN_MIN,
                    "conf_ok":   pred_conf >= CONF_THRESHOLD,
                })
                if sentence_done:
                    state["sentence"] = []
                    state["smooth_counts"] = {k: 0 for k in label_map}
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

@app.route("/favicon.ico")
def serve_favicon():
    return "", 204

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SignBridge DENEY - Yuzsuz Model (Port 5051)")
    print("="*50)
    print(f"Kelimeler ({num_classes}): {', '.join(list(label_map.keys())[:10])}...")
    print(f"Adres: http://localhost:{PORT}")
    print(f"Val Acc: {val_acc:.2f}%")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False, allow_unsafe_werkzeug=True)
