"""
experiment_v2 Web Arayuzu + YOLO PERSON CROP
YOLO ile kisiyi tespit edip, sadece o bolgeyi MediaPipe'a verir.
Daha temiz landmark -> daha dogru tahmin.

Baslat: python experiment_v2/web_app_yolo.py
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
from pathlib import Path
from collections import deque
from flask import Flask, render_template_string, Response, request, jsonify, redirect
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

BASE    = Path(__file__).parent
CKPT    = BASE / "checkpoints" / "best_model.pt"
SEQ_LEN = 30  # Model bu uzunlukla egitildi
HTTP_SEQ_LEN = 15  # Shorter buffer for HTTP/mobile (faster predictions)
PREDICT_EVERY_N_FRAMES = 1  # Her frame tahmin (stale state olmasin)
PROB_AVG_WINDOW = 3  # Son 3 tahminin olasilik ortalamasi (gurultu azaltir)
MARGIN_MIN = 0.05  # Cok gevsek - sadece asiri karisik durumu reddet

# ---- Model ----
class GRUModel(nn.Module):
    def __init__(self, input_size=1629, hidden=256, layers=2, classes=3, dropout=0.3):
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
print(f"Model yuklendi | {num_classes} kelime | {device}")

# ---- YOLO: Kisi Tespit Modeli ----
YOLO_PATH = BASE.parent / "yolo11n.pt"
yolo_model = YOLO(str(YOLO_PATH))
print(f"YOLO yuklendi: {YOLO_PATH}")

# YOLO ile kisi tespit + kirpma (COCO class 0 = person)
def yolo_person_crop(frame, pad=40):
    """
    Frame icinde kisiyi tespit eder, bounding box'a pad ekleyip kirpar.
    Kisi bulunamazsa orijinal frame'i dondurur.
    Returns: (cropped_frame, (x1, y1, x2, y2) or None)
    """
    try:
        results = yolo_model.predict(frame, classes=[0], conf=0.4, verbose=False, imgsz=416)
        if not results or len(results[0].boxes) == 0:
            return frame, None
        # En buyuk kisi kutusunu sec (konusan kisi)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[i].astype(int)
        h, w = frame.shape[:2]
        # Pad ekle
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        cropped = frame[y1:y2, x1:x2].copy()
        return cropped, (x1, y1, x2, y2)
    except Exception as e:
        print(f"YOLO hata: {e}")
        return frame, None

# ---- MediaPipe ----
mp_holistic = mp.solutions.holistic
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# Hand fallback: Holistic el bulamadiginda daha hassas Hands modulu devreye girer
hands_fallback = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,   # Holistic'ten daha dusuk - daha duyarli
    min_tracking_confidence=0.4
)

def hand_recovery(results, rgb_frame):
    """
    Holistic el bulamadiysa MediaPipe Hands ile bul.
    Eksik olan eli landmark'larla doldur.
    Returns: (left_lm_list, right_lm_list) - her biri None veya 21'lik liste
    """
    has_left = results.left_hand_landmarks is not None
    has_right = results.right_hand_landmarks is not None
    if has_left and has_right:
        return None  # ikisi de var, recovery gerekmez

    hand_results = hands_fallback.process(rgb_frame)
    if not hand_results.multi_hand_landmarks:
        return None

    recovered = {"left": None, "right": None}
    for lm, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
        # MediaPipe'ta tersine: ayna modunda "Left" etiketi bizim sag elimiz
        label = handedness.classification[0].label  # 'Left' veya 'Right'
        # Flip yapildigi icin tersine cevir
        if label == "Left":
            if not has_right and recovered["right"] is None:
                recovered["right"] = lm.landmark
        else:
            if not has_left and recovered["left"] is None:
                recovered["left"] = lm.landmark
    return recovered

def extract_landmarks(results, recovered_hands=None):
    """
    recovered_hands: {'left': landmark_list or None, 'right': landmark_list or None}
    Holistic'in kaciran ellerini Hands fallback'ten tamamlar.
    """
    lm = []
    if results.face_landmarks:
        for p in results.face_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 468 * 3)
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 33 * 3)
    # Sol el
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    elif recovered_hands and recovered_hands.get("left") is not None:
        for p in recovered_hands["left"]: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    # Sag el
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    elif recovered_hands and recovered_hands.get("right") is not None:
        for p in recovered_hands["right"]: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    return np.array(lm, dtype=np.float32)

# ---- Uygulama durumu ----
state = {
    "buffer":        deque(maxlen=SEQ_LEN),
    "prob_history":  deque(maxlen=PROB_AVG_WINDOW),  # Son N prob dist (gurultu azaltma)
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

CONF_THRESHOLD = 0.90   # ORIJINAL deger - yuksek guven (yanlis ekleme onler)
SMOOTH_NEEDED  = 10     # ORIJINAL deger - 10 frame onay gerekli
COOLDOWN_SEC   = 1.2    # Kelimeler arasi bekleme (orijinal 1.5'ten az hizli)
HAND_GONE_SEC  = 2.0    # el cekilince cumle bitis suresi

# ---- Flask ----
app = Flask(__name__)
app.config["SECRET_KEY"] = "signbridge_v2"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignBridge YOLO - İşaret Dili Çevirisi</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid #2a2a4a;
  }
  header h1 { font-size: 1.5rem; color: #a78bfa; }
  header span { font-size: 0.85rem; color: #6b7280; }
  .pill {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #94a3b8;
  }
  .main {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 20px;
    padding: 20px;
    max-width: 1300px;
    margin: 0 auto;
  }
  /* Sol: kamera */
  .camera-section { display: flex; flex-direction: column; gap: 16px; }
  .camera-wrapper {
    position: relative;
    background: #111;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #1e293b;
    aspect-ratio: 4/3;
  }
  #cameraFeed {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .camera-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: #4b5563;
  }
  .camera-placeholder svg { width: 64px; opacity: 0.3; }

  /* Anlik tahmin overlay */
  .pred-overlay {
    position: absolute;
    top: 0; left: 0; right: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0.75) 0%, transparent 100%);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  #predWord {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 2px;
  }
  #predConf {
    font-size: 1.1rem;
    color: #6ee7b7;
    font-weight: 600;
  }
  .conf-bar-wrap {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: rgba(255,255,255,0.1);
  }
  #confBar {
    height: 100%;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    transition: width 0.15s ease;
    width: 0%;
  }

  /* Kontrol butonlari */
  .controls {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  button {
    padding: 10px 20px;
    border: none;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  #btnStart {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    flex: 1;
  }
  #btnStart:hover { opacity: 0.85; transform: translateY(-1px); }
  #btnStart.active {
    background: linear-gradient(135deg, #dc2626, #b91c1c);
  }
  #btnClear {
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
  }
  #btnClear:hover { background: #334155; color: #e2e8f0; }

  /* Sag: panel */
  .side-panel { display: flex; flex-direction: column; gap: 16px; }

  .card {
    background: #1a1a2e;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 20px;
  }
  .card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7280;
    margin-bottom: 14px;
  }

  /* Cumle kutusu */
  #sentenceBox {
    min-height: 90px;
    background: #0f172a;
    border-radius: 10px;
    padding: 14px;
    font-size: 1.15rem;
    line-height: 1.6;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    word-break: break-word;
  }
  #sentenceBox.empty { color: #4b5563; font-style: italic; }

  /* Gecmis cumle */
  #historyList {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 200px;
    overflow-y: auto;
  }
  .history-item {
    background: #0f172a;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.85rem;
    color: #94a3b8;
    border-left: 3px solid #7c3aed;
  }

  /* Kelime listesi */
  #wordList {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .word-chip {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #94a3b8;
    transition: all 0.3s;
  }
  .word-chip.active {
    background: #7c3aed;
    border-color: #7c3aed;
    color: white;
    transform: scale(1.05);
  }

  /* Durum indikatoru */
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4b5563;
    display: inline-block;
    margin-right: 6px;
  }
  .status-dot.active {
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Cooldown bar */
  #cooldownBar {
    height: 3px;
    background: #06b6d4;
    border-radius: 2px;
    margin-top: 8px;
    width: 0%;
    transition: width 0.1s linear;
  }
</style>
</head>
<body>

<header>
  <h1>SignBridge <span style="color:#22c55e;font-size:0.7rem;padding:3px 10px;border:1px solid #22c55e;border-radius:10px;vertical-align:middle;">YOLO</span></h1>
  <span>Türk İşaret Dili Çevirisi + Kişi Tespit</span>
  <div class="pill" id="modelInfo">Yükleniyor...</div>
  <div style="margin-left:auto; display:flex; align-items:center; font-size:0.85rem; color:#6b7280;">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">Hazır</span>
  </div>
</header>

<div class="main">
  <!-- Sol: Kamera -->
  <div class="camera-section">
    <div class="camera-wrapper">
      <div class="camera-placeholder" id="placeholder">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 9.75v9A2.25 2.25 0 004.5 18.75z"/>
        </svg>
        <p>Kamerayı başlatmak için aşağıdaki butona bas</p>
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
      <button id="btnAdd" onclick="manualAdd()" style="background:#22c55e;color:#fff;font-size:1.1rem;padding:12px 28px;">
        ✋ Kelime Ekle (Space)
      </button>
      <button id="btnClear" onclick="clearSentence()">Temizle</button>
      <button onclick="saveSentence()" style="background:#1e293b;color:#94a3b8;border:1px solid #334155;">
        Kaydet
      </button>
      <label style="display:flex;align-items:center;gap:6px;color:#94a3b8;font-size:0.8rem;cursor:pointer;">
        <input type="checkbox" id="autoMode" checked> Otomatik Ekle
      </label>
    </div>
  </div>

  <!-- Sag: Panel -->
  <div class="side-panel">

    <!-- Cumle -->
    <div class="card">
      <div class="card-title">İşaret Kelimeleri</div>
      <div id="sentenceBox" class="empty">Henüz kelime algılanmadı...</div>
      <div id="cooldownBar"></div>
    </div>
    <div class="card">
      <div class="card-title">Türkçe Cümle</div>
      <div id="nlpBox" style="min-height:38px;font-size:1.1rem;font-weight:600;color:#a78bfa;padding:4px 0;">—</div>
    </div>

    <!-- Gecmis -->
    <div class="card">
      <div class="card-title">Geçmiş Cümleler</div>
      <div id="historyList">
        <div style="color:#4b5563; font-size:0.85rem;">Henüz cümle yok</div>
      </div>
    </div>

    <!-- DEBUG PANEL -->
    <div class="card" style="border:1px solid #f59e0b;">
      <div class="card-title" style="color:#f59e0b;">🔍 DEBUG PANEL</div>
      <div id="debugPanel" style="font-family:monospace;font-size:0.8rem;color:#94a3b8;line-height:1.7;">
        <div>El: <span id="dbgHand" style="color:#6b7280;">-</span></div>
        <div>Buffer: <span id="dbgBuffer" style="color:#6b7280;">0/30</span></div>
        <div>Margin: <span id="dbgMargin" style="color:#6b7280;">-</span> <span id="dbgMarginOK"></span></div>
        <div>Conf ≥ 0.75: <span id="dbgConfOK"></span></div>
        <div style="margin-top:6px;color:#f59e0b;">TOP 5 TAHMIN:</div>
        <div id="dbgTop5" style="padding-left:8px;"></div>
        <div style="margin-top:6px;color:#f59e0b;">SMOOTH (/4):</div>
        <div id="dbgSmooth" style="padding-left:8px;"></div>
      </div>
    </div>

    <!-- Kelime listesi -->
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
let history = [];
let lastPredWord = '';
let lastPredConf = 0;

function manualAdd() {
  if (lastPredWord && lastPredConf >= 0.70) {
    // Ayni kelimeyi art arda ekleme
    if (sentence.length > 0 && sentence[sentence.length-1] === lastPredWord) return;
    sentence.push(lastPredWord);
    updateSentenceBox();
    socket.emit("force_add");
    // Flash efekti
    const btn = document.getElementById("btnAdd");
    btn.style.background = "#a78bfa";
    setTimeout(() => btn.style.background = "#22c55e", 300);
  }
}

// Model bilgisi al
socket.emit("get_info");
socket.on("info", (data) => {
  document.getElementById("modelInfo").textContent = data.num_classes + " kelime";
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

// Tahmin guncelleme
socket.on("prediction", (data) => {
  const word = data.word;
  const conf = data.conf;
  const added = data.added;
  const sentenceData = data.sentence;

  lastPredWord = word;
  lastPredConf = conf;

  // Anlik tahmin goster
  document.getElementById("predWord").textContent = word || "—";
  document.getElementById("predConf").textContent = "%" + Math.round(conf * 100);
  document.getElementById("confBar").style.width = (conf * 100) + "%";

  // Aktif kelime chipini vurgula
  document.querySelectorAll(".word-chip").forEach(c => c.classList.remove("active"));
  const chip = document.getElementById("chip_" + word);
  if (chip && conf >= 0.80) chip.classList.add("active");

  // Cumle guncelle
  sentence = sentenceData;
  updateSentenceBox();

  // ═══ DEBUG PANEL ═══
  document.getElementById("dbgHand").textContent = data.hand ? "✓ GORUNUR" : "✗ YOK";
  document.getElementById("dbgHand").style.color = data.hand ? "#22c55e" : "#ef4444";
  document.getElementById("dbgBuffer").textContent = (data.buffer_size||0) + "/" + (data.buffer_max||30);
  document.getElementById("dbgBuffer").style.color = data.buffer_size >= data.buffer_max ? "#22c55e" : "#f59e0b";
  const marg = (data.margin||0);
  document.getElementById("dbgMargin").textContent = marg.toFixed(3);
  document.getElementById("dbgMargin").style.color = data.margin_ok ? "#22c55e" : "#ef4444";
  document.getElementById("dbgMarginOK").textContent = data.margin_ok ? "✓" : "✗ KARISIK!";
  document.getElementById("dbgMarginOK").style.color = data.margin_ok ? "#22c55e" : "#ef4444";
  document.getElementById("dbgConfOK").textContent = data.conf_ok ? "✓" : "✗";
  document.getElementById("dbgConfOK").style.color = data.conf_ok ? "#22c55e" : "#ef4444";

  const top5 = data.top5 || [];
  document.getElementById("dbgTop5").innerHTML = top5.map((p, i) => {
    const pct = (p.conf * 100).toFixed(1);
    const color = i === 0 ? "#a78bfa" : (i === 1 ? "#f59e0b" : "#6b7280");
    return `<div style="color:${color}">${i+1}. ${p.word}: ${pct}%</div>`;
  }).join("") || "<div style='color:#6b7280'>-</div>";

  const smTop = data.smooth_top || [];
  document.getElementById("dbgSmooth").innerHTML = smTop.map(s => {
    const color = s.c >= (data.smooth_needed||4) ? "#22c55e" : "#94a3b8";
    return `<div style="color:${color}">${s.w}: ${s.c}</div>`;
  }).join("") || "<div style='color:#6b7280'>-</div>";

  // Kelime eklendiyse cooldown animasyonu
  if (added) {
    const bar = document.getElementById("cooldownBar");
    bar.style.width = "100%";
    setTimeout(() => { bar.style.width = "0%"; }, 1200);
  }

  // Eller cekildi, cumle bitti
  if (data.done && sentence.length > 0) {
    const gloss = sentence.join(" ");
    const turk = glossToSentence(sentence);
    history.unshift({gloss, turk});
    if (history.length > 10) history.pop();

    const hl = document.getElementById("historyList");
    hl.innerHTML = history.map(h =>
      `<div class="history-item"><div style="color:#a78bfa;font-weight:600">${h.turk}</div><div style="font-size:0.75rem;color:#4b5563;margin-top:2px">${h.gloss}</div></div>`
    ).join("");

    sentence = [];
    updateSentenceBox();
  }
});

// ═══════════════════════════════════════════════
// NLP: Gloss kelimeleri → Düzgün Türkçe cümle
// ═══════════════════════════════════════════════
const GLOSS_TO_TURKISH = {
  'MERHABA':'merhaba', 'TESEKKUR':'teşekkür ederim', 'EVET':'evet', 'HAYIR':'hayır',
  'IYI':'iyi', 'KOTU':'kötü', 'BUYUK':'büyük', 'KUCUK':'küçük',
  'BEN':'ben', 'SEN':'sen', 'BABA':'baba', 'ANNE':'anne', 'ABLA':'abla',
  'ABI':'abi', 'AMCA':'amca', 'ARKADAS':'arkadaş', 'OGRETMEN':'öğretmen',
  'DOKTOR':'doktor', 'HEMSIRE':'hemşire',
  'OKUL':'okul', 'EV':'ev', 'TUVALET':'tuvalet', 'SU':'su',
  'CAY':'çay', 'KAHVE':'kahve', 'EKMEK':'ekmek', 'YEMEK':'yemek',
  'CIKOLATA':'çikolata', 'BAKLAVA':'baklava', 'AYRAN':'ayran', 'ERIK':'erik',
  'SABAH':'sabah', 'AKSAM':'akşam', 'BUGUN':'bugün', 'DUN':'dün', 'GUN':'gün',
  'KIS':'kış',
  'BIR':'bir', 'IKI':'iki', 'UC':'üç', 'DORT':'dört', 'BES':'beş',
  'ALTI':'altı', 'YEDI':'yedi', 'SEKIZ':'sekiz', 'DOKUZ':'dokuz', 'SIFIR':'sıfır',
  'OCAK':'ocak', 'SUBAT':'şubat', 'MART':'mart', 'NISAN':'nisan',
  'MAYIS':'mayıs', 'HAZIRAN':'haziran', 'TEMMUZ':'temmuz', 'AGUSTOS':'ağustos',
  'EYLUL':'eylül', 'EKIM':'ekim', 'KASIM':'kasım', 'ARALIK':'aralık',
  'PAZARTESI':'pazartesi', 'SALI':'salı', 'CARSAMBA':'çarşamba',
  'PERSEMBE':'perşembe', 'CUMA':'cuma', 'CUMARTESI':'cumartesi', 'PAZAR':'pazar',
  'ASK':'aşk', 'YARDIM':'yardım', 'DIKKAT':'dikkat', 'CEZA':'ceza',
  'FUTBOL':'futbol', 'VOLEYBOL':'voleybol', 'HENTBOL':'hentbol',
  'TV':'TV', 'CD':'CD', 'BILGISAYAR':'bilgisayar',
  'AKILLI':'akıllı', 'APTAL':'aptal', 'ZENGIN':'zengin',
  'ALERJI':'alerji', 'AYIP':'ayıp', 'ORUC':'oruç', 'FINAL':'final',
  'ALLAH':'Allah', 'AMIN':'amin', 'ELHAMDULILLAH':'elhamdülillah',
  'ATATURK':'Atatürk', 'ARAP':'Arap', 'BARIS':'barış',
  'EVLENMEK':'evlenmek', 'YATMAK':'yatmak', 'DOYMAK':'doymak',
  'HANGI':'hangi', 'CUNKU':'çünkü', 'YETER':'yeter',
  'HOSCA KAL':'hoşça kal', 'BOS VERMEK':'boş vermek', 'BU KADAR':'bu kadar',
  'CEP TELEFONU':'cep telefonu', 'LUTFEN':'lütfen',
};

// Cümle kalıpları — gloss sırasına göre Türkçe cümle üret
function glossToSentence(glossWords) {
  if (!glossWords || glossWords.length === 0) return '—';

  const words = glossWords.map(g => GLOSS_TO_TURKISH[g] || g.toLowerCase());

  // Tek kelime
  if (words.length === 1) {
    const w = words[0];
    // Selamlaşma
    if (['merhaba','hoşça kal'].includes(w)) return w.charAt(0).toUpperCase() + w.slice(1) + '!';
    if (w === 'evet') return 'Evet.';
    if (w === 'hayır') return 'Hayır.';
    if (w === 'teşekkür ederim') return 'Teşekkür ederim.';
    if (w === 'yeter') return 'Yeter!';
    if (w === 'dikkat') return 'Dikkat!';
    if (w === 'yardım') return 'Yardım!';
    if (w === 'lütfen') return 'Lütfen.';
    return w.charAt(0).toUpperCase() + w.slice(1) + '.';
  }

  // Kalıp eşleştirme
  const g = glossWords;
  const len = g.length;

  // İki kelimelik kalıplar
  if (len === 2) {
    const [a, b] = g;
    // Selamlama + kişi: MERHABA BABA → Merhaba baba!
    if (a === 'MERHABA') return 'Merhaba ' + (GLOSS_TO_TURKISH[b]||b.toLowerCase()) + '!';
    // Kişi + selamlama: BABA MERHABA → Baba, merhaba!
    if (b === 'MERHABA') return (GLOSS_TO_TURKISH[a]||a.toLowerCase()).charAt(0).toUpperCase() + (GLOSS_TO_TURKISH[a]||a.toLowerCase()).slice(1) + ', merhaba!';
    // Sıfat + isim: BUYUK EV → Büyük ev
    if (['IYI','KOTU','BUYUK','KUCUK','AKILLI','APTAL','ZENGIN'].includes(a))
      return (GLOSS_TO_TURKISH[a]||a) + ' ' + (GLOSS_TO_TURKISH[b]||b.toLowerCase()) + '.';
    // İsim + sıfat: EV BUYUK → Ev büyük.
    if (['IYI','KOTU','BUYUK','KUCUK','AKILLI','APTAL','ZENGIN'].includes(b))
      return (GLOSS_TO_TURKISH[a]||a).charAt(0).toUpperCase() + (GLOSS_TO_TURKISH[a]||a).slice(1) + ' ' + (GLOSS_TO_TURKISH[b]||b.toLowerCase()) + '.';
    // Kişi + fiil: BEN YEMEK → Ben yemek yiyorum / yiyeceğim
    // İsim + isim: BABA ANNE → Baba ve anne
    // Genel: a b
  }

  // Üç+ kelimelik — akıllı birleştirme
  let result = '';
  let i = 0;
  const parts = [];

  while (i < len) {
    const curr = g[i];
    const tw = GLOSS_TO_TURKISH[curr] || curr.toLowerCase();

    // Selamlama başta
    if (i === 0 && ['MERHABA','HOSCA KAL'].includes(curr)) {
      parts.push(tw.charAt(0).toUpperCase() + tw.slice(1));
      // Sonraki kelime varsa virgül
      if (i < len - 1) parts[parts.length-1] += ',';
      i++; continue;
    }

    // Teşekkür sonda
    if (curr === 'TESEKKUR') {
      parts.push('teşekkür ederim');
      i++; continue;
    }

    // Lütfen
    if (curr === 'LUTFEN') {
      parts.push('lütfen');
      i++; continue;
    }

    parts.push(tw);
    i++;
  }

  result = parts.join(' ');
  // İlk harf büyük, sonda nokta
  result = result.charAt(0).toUpperCase() + result.slice(1);
  if (!result.endsWith('!') && !result.endsWith('?') && !result.endsWith('.'))
    result += '.';

  return result;
}

function updateSentenceBox() {
  const box = document.getElementById("sentenceBox");
  const nlpBox = document.getElementById("nlpBox");
  if (sentence.length === 0) {
    box.textContent = "Henüz kelime algılanmadı...";
    box.classList.add("empty");
    nlpBox.textContent = "—";
  } else {
    box.textContent = sentence.join(" ");
    box.classList.remove("empty");
    nlpBox.textContent = glossToSentence(sentence);
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

function saveSentence() {
  if (sentence.length === 0) return;
  const gloss = sentence.join(" ");
  const turk = glossToSentence(sentence);
  history.unshift({gloss, turk});
  if (history.length > 10) history.pop();

  const hl = document.getElementById("historyList");
  hl.innerHTML = history.map(h =>
    `<div class="history-item"><div style="color:#a78bfa;font-weight:600">${h.turk}</div><div style="font-size:0.75rem;color:#4b5563;margin-top:2px">${h.gloss}</div></div>`
  ).join("");

  clearSentence();
}

// Video frame al
socket.on("frame", (data) => {
  document.getElementById("cameraFeed").src = "data:image/jpeg;base64," + data.img;
});

// Klavye kisayollari
document.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") clearSentence();
  if (e.key === "s" || e.key === "S") saveSentence();
  if (e.key === " ") { e.preventDefault(); manualAdd(); }
});
</script>
</body>
</html>
"""

# ---- SocketIO olaylar ----
@socketio.on("connect")
def handle_connect():
    # Sayfa yenilenince HER SEYI sifirla
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
    # EGITIM ile BIREBIR AYNI parametreler (pipeline.py:94-97)
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    state["holistic"] = holistic

    import threading
    def camera_loop():
        mp_draw = mp.solutions.drawing_utils
        frame_counter = 0  # Optimized: tahmin skip sayaci
        try:
            while state["running"]:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_counter += 1

                frame = cv2.flip(frame, 1)

                # === YOLO: Kisiyi tespit et, sadece o bolgeyi MediaPipe'a ver ===
                cropped, bbox = yolo_person_crop(frame, pad=50)
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # === HAND RECOVERY: Holistic el kacirdiysa Hands ile bul ===
                recovered = None
                if (results.left_hand_landmarks is None or
                    results.right_hand_landmarks is None):
                    recovered = hand_recovery(results, rgb)

                # Landmark ciz - bbox varsa kirpilmis frame'e ciz, yoksa full frame
                target_for_draw = cropped
                mp_draw.draw_landmarks(target_for_draw, results.left_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(target_for_draw, results.right_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(target_for_draw, results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS)

                # Recovery ellerini farkli renkte ciz (turuncu = kurtarilmis)
                if recovered:
                    for side in ("left", "right"):
                        lms = recovered.get(side)
                        if lms is None: continue
                        h_c, w_c = target_for_draw.shape[:2]
                        pts = [(int(p.x*w_c), int(p.y*h_c)) for p in lms]
                        for (x, y) in pts:
                            cv2.circle(target_for_draw, (x, y), 4, (0, 140, 255), -1)
                        # Bounding box ciz
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        cv2.rectangle(target_for_draw,
                                      (min(xs)-8, min(ys)-8), (max(xs)+8, max(ys)+8),
                                      (0, 140, 255), 2)
                        cv2.putText(target_for_draw, "HAND RECOVERY",
                                    (min(xs)-8, min(ys)-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

                # Kirpilmis bolgeyi orijinal frame'e geri yaz (landmarklar gozuksun)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    frame[y1:y2, x1:x2] = target_for_draw
                    # YOLO box'u ciz (yesil)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "YOLO: KISI", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                lm = extract_landmarks(results, recovered_hands=recovered)
                # NOT: EMA smoothing kaldirildi - egitimde yoktu, dagilim kaymasi yapiyordu
                state["buffer"].append(lm)

                # El var mi? Holistic + Recovery birlikte sayilir
                rec_left = recovered and recovered.get("left") is not None
                rec_right = recovered and recovered.get("right") is not None
                hand_visible = (results.left_hand_landmarks is not None or
                                results.right_hand_landmarks is not None or
                                rec_left or rec_right)

                now_t = time.time()
                sentence_done = False

                if hand_visible:
                    state["last_hand_time"] = now_t
                else:
                    # El cekileli HAND_GONE_SEC gecti ve cumle varsa → cumle bitti
                    if (state["sentence"] and state["last_hand_time"] > 0 and
                        (now_t - state["last_hand_time"]) >= HAND_GONE_SEC):
                        sentence_done = True
                        state["last_hand_time"] = 0.0  # tekrar tetikleme

                pred_word = state["current_word"]
                pred_conf = 0.0
                added     = False

                # Optimized: her PREDICT_EVERY_N_FRAMES frame'de tahmin (CPU yuku azalir)
                do_predict = (frame_counter % PREDICT_EVERY_N_FRAMES == 0)
                pred_margin = 0.0  # Top-1 ile Top-2 arasi fark
                top5_debug = []  # DEBUG: top-5 tahmin listesi
                if len(state["buffer"]) == SEQ_LEN and do_predict:
                    seq = np.clip(np.array(state["buffer"], dtype=np.float32), 0, 1)
                    inp = torch.tensor(seq).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out  = model(inp)
                        prob = torch.softmax(out, dim=1)[0].cpu().numpy()

                    # Olasilik gecmisine ekle ve ortalama al
                    state["prob_history"].append(prob)
                    avg_prob = np.mean(np.stack(state["prob_history"]), axis=0)

                    # Top-5 bul (DEBUG icin)
                    top5_idx = np.argsort(avg_prob)[-5:][::-1]
                    top5_debug = [
                        {"word": idx_to_label[int(i)], "conf": float(avg_prob[i])}
                        for i in top5_idx
                    ]

                    top1_conf = top5_debug[0]["conf"]
                    top2_conf = top5_debug[1]["conf"]
                    pred_margin = top1_conf - top2_conf

                    pred_conf = top1_conf
                    pred_word = top5_debug[0]["word"]

                    state["current_word"] = pred_word
                    state["current_conf"] = pred_conf
                    state["_top5_debug"] = top5_debug
                    state["_margin_debug"] = pred_margin
                elif len(state["buffer"]) == SEQ_LEN:
                    # Skip'lenen frame'de son tahmini kullan
                    pred_word = state["current_word"]
                    pred_conf = state["current_conf"]
                    pred_margin = MARGIN_MIN  # skip frame'de margin'i tamam say
                    top5_debug = state.get("_top5_debug", [])

                if len(state["buffer"]) == SEQ_LEN:

                    now = time.time()
                    in_cooldown = (now - state["last_add_time"]) < COOLDOWN_SEC

                    # Margin check: Top-1 ile Top-2 cok yakinsa karisik say, ekleme
                    is_confident = (pred_conf >= CONF_THRESHOLD and pred_margin >= MARGIN_MIN)

                    if is_confident and hand_visible and not in_cooldown:
                        state["smooth_counts"][pred_word] += 1
                        for k in state["smooth_counts"]:
                            if k != pred_word:
                                state["smooth_counts"][k] = max(0, state["smooth_counts"][k] - 1)

                        if state["smooth_counts"][pred_word] >= SMOOTH_NEEDED:
                            # Ayni kelimeyi art arda ekleme
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

                # Frame'i encode et
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_b64 = base64.b64encode(buf).decode("utf-8")

                socketio.emit("frame", {"img": img_b64})
                # DEBUG bilgisi
                top_smooth = sorted(state["smooth_counts"].items(), key=lambda x: -x[1])[:3]
                socketio.emit("prediction", {
                    "word":     pred_word if hand_visible else "",
                    "conf":     pred_conf if hand_visible else 0.0,
                    "added":    added,
                    "sentence": list(state["sentence"]),
                    "hand":     hand_visible,
                    "done":     sentence_done,
                    "buffer_size": len(state["buffer"]),
                    "buffer_max":  SEQ_LEN,
                    # DEBUG
                    "top5":     top5_debug,
                    "margin":   pred_margin,
                    "margin_ok": pred_margin >= MARGIN_MIN,
                    "conf_ok":   pred_conf >= CONF_THRESHOLD,
                    "smooth_top": [{"w": w, "c": c} for w, c in top_smooth if c > 0],
                    "smooth_needed": SMOOTH_NEEDED,
                })

                # Cumle bittiyse sifirla
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

# Separate state for HTTP predict_frame (mobile app)
# Use SAME buffer size as WebSocket (SEQ_LEN=30) for consistent predictions
http_state = {
    "buffer": deque(maxlen=SEQ_LEN),
    "holistic": None,
}

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """HTTP endpoint for mobile app - same logic as WebSocket for consistency"""
    import base64
    try:
        data = request.json
        image_base64 = data.get("image_base64", "")
        if not image_base64:
            return jsonify({"success": False, "error": "No image"})

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Cannot decode image"})

        # Flutter takePicture already gives correctly oriented pixels (no rotation needed)

        # Flip horizontally if requested (front camera mirror)
        if data.get("flip", False):
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        if h > w:
            # Portrait: letterbox to landscape (pad sides with black)
            # so person stays upright, same as webcam landscape frame
            new_w = int(h * 640 / 480)
            canvas = np.zeros((h, new_w, 3), dtype=np.uint8)
            offset = (new_w - w) // 2
            canvas[:, offset:offset+w] = img
            img = canvas

        # Resize to 640x480 to match web canvas dimensions
        img = cv2.resize(img, (640, 480))

        # Persistent holistic with static_image_mode=True for HTTP
        # (each frame is independent - no tracking state needed)
        if http_state["holistic"] is None:
            http_state["holistic"] = mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=0,  # Optimized: was 1, mobil icin daha hizli
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = http_state["holistic"].process(rgb)
        lm = extract_landmarks(results)

        hand_visible = (results.left_hand_landmarks is not None or
                        results.right_hand_landmarks is not None)

        # Only add to buffer if hands are detected (don't pollute with empty frames)
        if hand_visible:
            http_state["buffer"].append(lm)

        buf_len = len(http_state["buffer"])

        if buf_len < SEQ_LEN:
            return jsonify({
                "success": True, "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len, "hands_detected": hand_visible,
                "message": f"Buffer: {buf_len}/{SEQ_LEN}"
            })

        # Only predict if hands are currently visible
        if not hand_visible:
            return jsonify({
                "success": True, "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len, "hands_detected": False,
                "message": "El algilanmadi"
            })

        seq = np.clip(np.array(http_state["buffer"], dtype=np.float32), 0, 1)
        inp = torch.tensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
            prob = torch.softmax(out, dim=1)[0]
            conf, idx = prob.max(0)
            pred_conf = conf.item()
            pred_word = idx_to_label[idx.item()]

        return jsonify({
            "success": True, "prediction_text": pred_word, "confidence": pred_conf,
            "buffer_size": buf_len, "hands_detected": hand_visible
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@app.route("/reset_buffer", methods=["POST"])
def reset_buffer():
    http_state["buffer"] = deque(maxlen=SEQ_LEN)
    state["buffer"] = deque(maxlen=SEQ_LEN)
    state["smooth_counts"] = {k: 0 for k in label_map}
    state["sentence"] = []
    return jsonify({"success": True})

@app.route("/")
def index():
    return render_template_string(HTML)

# ---- Avatar & Mobile routes ----
import os, json
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
AVATAR_DIR = os.path.join(WEB_DIR, "avatar")
POSES_FILE = os.path.join(AVATAR_DIR, "saved_poses.json")

@app.route("/manifest.json")
def serve_manifest():
    from flask import send_from_directory
    return send_from_directory(WEB_DIR, "manifest.json")

@app.route("/favicon.ico")
def serve_favicon():
    return "", 204

@app.route("/mobile.html")
def serve_mobile():
    from flask import send_from_directory
    return send_from_directory(AVATAR_DIR, "mobile.html")

@app.route("/avatar/<path:filename>")
def serve_avatar(filename):
    from flask import send_from_directory
    return send_from_directory(AVATAR_DIR, filename)

@app.route("/api/list-poses")
def list_poses():
    if os.path.exists(POSES_FILE):
        with open(POSES_FILE, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route("/api/save-pose", methods=["POST"])
def save_pose():
    data = request.json
    poses = {}
    if os.path.exists(POSES_FILE):
        with open(POSES_FILE, "r", encoding="utf-8") as f:
            poses = json.load(f)
    word = data.get("word", "")
    pose = data.get("pose", {})
    if word:
        poses[word] = pose
        with open(POSES_FILE, "w", encoding="utf-8") as f:
            json.dump(poses, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True, "count": len(poses)})

@app.route("/api/word-request", methods=["POST"])
def word_request():
    data = request.json
    word = data.get("word", "").strip()
    if not word:
        return jsonify({"error": "empty"}), 400
    req_file = os.path.join(os.path.dirname(POSES_FILE), "word_requests.json")
    requests = []
    if os.path.exists(req_file):
        with open(req_file, "r", encoding="utf-8") as f:
            requests = json.load(f)
    requests.append({"word": word, "date": datetime.now().isoformat()})
    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(requests, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SignBridge Web Arayuzu + YOLO")
    print("="*50)
    print(f"Kelimeler ({num_classes}): {', '.join(label_map.keys())}")
    print(f"Adres: http://localhost:5051 (YOLO versiyonu)")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5051, debug=False, allow_unsafe_werkzeug=True)
