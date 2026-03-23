"""
experiment_v2 Web Arayuzu
Flask + WebSocket ile gercek zamanli isaret dili cevirisi

Baslat: python experiment_v2/web_app.py
Tarayici: http://localhost:5050
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
CKPT    = BASE / "checkpoints" / "best_model.pt"
SEQ_LEN = 30

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

# ---- Uygulama durumu ----
state = {
    "buffer":        deque(maxlen=SEQ_LEN),
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

CONF_THRESHOLD = 0.90   # yuksek guven gerektir
SMOOTH_NEEDED  = 10     # 10 frame art arda ayni kelime
COOLDOWN_SEC   = 1.5    # kelimeler arasi bekleme
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
<title>SignBridge - İşaret Dili Çevirisi</title>
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
  <h1>SignBridge</h1>
  <span>Türk İşaret Dili Çevirisi</span>
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
    state["smooth_counts"] = {k: 0 for k in label_map}

    cap = cv2.VideoCapture(0)
    state["cap"] = cap
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
        try:
            while state["running"]:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Landmark ciz
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS)

                lm = extract_landmarks(results)
                state["buffer"].append(lm)

                hand_visible = (results.left_hand_landmarks is not None or
                                results.right_hand_landmarks is not None)

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

                if len(state["buffer"]) == SEQ_LEN:
                    seq = np.clip(np.array(state["buffer"], dtype=np.float32), 0, 1)
                    inp = torch.tensor(seq).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out  = model(inp)
                        prob = torch.softmax(out, dim=1)[0]
                        conf, idx = prob.max(0)
                        pred_conf = conf.item()
                        pred_word = idx_to_label[idx.item()]

                    state["current_word"] = pred_word
                    state["current_conf"] = pred_conf

                    now = time.time()
                    in_cooldown = (now - state["last_add_time"]) < COOLDOWN_SEC

                    if pred_conf >= CONF_THRESHOLD and hand_visible and not in_cooldown:
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
                socketio.emit("prediction", {
                    "word":     pred_word if hand_visible else "",
                    "conf":     pred_conf if hand_visible else 0.0,
                    "added":    added,
                    "sentence": list(state["sentence"]),
                    "hand":     hand_visible,
                    "done":     sentence_done,
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
http_state = {
    "buffer": deque(maxlen=SEQ_LEN),
    "holistic": None,
}

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """HTTP endpoint for mobile app - accepts base64 image, returns prediction"""
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

        # Separate holistic for HTTP mode
        if http_state["holistic"] is None:
            http_state["holistic"] = mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = http_state["holistic"].process(rgb)
        lm = extract_landmarks(results)

        hand_visible = (results.left_hand_landmarks is not None or
                        results.right_hand_landmarks is not None)

        http_state["buffer"].append(lm)
        buf_len = len(http_state["buffer"])

        if buf_len < SEQ_LEN:
            return jsonify({
                "success": True, "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len, "hands_detected": hand_visible,
                "message": f"Buffer: {buf_len}/{SEQ_LEN}"
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

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SignBridge Web Arayuzu")
    print("="*50)
    print(f"Kelimeler ({num_classes}): {', '.join(label_map.keys())}")
    print(f"Adres: http://localhost:5050")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)
