"""Sistem mimari diyagrami — temiz, sadece yatay akış."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"C:/Projects/sign_bridge/experiment_v2/poster_artifacts")
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(18, 7.5), dpi=150)
ax.set_xlim(0, 18)
ax.set_ylim(0, 7.5)
ax.set_aspect("equal")
ax.axis("off")

C_INPUT  = "#6C63FF"
C_PROC   = "#22D3EE"
C_MODEL  = "#A78BFA"
C_NLP    = "#EC4899"
C_OUT    = "#10B981"

def box(x, y, w, h, text, color, fontsize=14):
    bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                        linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.95)
    ax.add_patch(bb)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold",
            multialignment="center")

def arrow(x1, y1, x2, y2, label=""):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>,head_width=0.4,head_length=0.5",
                        linewidth=2.5, color="#475569")
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.4, label,
                ha="center", va="center", fontsize=11, color="#1E293B",
                style="italic", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="#CBD5E1", linewidth=1))

# ============ ANA BASLIK ============
ax.text(9, 7.05, "SignBridge — Sistem Mimarisi",
        ha="center", fontsize=22, fontweight="bold", color="#0F172A")

# ============ ÜST AKIS: TID → TURKÇE ============
ax.text(9, 6.3, "TİD → TÜRKÇE  (Tanıma)", ha="center",
        fontsize=16, fontweight="bold", color="#6C63FF")

# 6 kutu, esit aralikli yerlestir
y1 = 4.5
h1 = 1.1
# Toplam 17 birim genislik kullan, 6 kutu + 5 ok
widths = [2.0, 2.4, 2.0, 2.4, 2.2, 2.4]   # Kamera, MP, GRU, NLP, Cümle, TTS
gap = 0.5
total = sum(widths) + gap * (len(widths)-1)
start_x = (18 - total) / 2
xs = []
x = start_x
for w in widths:
    xs.append(x)
    x += w + gap

labels = [
    ("KAMERA\n30 FPS", C_INPUT),
    ("MediaPipe\nHolistic", C_PROC),
    ("GRU\n2-layer · 256h", C_MODEL),
    ("Akıllı NLP\nÇekim · Bağlam", C_NLP),
    ("TÜRKÇE\nCÜMLE", C_OUT),
    ("TTS\n(sesli okuma)", "#059669"),
]
arrows_labels = ["RGB frame", "1755-D vektör", "kelime + conf", "çekimli cümle", ""]

for i, ((text, color), w) in enumerate(zip(labels, widths)):
    box(xs[i], y1, w, h1, text, color)
    if i < len(labels) - 1:
        arrow(xs[i] + w, y1 + h1/2, xs[i+1], y1 + h1/2,
              arrows_labels[i])

# ============ AYIRICI ============
ax.plot([0.5, 17.5], [3.4, 3.4], "--", color="#94A3B8", linewidth=1.5, alpha=0.6)

# ============ ALT AKIS: TURKÇE → TID ============
ax.text(9, 3.0, "TÜRKÇE → TİD  (Avatar ile Üretim)", ha="center",
        fontsize=16, fontweight="bold", color="#10B981")

y2 = 1.0
h2 = 1.1
widths2 = [2.6, 3.0, 3.2, 3.0]
gap2 = 0.6
total2 = sum(widths2) + gap2 * (len(widths2)-1)
start_x2 = (18 - total2) / 2
xs2 = []
x = start_x2
for w in widths2:
    xs2.append(x)
    x += w + gap2

labels2 = [
    ("TÜRKÇE METİN\nyazı / mikrofon", C_OUT),
    ("Cümle Ayrıştırıcı\nKelime → İşaret", C_NLP),
    ("122 Poz Veritabanı\n(JSON keyframe)", C_MODEL),
    ("3B AVATAR\nThree.js · rain.glb", C_INPUT),
]
arrows_labels2 = ["kelime listesi", "poz adı", "poz dizisi"]

for i, ((text, color), w) in enumerate(zip(labels2, widths2)):
    box(xs2[i], y2, w, h2, text, color)
    if i < len(labels2) - 1:
        arrow(xs2[i] + w, y2 + h2/2, xs2[i+1], y2 + h2/2,
              arrows_labels2[i])

# ============ FOOTER ============
ax.text(9, 0.2,
        "Teknoloji:  PyTorch  ·  MediaPipe Holistic  ·  Flask + SocketIO  ·  Flutter  ·  Three.js",
        ha="center", fontsize=11, color="#64748B", style="italic",
        fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "architecture_diagram.png", dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
plt.close()
print(f"Kaydedildi: {OUT / 'architecture_diagram.png'}")
