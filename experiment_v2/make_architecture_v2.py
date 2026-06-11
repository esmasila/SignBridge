"""Sistem mimari diyagrami v2 — BÜYÜK kutular + BÜYÜK yazı + canlı renkler."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"C:/Projects/sign_bridge/experiment_v2/poster_artifacts")
OUT.mkdir(exist_ok=True)

# Daha büyük figür, daha büyük yazı = uzaktan okunabilir
fig, ax = plt.subplots(figsize=(20, 9), dpi=180)
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")

# Canlı renkler (daha doygun)
C_INPUT  = "#6C63FF"   # Mor (kamera/text)
C_PROC   = "#06B6D4"   # Cyan (MediaPipe)
C_MODEL  = "#8B5CF6"   # Koyu mor (GRU)
C_NLP    = "#EC4899"   # Pembe (NLP)
C_OUT    = "#10B981"   # Yeşil (çıktı)

def box(x, y, w, h, text, color, fontsize=18):
    """Daha büyük kutular, daha büyük yazı."""
    bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25",
                        linewidth=3, edgecolor=color, facecolor=color, alpha=0.95)
    ax.add_patch(bb)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold",
            multialignment="center")

def arrow(x1, y1, x2, y2, label=""):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>,head_width=0.55,head_length=0.7",
                        linewidth=3.5, color="#334155")
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.45, label,
                ha="center", va="center", fontsize=14, color="#1E293B",
                style="italic", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CBD5E1", linewidth=1.2))

# Ana başlık (büyük)
ax.text(10, 8.55, "SignBridge — Sistem Mimarisi",
        ha="center", fontsize=28, fontweight="bold", color="#0F172A")

# Üst akış başlık
ax.text(10, 7.7, "TİD → TÜRKÇE  (Tanıma Akışı)", ha="center",
        fontsize=20, fontweight="bold", color="#6C63FF")

# Üst sıra — 6 kutu, daha büyük
y1 = 5.7
h1 = 1.5
widths = [2.3, 2.8, 2.4, 2.8, 2.5, 2.6]
gap = 0.4
total = sum(widths) + gap * (len(widths)-1)
start_x = (20 - total) / 2
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
    box(xs[i], y1, w, h1, text, color, fontsize=17)
    if i < len(labels) - 1:
        arrow(xs[i] + w, y1 + h1/2, xs[i+1], y1 + h1/2,
              arrows_labels[i])

# Ayırıcı
ax.plot([0.5, 19.5], [4.3, 4.3], "--", color="#94A3B8", linewidth=2, alpha=0.6)

# Alt başlık
ax.text(10, 3.85, "TÜRKÇE → TİD  (Avatar ile Üretim)", ha="center",
        fontsize=20, fontweight="bold", color="#10B981")

# Alt sıra — 4 kutu, büyük
y2 = 1.6
h2 = 1.5
widths2 = [3.0, 3.5, 3.8, 3.5]
gap2 = 0.5
total2 = sum(widths2) + gap2 * (len(widths2)-1)
start_x2 = (20 - total2) / 2
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
    box(xs2[i], y2, w, h2, text, color, fontsize=17)
    if i < len(labels2) - 1:
        arrow(xs2[i] + w, y2 + h2/2, xs2[i+1], y2 + h2/2,
              arrows_labels2[i])

# Footer
ax.text(10, 0.35,
        "Teknoloji:  PyTorch  ·  MediaPipe Holistic  ·  Flask + SocketIO  ·  Flutter  ·  Three.js",
        ha="center", fontsize=14, color="#475569", style="italic",
        fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "architecture_diagram.png", dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.4)
plt.close()
print(f"Kaydedildi: {OUT / 'architecture_diagram.png'}")
