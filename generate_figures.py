# -*- coding: utf-8 -*-
"""Tez icin kavramsal sema gorselleri uretir (3.3, 3.4, 3.5, 3.6)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\leven\Downloads\thesis-figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

PURPLE = "#6D28D9"
TEAL = "#0EA5A4"
PINK = "#DB2777"
GREEN = "#059669"
GRAY = "#475569"
LIGHT = "#EDE9FE"


def box(ax, x, y, w, h, text, fc, tc="white", fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=16, lw=2, color=color))


# ---------- Sekil 3.3: Dizi (30 kare) ----------
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
ax.text(5, 2.75, "Bir kelime = 30 karelik dizi (~1 saniye)",
        ha="center", fontsize=13, fontweight="bold", color=PURPLE)
labels = ["Kare 1", "Kare 2", "Kare 3", "...", "Kare 29", "Kare 30"]
n = len(labels); w = 1.25; gap = 0.25
total = n * w + (n - 1) * gap
x0 = (10 - total) / 2
for i, lab in enumerate(labels):
    x = x0 + i * (w + gap)
    col = LIGHT if lab == "..." else TEAL
    tc = GRAY if lab == "..." else "white"
    box(ax, x, 1.1, w, 1.1, lab, col, tc=tc, fs=10)
    if i < n - 1:
        ax.annotate("", xy=(x + w + gap, 1.65), xytext=(x + w, 1.65),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3))
ax.annotate("", xy=(x0 + total, 0.6), xytext=(x0, 0.6),
            arrowprops=dict(arrowstyle="->", color=PINK, lw=2))
ax.text(5, 0.25, "zaman  →   her kare: 543 anahtar nokta → 1755 öznitelik",
        ha="center", fontsize=10, color=GRAY)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_3_3_dizi.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------- Sekil 3.4: Iki asamali normallestirme ----------
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
ax.text(5, 3.15, "İki Aşamalı Normalleştirme",
        ha="center", fontsize=13, fontweight="bold", color=PURPLE)
box(ax, 0.3, 1.1, 2.7, 1.4,
    "Ham koordinatlar\n(kameraya uzaklığa\nbağımlı)", GRAY, fs=10)
box(ax, 3.65, 1.1, 2.7, 1.4,
    "1) Omuz merkezli\nölçekleme\n(uzaklıktan bağımsız)", TEAL, fs=10)
box(ax, 7.0, 1.1, 2.7, 1.4,
    "2) Bilek merkezli\nölçekleme\n(parmak duruşu öne çıkar)", PURPLE, fs=10)
arrow(ax, 3.0, 1.8, 3.65, 1.8)
arrow(ax, 6.35, 1.8, 7.0, 1.8)
ax.text(5, 0.45, "Sonuç: aynı işaret, farklı uzaklık ve konumda benzer öznitelik üretir",
        ha="center", fontsize=10, color=GRAY)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_3_4_normallestirme.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------- Sekil 3.5: GRU model katman yapisi ----------
fig, ax = plt.subplots(figsize=(6.2, 6.8))
ax.set_xlim(0, 6); ax.set_ylim(0, 12); ax.axis("off")
ax.text(3, 11.5, "GRU Tabanlı Sınıflandırma Modeli",
        ha="center", fontsize=13, fontweight="bold", color=PURPLE)
layers = [
    ("Giriş: 30 × 1755\n(dizi × öznitelik)", GRAY),
    ("GRU Katmanı 1\n256 gizli birim", TEAL),
    ("GRU Katmanı 2\n256 gizli birim", TEAL),
    ("Tam Bağlı\nLinear(256 → 128)", PURPLE),
    ("ReLU", PINK),
    ("Tam Bağlı\nLinear(128 → 122)", PURPLE),
    ("Çıkış: 122 sınıf\n(kelime olasılıkları)", GREEN),
]
h = 1.25; gap = 0.35; w = 4.4; x = 0.8
y = 9.7
for txt, col in layers:
    box(ax, x, y, w, h, txt, col, fs=10)
    if y > 1:
        ax.annotate("", xy=(x + w / 2, y - gap), xytext=(x + w / 2, y),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6))
    y -= (h + gap)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_3_5_model.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------- Sekil 3.6: Gercek zamanli tahmin + yumusatma akisi ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
ax.text(5, 4.7, "Gerçek Zamanlı Tahmin ve Yumuşatma Akışı",
        ha="center", fontsize=13, fontweight="bold", color=PURPLE)
box(ax, 0.3, 3.0, 2.2, 1.0, "Kamera karesi", GRAY, fs=10)
box(ax, 2.9, 3.0, 2.2, 1.0, "MediaPipe +\nöznitelik", TEAL, fs=10)
box(ax, 5.5, 3.0, 2.2, 1.0, "Tampon\n(≥15 kare)", PURPLE, fs=10)
box(ax, 7.5, 1.6, 2.2, 1.0, "GRU tahmini\n(kelime + güven)", PINK, fs=10)
arrow(ax, 2.5, 3.5, 2.9, 3.5)
arrow(ax, 5.1, 3.5, 5.5, 3.5)
arrow(ax, 7.7, 3.0, 8.6, 2.6)
# karar
box(ax, 5.4, 1.6, 1.8, 1.0, "Güven ≥ 0,88\nve 2 kare tekrar?", "#F59E0B", fs=9)
arrow(ax, 7.5, 2.1, 7.2, 2.1)
box(ax, 2.9, 1.6, 2.0, 1.0, "Kelimeyi kabul et\n(cooldown 0,4 sn)", GREEN, fs=9)
arrow(ax, 5.4, 2.1, 4.9, 2.1)
box(ax, 2.9, 0.2, 2.0, 1.0, "Akıllı NLP →\nTürkçe cümle + TTS", PURPLE, fs=9)
arrow(ax, 3.9, 1.6, 3.9, 1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_3_6_akis.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Uretildi:", os.listdir(OUT))
