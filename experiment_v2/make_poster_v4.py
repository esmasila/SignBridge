"""SignBridge posteri v4 - DAHA YOĞUN, TÜM görseller dolu, boşluksuz.
Mobil ss (1:2 tall) için TELEFON sütunları kullanıyoruz.
Web ss (2:1 wide) için GENİŞ bantlar kullanıyoruz.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path
from PIL import Image

TEMPLATE = r"C:\Users\leven\Downloads\template.pptx"
OUT_PATH = Path(r"C:\Users\leven\Downloads\SignBridge_poster.pptx")
ART = Path(r"C:/Projects/sign_bridge/experiment_v2/poster_artifacts")
IMG = Path(r"C:/Users/leven/Downloads/poster-images")

# Renkler
PURPLE   = RGBColor(0x6C, 0x63, 0xFF)
PURPLE_D = RGBColor(0x4C, 0x44, 0xCF)
CYAN     = RGBColor(0x06, 0xB6, 0xD4)
PINK     = RGBColor(0xEC, 0x48, 0x99)
GREEN    = RGBColor(0x10, 0xB9, 0x81)
ORANGE   = RGBColor(0xF5, 0x9E, 0x0B)
SOFT_BG  = RGBColor(0xF5, 0xF3, 0xFF)
SOFT_CY  = RGBColor(0xEC, 0xFE, 0xFF)
SOFT_PK  = RGBColor(0xFD, 0xF2, 0xF8)
SOFT_GR  = RGBColor(0xEC, 0xFD, 0xF5)
CARD_BG  = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TXT = RGBColor(0x0F, 0x17, 0x2A)
DARK_2   = RGBColor(0x1E, 0x29, 0x3B)
GREY_TXT = RGBColor(0x47, 0x55, 0x69)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
BORDER   = RGBColor(0xCB, 0xD5, 0xE1)

FONT = "Calibri"

prs = Presentation(TEMPLATE)
slide = prs.slides[0]

# Talimat balonlarini sil
for sh in list(slide.shapes):
    x_cm = sh.left / 360000
    if x_cm < 0 or x_cm > 65:
        sh._element.getparent().remove(sh._element)

# --- Helpers ---
def add_rect(x, y, w, h, color=None, fill=True, line_color=None,
             line_width=1.5, rounding=0.04):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Cm(x), Cm(y), Cm(w), Cm(h))
    sh.adjustments[0] = rounding
    if fill and color is not None:
        sh.fill.solid()
        sh.fill.fore_color.rgb = color
    else:
        sh.fill.background()
    if line_color is not None:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(line_width)
    else:
        sh.line.fill.background()
    return sh

def add_text(x, y, w, h, text, size=14, bold=False, color=DARK_TXT,
             font=FONT, align=PP_ALIGN.LEFT, vert=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Cm(0.15)
    tf.margin_top = tf.margin_bottom = Cm(0.05)
    tf.vertical_anchor = vert
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb

def section_header(x, y, w, title, color=PURPLE, icon="", size=22):
    add_rect(x, y, 0.7, 1.5, color)
    txt = f"  {icon}  {title}" if icon else f"  {title}"
    add_text(x + 0.9, y - 0.05, w - 0.9, 1.6, txt,
             size=size, bold=True, color=DARK_TXT,
             vert=MSO_ANCHOR.MIDDLE)

def add_pic_fit(x, y, max_w, max_h, path, center=True):
    img = Image.open(str(path))
    iw, ih = img.size
    ratio = iw / ih
    if max_w / max_h > ratio:
        h = max_h; w = h * ratio
    else:
        w = max_w; h = w / ratio
    if center:
        xo = (max_w - w) / 2; yo = (max_h - h) / 2
    else:
        xo = yo = 0
    slide.shapes.add_picture(str(path), Cm(x + xo), Cm(y + yo), Cm(w), Cm(h))
    return w, h

# --- Basliklari guncelle ---
for sh in list(slide.shapes):
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text.strip()
    if "PROJE BAŞLIĞI" in t:
        sh._element.getparent().remove(sh._element)
        add_text(5.5, 10.5, 59, 4.5, "SignBridge", size=82, bold=True,
                 color=PURPLE, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
        add_text(5.5, 15.4, 59, 2.0,
                 "Türk İşaret Dili için Çift Yönlü Çeviri Sistemi",
                 size=28, bold=True, color=DARK_TXT,
                 align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
        continue
    if "Proje Sahibi" in t and "Adı SOYADI" in t:
        tf = sh.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = "PROJE SAHİBİ"
        r.font.name = FONT
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = GREY_TXT
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = "Esma Sıla ŞAHİNCİ"
        r2.font.name = FONT
        r2.font.size = Pt(22)
        r2.font.bold = True
        r2.font.color.rgb = PURPLE
    elif "Proje Danışmanı" in t:
        tf = sh.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = "PROJE DANIŞMANI"
        r.font.name = FONT
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = GREY_TXT
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = "Öğr. Gör. Kadir HALTAŞ"
        r2.font.name = FONT
        r2.font.size = Pt(22)
        r2.font.bold = True
        r2.font.color.rgb = CYAN

# ============ ALT İÇERİK (YOĞUN, DOLU) ============

# ============ ŞERİT 1 (y=21-32, 11cm): ÖZET + 2 mobil ss yan yana ============
y = 21.0
add_rect(2.5, y, 65, 11, SOFT_BG, rounding=0.04)
section_header(3.0, y + 0.3, 64, "ÖZET", PURPLE, icon="📝")

# Sol: özet metin
add_text(3.2, y + 2.0, 35.5, 9,
         "Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmasına rağmen Türk "
         "İşaret Dili için son kullanıcıya yönelik çeviri çözümleri yok denecek kadar "
         "azdır. Bu çalışmada TİD ile Türkçe arasında ÇİFT YÖNLÜ çeviri yapan bütünleşik "
         "bir sistem geliştirilmiştir.\n\n"
         "Sistem üç ana bileşenden oluşur: MediaPipe Holistic ile gerçek zamanlı işaret "
         "noktası çıkarımı, 122 kelimelik GRU sinir ağı ile dizi sınıflandırma ve akıllı "
         "Türkçe NLP modülü ile dilbilgisi doğru cümle üretimi. Tersine yönde "
         "kullanıcının yazdığı/söylediği Türkçe cümle 3B avatar üzerinde işaret dili "
         "olarak gösterilir.\n\n"
         "Model 122 sınıf üzerinde %99.88 doğrulama doğruluğuna ulaşmış; sistem hem "
         "MOBİL (Flutter) hem WEB (Flask + Three.js) platformlarında çalışmaktadır.",
         size=14, color=DARK_TXT, align=PP_ALIGN.JUSTIFY)

# Sag: 2 mobil ss yan yana (TID -> Turkce + Turkce -> TID)
hero_x = 40
hero_w = 27.5
add_rect(hero_x, y + 1.8, hero_w, 9.0, DARK_2, rounding=0.03)
add_text(hero_x, y + 1.85, hero_w, 0.9,
         "🎬  ÇİFT YÖNLÜ ÇEVİRİ — Canlı Demo",
         size=13, bold=True, color=CYAN,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
# 2 telefon
phone_y = y + 2.7
phone_h = 7.8
phone_w = (hero_w - 1.5) / 2  # ~13cm
# Sol telefon: TID -> Turkce
add_text(hero_x + 0.5, phone_y - 0.4, phone_w, 0.6,
         "📷  TİD → Türkçe",
         size=10, bold=True, color=PURPLE,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(hero_x + 0.5, phone_y, phone_w, phone_h,
            IMG / "mobil_kamera_nasılsın.jpeg")
# Sag telefon: Turkce -> TID (avatar)
add_text(hero_x + 1.0 + phone_w, phone_y - 0.4, phone_w, 0.6,
         "🎭  Türkçe → TİD",
         size=10, bold=True, color=GREEN,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(hero_x + 1.0 + phone_w, phone_y, phone_w, phone_h,
            IMG / "mobil_avatar_merhaba.jpeg")

# ============ ŞERİT 2 (y=33-49, 16cm): AMAÇ + MİMARİ + WEB AVATAR ============
y = 33.0

# Sol: AMAÇ 2x2 grid (kompakt)
amac_w = 22
add_rect(2.5, y, amac_w, 16, CARD_BG, line_color=BORDER)
section_header(3.0, y + 0.3, amac_w - 1, "AMAÇ", PINK, icon="🎯")
goals = [
    ("TİD → Türkçe", "Canlı kameradan işaretleri\ngramatik Türkçeye çevir", PURPLE),
    ("Türkçe → TİD", "3B avatar ile yazılı/sözlü\nTürkçe'yi işaret diline çevir", CYAN),
    ("Çift Platform", "Mobil (Flutter) +\nWeb (Three.js)\neş zamanlı", GREEN),
    ("Düşük Gecikme", "İşaret başına\n< 1.5 saniye\ngerçek zamanlı", ORANGE),
]
gy = y + 2.2
# 2x2 grid
cell_w = (amac_w - 1.0) / 2
cell_h = 6.5
for i, (head, body, color) in enumerate(goals):
    col = i % 2
    row = i // 2
    gx = 3.0 + col * (cell_w + 0.2)
    cy = gy + row * (cell_h + 0.3)
    add_rect(gx, cy, cell_w, cell_h, color)
    # Numara
    add_text(gx, cy + 0.3, cell_w, 1.5, str(i+1),
             size=36, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
    add_text(gx, cy + 1.8, cell_w, 1.0, head,
             size=14, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
    add_text(gx + 0.3, cy + 3.0, cell_w - 0.6, 3.3, body,
             size=11, color=RGBColor(0xF1, 0xF5, 0xF9),
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.TOP)

# Sağ: MIMARI + WEB AVATAR alt-alta
mim_x = 26
mim_w = 41.5

# Mimari diyagram (üst, 10cm)
add_rect(mim_x, y, mim_w, 10, CARD_BG, line_color=BORDER)
section_header(mim_x + 0.5, y + 0.3, mim_w - 1, "SİSTEM MİMARİSİ", CYAN, icon="🏗️")
add_pic_fit(mim_x + 0.5, y + 2.0, mim_w - 1, 7.5,
            ART / "architecture_diagram.png")

# Web avatar (alt, 5.5cm) - geniş wide aspect 2.22
web_av_y = y + 10.3
add_rect(mim_x, web_av_y, mim_w, 5.7, DARK_2)
add_text(mim_x + 0.5, web_av_y + 0.3, mim_w - 1, 0.9,
         "🎭  WEB AVATAR — 3B Karakter ile İşaret Dili Üretimi",
         size=14, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(mim_x + 0.5, web_av_y + 1.2, mim_w - 1, 4.3,
            IMG / "web_avatar.jpeg")

# ============ ŞERİT 3 (y=50-66, 16cm): YÖNTEM + 1 mobil ss kenar ============
y = 50.0
add_rect(2.5, y, 65, 15, CARD_BG, line_color=BORDER)
section_header(3.0, y + 0.3, 64, "YÖNTEM", PURPLE, icon="🧠")

# 3 kart + mobil ss (sağda)
sub_y = y + 2.2
content_w = 50  # 3 kart icin
card_w = (content_w - 1.2) / 3  # ~16.3cm
gap = 0.6

cards_data = [
    ("📊  VERİ SETİ", PURPLE, SOFT_BG,
     "•  122 TİD kelimesi\n"
     "•  105.182 örnek (augment)\n\n"
     "•  Her örnek:\n"
     "   30 frame × 1755 boyut\n"
     "   1629 landmark\n"
     "   + 126 hız bileşeni\n\n"
     "•  Augmentation:\n"
     "   gürültü · ölçek · ayna\n"
     "   çevirme · döndürme\n"
     "   zaman bükme"),
    ("🧠  GRU MODELİ", CYAN, SOFT_CY,
     "•  2 katmanlı GRU\n   (Gated Recurrent Unit)\n\n"
     "•  Giriş: 1755 boyut\n"
     "   Gizli: 256 · Dropout 0.3\n\n"
     "•  Sınıflandırıcı:\n"
     "   FC(256→128) + ReLU\n"
     "   FC(128→122)\n\n"
     "•  ~1.9 M parametre\n"
     "•  AdamW · LR=1e-3\n"
     "   80 epoch ~20 dakika"),
    ("🌐  AKILLI NLP", PINK, SOFT_PK,
     "•  Özne tespiti:\n"
     "   BEN/SEN/BERABER/KENDI\n"
     "   → doğru fiil çekimi\n\n"
     "•  Bağlam ayrımı:\n"
     "   KAHVE ↔ KAFE benzer\n"
     "   işaretler ayırt edilir\n\n"
     "•  Türkçe ekler:\n"
     "   ablatif (-DAn)\n"
     "   locatif (-DA)\n\n"
     "•  Kohortatif soru:\n"
     "   '…elim mi?'"),
]
for i, (head, color, bg, body) in enumerate(cards_data):
    sx = 3.0 + i * (card_w + gap)
    add_rect(sx, sub_y, card_w, 12.4, bg, rounding=0.06)
    # head bar
    add_rect(sx, sub_y, card_w, 1.4, color, rounding=0.06)
    add_text(sx, sub_y, card_w, 1.4, head,
             size=15, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
    add_text(sx + 0.4, sub_y + 1.7, card_w - 0.8, 10.5, body,
             size=12, color=DARK_TXT)

# Sağ: MOBİL KAMERA SS (3. kartin sagi)
mob_x = 54.5
mob_w = 13
mob_y = sub_y
mob_h = 12.4
add_rect(mob_x, mob_y, mob_w, mob_h, DARK_2, rounding=0.04)
add_text(mob_x, mob_y + 0.2, mob_w, 0.8,
         "📱 Mobil Kamera",
         size=11, bold=True, color=CYAN,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(mob_x + 0.3, mob_y + 1.1, mob_w - 0.6, mob_h - 1.4,
            IMG / "mobil_kamera_merhaba.jpeg")

# ============ ŞERİT 4 (y=66-83, 17cm): SONUÇLAR (büyük) ============
y = 66.0
add_rect(2.5, y, 65, 17, CARD_BG, line_color=BORDER)
section_header(3.0, y + 0.3, 64, "SONUÇLAR", GREEN, icon="📈")

# Sol: %99.97 + tablo
big_x = 3.0
big_w = 13
big_y = y + 2.2
add_rect(big_x, big_y, big_w, 5.0, PURPLE)
add_text(big_x, big_y + 0.2, big_w, 2.8, "%99.97",
         size=52, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_text(big_x, big_y + 3.0, big_w, 1.7, "GENEL DOĞRULUK\n3.660 örnek",
         size=11, bold=True, color=RGBColor(0xCB, 0xD5, 0xE1),
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

tbl_y = big_y + 5.5
add_text(big_x, tbl_y - 0.6, big_w, 0.6, "METRİKLER",
         size=11, bold=True, color=GREY_TXT)
stats = [
    ("Doğrulama doğruluğu",      "%99.88"),
    ("Eğitim doğruluğu",         "%100.0"),
    ("Sınıf sayısı",             "122"),
    ("Çıkarım (GPU)",            "~50 ms"),
    ("Çıkarım (CPU)",            "~80 ms"),
    ("Mobil canlı",              "~250 ms"),
    ("Eğitim",                   "20 dk"),
    ("Toplam veri",              "105k"),
]
row_h = 0.7
for i, (k, v) in enumerate(stats):
    bg = SOFT_BG if i % 2 == 0 else None
    if bg:
        add_rect(big_x - 0.1, tbl_y + i * row_h, big_w + 0.2, row_h, bg, rounding=0.0)
    add_text(big_x + 0.2, tbl_y + i * row_h, big_w * 0.6, row_h, k,
             size=10, color=DARK_TXT, vert=MSO_ANCHOR.MIDDLE)
    add_text(big_x + big_w * 0.55, tbl_y + i * row_h, big_w * 0.4, row_h, v,
             size=11, bold=True, color=PURPLE,
             align=PP_ALIGN.RIGHT, vert=MSO_ANCHOR.MIDDLE)

# Sağ: 2 grafik yan yana
gx = big_x + big_w + 0.8
gy_start = y + 2.2
gw_total = 65 - (gx - 2.5) - 0.8
gw_each = (gw_total - 0.5) / 2
gh_full = 14

add_text(gx, gy_start, gw_each, 0.7, "Eğitim Süreci  (80 epoch)",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx, gy_start + 0.8, gw_each, gh_full - 0.8,
            ART / "training_history_122.png")

add_text(gx + gw_each + 0.5, gy_start, gw_each, 0.7,
         "Karışıklık Matrisi  (30 sık kelime)",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx + gw_each + 0.5, gy_start + 0.8, gw_each, gh_full - 0.8,
            ART / "confusion_matrix_122.png")

# ============ ŞERİT 5 (y=84-89, 5cm): WEB KAMERA + Zorluklar ============
y = 84.0
add_rect(2.5, y, 65, 4.5, CARD_BG, line_color=BORDER)
add_text(3.0, y + 0.1, 64, 0.7, "💻  WEB ARAYÜZÜ — Çift Kamera (Landmark + Ham)",
         size=13, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(3.0, y + 0.9, 32, 3.4, IMG / "web_kamera.jpeg", center=False)

# Sağ: Zorluklar
add_rect(36.5, y + 0.6, 30.5, 3.6, SOFT_PK, rounding=0.06)
add_text(36.8, y + 0.75, 30, 0.7, "🔧  Karşılaşılan Zorluklar & Çözümler",
         size=12, bold=True, color=PINK)
add_text(36.8, y + 1.45, 30, 2.7,
         "•  Benzer işaretler (KAHVE↔KAFE): per-kelime güven eşiği\n"
         "•  MediaPipe paralel segfault: threading.Lock ile çözüldü\n"
         "•  Mobil kamera donması: zoom stream sonrası ertelendi\n"
         "•  Geçiş gürültüsü: 400ms cooldown + buffer reset",
         size=10, color=DARK_TXT)

# ============ ŞERİT 6 (y=89-98, 9cm): SEN DE DENE — 4 telefon mockup ============
y = 89.0
add_rect(2.5, y, 65, 9.0, DARK_2)

# Sol: davet (15cm)
add_text(3.5, y + 0.4, 16, 1.5, "🎮  SEN DE DENE!",
         size=22, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_text(3.5, y + 1.9, 16, 6.5,
         "Avatar 122 farklı Türkçe\nişaret kelimesini\ngerçek zamanlı oynatabilir.\n\n"
         "Yanımdaki bilgisayardan\nkomutla deneyebilirsiniz!\n\n"
         "→ Mikrofona konuş\n→ Avatar işaret diline çevirsin",
         size=11, color=RGBColor(0xCB, 0xD5, 0xE1))

# Sag: 4 telefon mockup (5cm wide × 8cm tall = mobile aspect 0.625 — biraz crop ama dolu)
av_imgs = [
    (IMG / "mobil_avatar_merhaba.jpeg", "MERHABA", PURPLE),
    (IMG / "mobil_avatar_ataturk.jpeg", "ATATÜRK", CYAN),
    (IMG / "mobil_avatar_elhamdulıllah.jpeg", "ELHAMDÜLİLLAH", PINK),
    (IMG / "mobil_avatar_hoscakal.jpeg", "HOŞÇA KAL", GREEN),
]
# 4 telefon arasi - hepsi kompakt
av_total_x = 21
av_total_w = 46
av_each_w = (av_total_w - 3 * 0.5) / 4  # ~11.1cm her
av_h = 8.0

for i, (path, label, color) in enumerate(av_imgs):
    x = av_total_x + i * (av_each_w + 0.5)
    img_y = y + 0.5
    # Beyaz mockup arka plan
    add_rect(x, img_y, av_each_w, av_h, CARD_BG, rounding=0.04)
    # Mobil ss - aspect korunarak ortalanmis
    add_pic_fit(x + 0.15, img_y + 0.15, av_each_w - 0.3, av_h - 1.3, path)
    # Label badge
    add_rect(x + 0.5, img_y + av_h - 1.0, av_each_w - 1.0, 0.85, color, rounding=0.5)
    add_text(x, img_y + av_h - 1.0, av_each_w, 0.85, label,
             size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# ============ FOOTER ============
add_text(2.5, 98.5, 65, 1.0,
         "SignBridge © 2026  ·  Nevşehir Hacı Bektaş Veli Üniversitesi  ·  Bilgisayar Mühendisliği Bitirme Projesi",
         size=10, color=GREY_TXT, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

prs.save(str(OUT_PATH))
print(f"\n✅ Poster v4 kaydedildi: {OUT_PATH}")
