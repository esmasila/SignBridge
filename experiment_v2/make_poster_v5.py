"""SignBridge poster v5 - Gemini-inspired tasarim + TEZ CIKTILARI bolumu.
- Yumusak mor gradient arka plan
- Beyaz yuvarlak kartlar ve renkli ust banlar
- Hero element + catchphrase
- Telefon mockup tarzi
- Detayli icerik (akademik + uygulama)
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

# Renkler - Gemini referansindan ilham (yumusak mor + cyan + pembe)
PURPLE   = RGBColor(0x6C, 0x63, 0xFF)
PURPLE_D = RGBColor(0x4C, 0x44, 0xCF)
PURPLE_VL = RGBColor(0xE9, 0xE5, 0xFF)   # cok acik mor (bg)
PURPLE_L = RGBColor(0xF5, 0xF3, 0xFF)    # acik mor (kart bg)
CYAN     = RGBColor(0x06, 0xB6, 0xD4)
CYAN_L   = RGBColor(0xCF, 0xFA, 0xFE)
PINK     = RGBColor(0xEC, 0x48, 0x99)
PINK_L   = RGBColor(0xFC, 0xE7, 0xF3)
GREEN    = RGBColor(0x10, 0xB9, 0x81)
GREEN_L  = RGBColor(0xD1, 0xFA, 0xE5)
ORANGE   = RGBColor(0xF5, 0x9E, 0x0B)
ORANGE_L = RGBColor(0xFE, 0xF3, 0xC7)

CARD_BG  = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TXT = RGBColor(0x0F, 0x17, 0x2A)
DARK_2   = RGBColor(0x1E, 0x29, 0x3B)
DARK_3   = RGBColor(0x33, 0x35, 0x66)
GREY_TXT = RGBColor(0x47, 0x55, 0x69)
GREY_L   = RGBColor(0xE2, 0xE8, 0xF0)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
BORDER   = RGBColor(0xE0, 0xE7, 0xFF)

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

def add_oval(x, y, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                Cm(x), Cm(y), Cm(w), Cm(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh

def add_text(x, y, w, h, text, size=14, bold=False, color=DARK_TXT,
             font=FONT, align=PP_ALIGN.LEFT, vert=MSO_ANCHOR.TOP, italic=False):
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
        r.font.italic = italic
        r.font.color.rgb = color
    return tb

def card(x, y, w, h, title, color, bg_color, icon="", size_title=16):
    """Beyaz kart + renkli ust bar + baslik."""
    add_rect(x, y, w, h, bg_color, rounding=0.04, line_color=BORDER, line_width=1)
    add_rect(x, y, w, 1.5, color, rounding=0.04)
    add_text(x, y, w, 1.5, f"  {icon}  {title}",
             size=size_title, bold=True, color=WHITE,
             align=PP_ALIGN.LEFT, vert=MSO_ANCHOR.MIDDLE)
    return y + 1.7  # icerik baslangic y

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

def bullet_list(x, y, w, items, size=12, color_dot=PURPLE, color_text=DARK_TXT,
                line_gap=0.6):
    """Renkli yuvarlak madde + metin."""
    cur_y = y
    for item in items:
        # bullet daire
        add_oval(x, cur_y + 0.18, 0.35, 0.35, color_dot)
        add_text(x + 0.55, cur_y, w - 0.55, 0.75, item, size=size, color=color_text,
                 vert=MSO_ANCHOR.TOP)
        cur_y += line_gap
    return cur_y

# ==================== BASLIKLAR ====================
for sh in list(slide.shapes):
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text.strip()
    if "PROJE BAŞLIĞI" in t:
        sh._element.getparent().remove(sh._element)
        add_text(5.5, 10.7, 59, 4.0, "SignBridge", size=88, bold=True,
                 color=PURPLE, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
        add_text(5.5, 14.9, 59, 1.5,
                 "Türk İşaret Dili için Çift Yönlü Çeviri Sistemi",
                 size=24, bold=True, color=DARK_TXT,
                 align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
        # catchphrase banner
        add_rect(20, 16.6, 30, 1.6, PINK, rounding=0.4)
        add_text(20, 16.6, 30, 1.6, "✨  Engelleri Aşan İletişim Köprüsü  ✨",
                 size=15, bold=True, color=WHITE,
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

# ==================== ANA İÇERİK ARKA PLAN ====================
# Yumusak mor gradient yapilamiyor ama acik mor bant ile kapla
add_rect(0, 20.2, 70, 79.8, PURPLE_VL, rounding=0.0)
# Dekoratif soft daireler (kose pattern)
add_oval(-8, 30, 16, 16, RGBColor(0xD4, 0xCE, 0xFF))
add_oval(62, 70, 18, 18, RGBColor(0xC7, 0xE9, 0xFA))
add_oval(-10, 90, 14, 14, RGBColor(0xF8, 0xCD, 0xE6))

# ==================== ŞERİT 1 (y=21-37): ÖZET (sol 24cm) + HEDEFLER (orta 18cm) + MİMARİ (sag 19cm) ====================
y = 20.8
band_h = 16.0

# Sol: ÖZET
ozet_x = 2.5; ozet_w = 24
y2 = card(ozet_x, y, ozet_w, band_h, "ÖZET", PURPLE, CARD_BG, icon="📋", size_title=18)
add_text(ozet_x + 0.5, y2 + 0.2, ozet_w - 1, band_h - 2,
         "Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmasına rağmen "
         "Türk İşaret Dili (TİD) için son kullanıcıya yönelik çeviri çözümleri "
         "yok denecek kadar azdır. \n\n"
         "Bu çalışmada TİD ile Türkçe arasında ÇİFT YÖNLÜ çeviri yapan bütünleşik "
         "bir sistem geliştirilmiştir. Sistem üç bileşenden oluşur:\n\n"
         "(1) MediaPipe Holistic ile gerçek zamanlı işaret noktası çıkarımı,\n"
         "(2) 122 kelimelik GRU sinir ağı ile dizi sınıflandırma,\n"
         "(3) Akıllı Türkçe NLP motoru ile dilbilgisi açısından doğru cümle üretimi.\n\n"
         "Tersine yönde kullanıcının yazdığı/söylediği Türkçe cümle 3B avatar "
         "üzerinde işaret dili olarak gösterilmektedir.\n\n"
         "Geliştirilen model 122 sınıf üzerinde "
         "%99.88 doğrulama doğruluğuna ulaşmış; sistem hem mobil (Flutter) hem "
         "web (Flask + Three.js) üzerinde eş zamanlı çalışmaktadır.",
         size=13, color=DARK_TXT, align=PP_ALIGN.JUSTIFY)

# Orta: HEDEFLER (numaralı)
hed_x = 27.5; hed_w = 17
y2 = card(hed_x, y, hed_w, band_h, "HEDEFLER", PINK, CARD_BG, icon="🎯", size_title=18)
goals = [
    ("TİD → Türkçe",
     "Kameradan canlı işaret\nokunup gramatik Türkçe\ncümleler üretilmesi", PURPLE),
    ("Türkçe → TİD",
     "Yazılı/sözlü Türkçe metnin\n3B avatar üzerinde işaret\ndiline çevrilmesi", CYAN),
    ("Çift Platform",
     "Mobil (Flutter) ve Web\n(Three.js) üzerinde eş\nzamanlı çalışma", GREEN),
    ("Düşük Gecikme",
     "Gerçek zamanlı kullanım\niçin < 1.5 saniye tanıma", ORANGE),
]
gy = y2 + 0.2
for i, (head, body, color) in enumerate(goals):
    add_oval(hed_x + 0.5, gy, 1.4, 1.4, color)
    add_text(hed_x + 0.5, gy, 1.4, 1.4, str(i+1),
             size=18, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
    add_text(hed_x + 2.1, gy - 0.05, hed_w - 2.5, 0.9, head,
             size=13, bold=True, color=color)
    add_text(hed_x + 2.1, gy + 0.85, hed_w - 2.5, 2.5, body,
             size=10, color=DARK_TXT)
    gy += 3.4

# Sag: MİMARİ DİAGRAM
mim_x = 46; mim_w = 21.5
y2 = card(mim_x, y, mim_w, band_h, "SİSTEM MİMARİSİ", CYAN, CARD_BG, icon="🏗️", size_title=18)
add_pic_fit(mim_x + 0.3, y2, mim_w - 0.6, band_h - 2.0,
            ART / "architecture_diagram.png")

# ==================== ŞERİT 2 (y=38-54): YÖNTEM (4 kart) ====================
y = 37.5
band_h = 16.0
# Outer karti (yumusak mor cerceve)
add_rect(2.5, y, 65, band_h, CARD_BG, line_color=BORDER, line_width=1, rounding=0.03)
# Bolum baslik
add_text(3, y + 0.3, 64, 1.5,
         "🧠   YÖNTEM   —   Sistemi Oluşturan Dört Temel Bileşen",
         size=22, bold=True, color=PURPLE,
         vert=MSO_ANCHOR.MIDDLE)
add_rect(3, y + 1.9, 64, 0.1, PURPLE_L, rounding=0)

# 4 kart yan yana
sub_y = y + 2.3
cards_w = (65 - 1) / 4 - 0.4   # ~15.5cm
gap = 0.4
cards_data = [
    ("VERİ SETİ", PURPLE, PURPLE_L, "📊",
     "122 TİD kelimesi\n105.182 örnek\n(augment dahil)",
     [
        "30 frame × 1755 boyut",
        "1629 ham landmark",
        "126 hız bileşeni",
        "6 farklı augmentation",
        "Gürültü · ölçek · ayna",
     ]),
    ("GRU MODELİ", CYAN, CYAN_L, "🧠",
     "2 katmanlı GRU\n~1.9 M parametre\n80 epoch ~20 dakika",
     [
        "Giriş: 1755 boyut",
        "Gizli: 256 · Dropout 0.3",
        "FC(256→128) → ReLU",
        "FC(128→122) sınıf",
        "AdamW · LR=1e-3",
     ]),
    ("AKILLI NLP", PINK, PINK_L, "🌐",
     "Özne tespiti + bağlam\nayrımı + Türkçe ekler\n+ soru çekimleri",
     [
        "BEN/SEN/BERABER/KENDI",
        "KAHVE ↔ KAFE ayrımı",
        "Ablatif (-DAn) eki",
        "'…elim mi?' kohortatif",
        "Otomatik fiil çekimi",
     ]),
    ("3B AVATAR", GREEN, GREEN_L, "🎭",
     "Three.js + Blender\nrain.glb iskelet\n122 işaret pozu",
     [
        "JSON keyframe veritabanı",
        "Bone-based animasyon",
        "60 FPS rendering",
        "Web + Mobil WebView",
        "Mikrofonla sesli giriş",
     ]),
]
for i, (head, color, bg, icon, summary, bullets) in enumerate(cards_data):
    sx = 3.0 + i * (cards_w + gap)
    # Kart govdesi
    add_rect(sx, sub_y, cards_w, 13.4, bg, rounding=0.05)
    # Renkli ust bar
    add_rect(sx, sub_y, cards_w, 2.2, color, rounding=0.05)
    add_text(sx, sub_y, cards_w, 1.2, icon,
             size=22, color=WHITE, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
    add_text(sx, sub_y + 1.15, cards_w, 1.0, head,
             size=14, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
    # Ozet
    add_text(sx + 0.3, sub_y + 2.4, cards_w - 0.6, 2.7, summary,
             size=11, bold=True, color=color, italic=True,
             align=PP_ALIGN.CENTER)
    # Ayrac
    add_rect(sx + 0.8, sub_y + 5.2, cards_w - 1.6, 0.05, color, rounding=0)
    # Madde liste
    by = sub_y + 5.5
    for b in bullets:
        add_oval(sx + 0.4, by + 0.13, 0.25, 0.25, color)
        add_text(sx + 0.8, by, cards_w - 1.0, 0.6, b,
                 size=10, color=DARK_TXT, vert=MSO_ANCHOR.TOP)
        by += 0.7

# ==================== ŞERİT 3 (y=55-72): TEZ ÇIKTILARI ====================
y = 54.5
band_h = 17.0
add_rect(2.5, y, 65, band_h, CARD_BG, line_color=BORDER, line_width=1, rounding=0.03)
add_text(3, y + 0.3, 64, 1.5,
         "📤   TEZ ÇIKTILARI   —   Uygulamanın Gerçek Hayat Sonuçları",
         size=22, bold=True, color=GREEN,
         vert=MSO_ANCHOR.MIDDLE)
add_rect(3, y + 1.9, 64, 0.1, GREEN_L, rounding=0)

# Sol: 10 DEMO CÜMLESİ TABLOSU
sub_y = y + 2.3
demo_x = 3.0
demo_w = 31
add_text(demo_x, sub_y, demo_w, 0.9,
         "🎬  10 DEMO CÜMLESİ  —  İşaret → Türkçe NLP Çıktıları",
         size=14, bold=True, color=PURPLE)
sub_y += 1.0

# Tablo basligi
add_rect(demo_x, sub_y, demo_w, 0.9, PURPLE, rounding=0)
add_text(demo_x + 0.3, sub_y, 13, 0.9, "  İŞARET DİZİSİ",
         size=10, bold=True, color=WHITE, vert=MSO_ANCHOR.MIDDLE)
add_text(demo_x + 13.5, sub_y, 17, 0.9, "TÜRKÇE ÇIKTI",
         size=10, bold=True, color=WHITE, vert=MSO_ANCHOR.MIDDLE)

demo_pairs = [
    ("MERHABA + NASILSIN",          "Merhaba, nasılsın?"),
    ("BEN + IYI + SEN + NE + YAPMAK","İyiyim sen ne yapıyorsun?"),
    ("BEN + COK + CALISMAK + YORULMAK","Ben çalışıyorum, yoruldum."),
    ("SAAT + KAC + BITMEK",         "Saat kaçta bitiyor?"),
    ("AKSAM + BES + SONRA + BOS",   "Akşam beşten sonra boşum."),
    ("BERABER + KAHVE + ICMEK",     "Beraber kahve içelim mi?"),
    ("TAMAM + NEREDE + BULUSMAK",   "Tamam nerede buluşalım?"),
    ("PARK + YAN + KAFE",           "Parkın yanındaki kafede."),
    ("TAMAM + GORUSMEK",            "Tamam görüşürüz."),
    ("KENDI + IYI + BAKMAK",        "Kendine iyi bak."),
]
row_y = sub_y + 0.9
row_h = 0.85
for i, (k, v) in enumerate(demo_pairs):
    bg = PURPLE_L if i % 2 == 0 else CARD_BG
    add_rect(demo_x, row_y, demo_w, row_h, bg, rounding=0, line_color=BORDER, line_width=0.5)
    add_text(demo_x + 0.3, row_y, 13, row_h, k,
             size=9, color=DARK_2, font="Consolas", vert=MSO_ANCHOR.MIDDLE)
    add_text(demo_x + 13.5, row_y, 17, row_h, v,
             size=11, bold=True, color=PURPLE_D, vert=MSO_ANCHOR.MIDDLE)
    row_y += row_h

# Tabloya alt aciklama
add_text(demo_x, row_y + 0.2, demo_w, 0.8,
         "→ Tüm 10 cümle için %100 başarı ile çevirim sağlandı (demo modu).",
         size=10, italic=True, color=GREEN, bold=True)

# Sag: 2 hero ekran goruntusu - mobil kamera + mobil avatar
img_x = 36.5
img_w = 31
img_y = sub_y - 0.2

# Üst: kamera ekran goruntusu
upper_h = 7.5
add_rect(img_x, img_y, img_w, upper_h, DARK_2, rounding=0.04)
add_text(img_x + 0.3, img_y + 0.2, img_w - 0.6, 0.8,
         "📷  TİD → Türkçe (Mobil Kamera)",
         size=13, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
# 2 telefon
ph_h = upper_h - 1.2
ph_w = (img_w - 1.0) / 2
add_pic_fit(img_x + 0.3, img_y + 1.1, ph_w, ph_h - 0.1,
            IMG / "mobil_kamera_merhaba.jpeg")
add_text(img_x + 0.3, img_y + upper_h - 0.6, ph_w, 0.5,
         "MERHABA  %99",
         size=10, bold=True, color=CYAN, align=PP_ALIGN.CENTER)
add_pic_fit(img_x + 0.7 + ph_w, img_y + 1.1, ph_w, ph_h - 0.1,
            IMG / "mobil_kamera_nasılsın.jpeg")
add_text(img_x + 0.7 + ph_w, img_y + upper_h - 0.6, ph_w, 0.5,
         "NASILSIN  %99 → 'Merhaba, nasılsın?'",
         size=9, bold=True, color=CYAN, align=PP_ALIGN.CENTER)

# Alt: web kamera ekran goruntusu (genis aspect)
img_y2 = img_y + upper_h + 0.4
lower_h = band_h - upper_h - 3.5
add_rect(img_x, img_y2, img_w, lower_h, DARK_2, rounding=0.04)
add_text(img_x + 0.3, img_y2 + 0.2, img_w - 0.6, 0.8,
         "💻  Web Arayüzü — Çift Kamera ve NLP Çıktısı",
         size=13, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(img_x + 0.3, img_y2 + 1.1, img_w - 0.6, lower_h - 1.3,
            IMG / "web_kamera.jpeg")

# ==================== ŞERİT 4 (y=72-87): SONUÇLAR + GRAFIKLER ====================
y = 72.0
band_h = 14.5
add_rect(2.5, y, 65, band_h, CARD_BG, line_color=BORDER, line_width=1, rounding=0.03)
add_text(3, y + 0.3, 64, 1.5,
         "📈   SONUÇLAR   —   Model Performansı ve Değerlendirme",
         size=22, bold=True, color=ORANGE,
         vert=MSO_ANCHOR.MIDDLE)
add_rect(3, y + 1.9, 64, 0.1, ORANGE_L, rounding=0)

sub_y = y + 2.3
# Sol: %99.97 hero stat
big_x = 3.0; big_w = 14
add_rect(big_x, sub_y, big_w, 5.0, PURPLE)
add_text(big_x, sub_y + 0.2, big_w, 2.8, "%99.97",
         size=54, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_text(big_x, sub_y + 3.0, big_w, 1.7, "GENEL DOĞRULUK\n3.660 test örneği",
         size=11, bold=True, color=RGBColor(0xCB, 0xD5, 0xE1),
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# Metrik tablo
tbl_y = sub_y + 5.5
add_text(big_x, tbl_y - 0.6, big_w, 0.6, "PERFORMANS METRİKLERİ",
         size=11, bold=True, color=GREY_TXT)
stats = [
    ("Doğrulama (eğitim)",   "%99.88"),
    ("Eğitim doğruluğu",     "%100.0"),
    ("Sınıf sayısı",         "122"),
    ("Çıkarım (GPU)",        "~50 ms"),
    ("Çıkarım (CPU)",        "~80 ms"),
    ("Mobil canlı tanıma",   "~250 ms"),
    ("Eğitim süresi",        "20 dakika"),
]
row_h = 0.65
for i, (k, v) in enumerate(stats):
    bg = PURPLE_L if i % 2 == 0 else None
    if bg:
        add_rect(big_x - 0.1, tbl_y + i * row_h, big_w + 0.2, row_h, bg, rounding=0.0)
    add_text(big_x + 0.2, tbl_y + i * row_h, big_w * 0.62, row_h, k,
             size=10, color=DARK_TXT, vert=MSO_ANCHOR.MIDDLE)
    add_text(big_x + big_w * 0.55, tbl_y + i * row_h, big_w * 0.43, row_h, v,
             size=11, bold=True, color=PURPLE,
             align=PP_ALIGN.RIGHT, vert=MSO_ANCHOR.MIDDLE)

# Sağ: 2 grafik
gx = big_x + big_w + 0.8
gw_total = 65 - (gx - 2.5) - 0.8
gw_each = (gw_total - 0.5) / 2
gh = band_h - 3.3

add_text(gx, sub_y, gw_each, 0.7, "Eğitim Süreci  (80 epoch)",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx, sub_y + 0.8, gw_each, gh - 0.8,
            ART / "training_history_122.png")

add_text(gx + gw_each + 0.5, sub_y, gw_each, 0.7,
         "Karışıklık Matrisi  (en sık 30 kelime)",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx + gw_each + 0.5, sub_y + 0.8, gw_each, gh - 0.8,
            ART / "confusion_matrix_122.png")

# ==================== ŞERİT 5 (y=87-98): SEN DE DENE + AVATAR + WEB ====================
y = 87.0
band_h = 10.5
add_rect(2.5, y, 65, band_h, DARK_2, rounding=0.03)

# Sol: davet (15cm)
add_text(3.3, y + 0.4, 17, 1.8, "🎮  SEN DE DENE!",
         size=22, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_text(3.3, y + 2.1, 17, 8.0,
         "Avatar 122 farklı Türkçe\nişaret kelimesini gerçek\nzamanlı oynatabilir.\n\n"
         "Yanımdaki bilgisayardan\nyazıyla veya mikrofonla\ndeneyebilirsiniz!\n\n"
         "→ Türkçe konuş\n→ Avatar işaret diline çevirsin",
         size=12, color=RGBColor(0xCB, 0xD5, 0xE1))

# Sag: 4 telefon mockup
av_imgs = [
    (IMG / "mobil_avatar_merhaba.jpeg",       "MERHABA",       PURPLE),
    (IMG / "mobil_avatar_ataturk.jpeg",       "ATATÜRK",       CYAN),
    (IMG / "mobil_avatar_elhamdulıllah.jpeg", "ELHAMDÜLİLLAH", PINK),
    (IMG / "mobil_avatar_hoscakal.jpeg",      "HOŞÇA KAL",     GREEN),
]
av_x_start = 21
av_total_w = 46
av_each_w = (av_total_w - 3 * 0.4) / 4
av_h = 9.5

for i, (path, label, color) in enumerate(av_imgs):
    x = av_x_start + i * (av_each_w + 0.4)
    img_y = y + 0.5
    # Beyaz mockup
    add_rect(x, img_y, av_each_w, av_h, CARD_BG, rounding=0.04)
    # mobil ss aspect korunarak
    add_pic_fit(x + 0.2, img_y + 0.2, av_each_w - 0.4, av_h - 1.2, path)
    # Renkli label bant
    add_rect(x + 0.6, img_y + av_h - 1.0, av_each_w - 1.2, 0.85, color, rounding=0.5)
    add_text(x, img_y + av_h - 1.0, av_each_w, 0.85, label,
             size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# ==================== FOOTER ====================
add_text(2.5, 98.0, 65, 1.5,
         "SignBridge © 2026  ·  Esma Sıla ŞAHİNCİ  ·  Nevşehir Hacı Bektaş Veli Üniversitesi  ·  Bilgisayar Mühendisliği Bitirme Projesi",
         size=11, color=GREY_TXT, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

prs.save(str(OUT_PATH))
print(f"\n✅ Poster v5 kaydedildi: {OUT_PATH}")
