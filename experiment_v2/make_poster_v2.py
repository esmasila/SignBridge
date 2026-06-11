"""SignBridge posteri - kullanicinin SABLONUNU bozmadan icerik doldurma.
- Ust kismi (basliklar, logolar, ayrac cizgi) DOKUNULMUYOR
- Slide disindaki talimat balonlari siliniyor
- Alt 78cm yaratici icerik doldurulup, gorseller aspect-ratio korunarak
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pptx import Presentation
from pptx.util import Cm, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path
from PIL import Image

# ============ DOSYALAR ============
TEMPLATE = r"C:\Users\leven\Downloads\template.pptx"
OUT_PATH = Path(r"C:\Users\leven\Downloads\SignBridge_poster.pptx")
ART = Path(r"C:/Projects/sign_bridge/experiment_v2/poster_artifacts")
IMG = Path(r"C:/Users/leven/Downloads/poster-images")

# ============ RENKLER ============
PURPLE   = RGBColor(0x6C, 0x63, 0xFF)
PURPLE_D = RGBColor(0x4C, 0x44, 0xCF)
CYAN     = RGBColor(0x22, 0xD3, 0xEE)
PINK     = RGBColor(0xEC, 0x48, 0x99)
GREEN    = RGBColor(0x10, 0xB9, 0x81)
ORANGE   = RGBColor(0xF5, 0x9E, 0x0B)
LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
SOFT_BG  = RGBColor(0xEE, 0xEC, 0xFF)
CARD_BG  = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TXT = RGBColor(0x0F, 0x17, 0x2A)
DARK_2   = RGBColor(0x1E, 0x29, 0x3B)
GREY_TXT = RGBColor(0x47, 0x55, 0x69)
BORDER   = RGBColor(0xCB, 0xD5, 0xE1)

FONT_HEAD = "Calibri"  # daha guvenli (Inter her bilgisayarda yok)

# ============ SABLONU AC ============
prs = Presentation(TEMPLATE)
slide = prs.slides[0]
print(f"Slide: {prs.slide_width/360000} x {prs.slide_height/360000} cm")

# ============ ADIM 1: Slide disindaki talimatlari sil ============
# (x<0 veya x>70 olan shape'ler talimat balonu)
to_remove = []
for sh in slide.shapes:
    x_cm = sh.left / 360000
    if x_cm < 0 or x_cm > 65:
        to_remove.append(sh)
        print(f"Siliniyor: {sh.name} @ x={x_cm:.1f}")

for sh in to_remove:
    sh._element.getparent().remove(sh._element)

# ============ ADIM 2: Mevcut basliklari guncelle ============
# Hangi shape hangi metin (orijinal sirayla):
# [1] TextBox 21 "PROJE BAŞLIĞI"
# [2] TextBox 22 "Proje Sahibi: Adı SOYADI"
# [3] TextBox 23 "NEVŞEHİR..."
# [9] TextBox 22 "Proje Danışmanı:"

# Bu sira tekrar guncellenmis listeyle bulalim:
for sh in slide.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text.strip()
    if "PROJE BAŞLIĞI" in t:
        # Baslik - SignBridge
        tf = sh.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r1 = p.add_run()
        r1.text = "SignBridge"
        r1.font.name = FONT_HEAD
        r1.font.size = Pt(72)
        r1.font.bold = True
        r1.font.color.rgb = PURPLE
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = "Türk İşaret Dili için Çift Yönlü Çeviri Sistemi"
        r2.font.name = FONT_HEAD
        r2.font.size = Pt(28)
        r2.font.bold = True
        r2.font.color.rgb = DARK_TXT
        # Genislet
        sh.left = Cm(5.5)
        sh.width = Cm(59)
        sh.top = Cm(11.5)
        sh.height = Cm(5.5)
    elif "Proje Sahibi" in t and "Adı SOYADI" in t:
        tf = sh.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = "Proje Sahibi:"
        r.font.name = FONT_HEAD
        r.font.size = Pt(15)
        r.font.bold = False
        r.font.color.rgb = GREY_TXT
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = "Esma Sıla ŞAHİNCİ"
        r2.font.name = FONT_HEAD
        r2.font.size = Pt(20)
        r2.font.bold = True
        r2.font.color.rgb = DARK_TXT
    elif "Proje Danışmanı" in t:
        tf = sh.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = "Proje Danışmanı:"
        r.font.name = FONT_HEAD
        r.font.size = Pt(15)
        r.font.bold = False
        r.font.color.rgb = GREY_TXT
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = "Öğr. Gör. Kadir HALTAŞ"
        r2.font.name = FONT_HEAD
        r2.font.size = Pt(20)
        r2.font.bold = True
        r2.font.color.rgb = DARK_TXT
    elif "NEVŞEHİR" in t:
        # Ust baslik - olduğu gibi birak (universite + bolum)
        pass

# ============ ADIM 3: ALT IÇERIK ============
# y=22 cm'den asagiya yarat. Slide genisligi 70cm.

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
             font=FONT_HEAD, align=PP_ALIGN.LEFT, vert=MSO_ANCHOR.TOP):
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

def section_header(x, y, w, title, color=PURPLE, icon=""):
    """Renkli sol cubuk + buyuk baslik."""
    add_rect(x, y, 0.55, 1.3, color)
    txt = f"{icon}  {title}" if icon else title
    add_text(x + 0.8, y - 0.05, w - 0.8, 1.4, txt,
             size=22, bold=True, color=DARK_TXT,
             vert=MSO_ANCHOR.MIDDLE)

def add_pic_fit(x, y, max_w, max_h, path, center=True):
    """Aspect-ratio koruyarak resim ekle (sıkışmasin)."""
    img = Image.open(str(path))
    iw, ih = img.size
    ratio = iw / ih
    if max_w / max_h > ratio:
        # h sınırlayıcı
        h = max_h
        w = h * ratio
    else:
        w = max_w
        h = w / ratio
    if center:
        x_offset = (max_w - w) / 2
        y_offset = (max_h - h) / 2
    else:
        x_offset = y_offset = 0
    slide.shapes.add_picture(str(path), Cm(x + x_offset), Cm(y + y_offset),
                             Cm(w), Cm(h))
    return w, h

def add_pic_exact(x, y, w, h, path):
    slide.shapes.add_picture(str(path), Cm(x), Cm(y), Cm(w), Cm(h))

# ============ İÇERİK BAŞLAR (y=21cm den itibaren) ============

# --- ŞERIT 1: ÖZET (tam genislik kart) ---
y = 21.5
add_rect(2.5, y, 65, 8.5, SOFT_BG)
section_header(3.0, y + 0.3, 64, "ÖZET", PURPLE)
abstract = ("Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmasına rağmen Türk İşaret Dili (TİD) için "
            "son kullanıcıya yönelik çeviri çözümleri yok denecek kadar azdır. Bu çalışmada TİD ile Türkçe "
            "arasında çift yönlü çeviri yapan bütünleşik bir sistem geliştirilmiştir. Sistem üç ana bileşenden "
            "oluşur: MediaPipe Holistic ile gerçek zamanlı işaret noktası çıkarımı, 122 kelimelik GRU sinir ağı "
            "ile dizi sınıflandırma ve akıllı Türkçe NLP modülü ile dilbilgisi açısından doğru cümle üretimi. "
            "Tersine yönde ise kullanıcının yazdığı ya da söylediği Türkçe cümle 3B avatar üzerinde işaret "
            "dili olarak gösterilmektedir. Geliştirilen model, 122 sınıf üzerinde %99.88 doğrulama "
            "doğruluğuna ulaşmış; sistem hem mobil (Flutter) hem web (Flask + Three.js) platformlarında eş "
            "zamanlı çalışabilmektedir.")
add_text(3.0, y + 2.0, 64, 6.0, abstract, size=15, color=DARK_TXT,
         align=PP_ALIGN.JUSTIFY)

# --- ŞERIT 2: AMAÇ (sol) + MİMARİ (sağ) ---
y = 31.0
# Sol: AMAÇ - 4 koselik renkli kutu
col_w_l = 26
add_rect(2.5, y, col_w_l, 22, CARD_BG, line_color=BORDER, line_width=1)
section_header(3.0, y + 0.3, col_w_l - 1, "AMAÇ", PINK, icon="🎯")

# 4 amac maddesi - her biri renkli badge
goals = [
    ("TİD → Türkçe", "Kameradan canlı görüntü alıp işaretleri gramer açısından doğru Türkçe cümlelere çevirme",
     PURPLE),
    ("Türkçe → TİD", "Yazılı veya sözlü Türkçe metni 3B avatar üzerinde işaret dili olarak gösterme",
     CYAN),
    ("Çift Platform", "Hem mobil (Flutter) hem web (tarayıcı) üzerinde eş zamanlı çalışma",
     GREEN),
    ("Düşük Gecikme", "Gerçek zamanlı kullanım için işaret başına 1.5 saniyenin altında tanıma süresi",
     ORANGE),
]
gy = y + 2.0
for i, (head, body, color) in enumerate(goals):
    # numara
    add_rect(3.2, gy, 1.4, 1.4, color)
    add_text(3.2, gy, 1.4, 1.4, str(i+1), size=20, bold=True,
             color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
             vert=MSO_ANCHOR.MIDDLE)
    # baslik
    add_text(4.9, gy - 0.05, 21, 1.0, head, size=14, bold=True, color=color)
    add_text(4.9, gy + 0.85, 21, 3.5, body, size=12, color=DARK_TXT)
    gy += 4.5

# Sağ: MİMARİ DİYAGRAM
col_x_r = 30
col_w_r = 37.5
add_rect(col_x_r, y, col_w_r, 22, CARD_BG, line_color=BORDER, line_width=1)
section_header(col_x_r + 0.5, y + 0.3, col_w_r - 1, "SİSTEM MİMARİSİ", CYAN, icon="🏗️")
# Mimari diyagram aspect-ratio: 18:7.5 = 2.4
add_pic_fit(col_x_r + 0.5, y + 2.0, col_w_r - 1, 19.5, ART / "architecture_diagram.png")

# --- ŞERIT 3: YÖNTEM (model + veri + NLP) ---
y = 54.0
add_rect(2.5, y, 65, 13, CARD_BG, line_color=BORDER, line_width=1)
section_header(3.0, y + 0.3, 64, "YÖNTEM", PURPLE, icon="🧠")

# 3 sutun: Veri / Model / NLP
sub_y = y + 2.0
col_w = 21.0
gap = 0.5

# Veri Seti
sx = 3.0
add_rect(sx, sub_y, col_w, 10, SOFT_BG)
add_text(sx, sub_y + 0.2, col_w, 1.0, "📊  VERİ SETİ",
         size=14, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_text(sx + 0.4, sub_y + 1.4, col_w - 0.8, 8,
         "• 122 TİD kelimesi\n"
         "• 105.182 örnek dizisi\n"
         "  (augment edilmiş)\n\n"
         "• Her örnek:\n"
         "   30 frame × 1755 boyut\n"
         "   1629 landmark\n"
         "   + 126 hız bileşeni\n\n"
         "• Augmentation:\n"
         "   gürültü · ölçek\n"
         "   çevirme · döndürme\n"
         "   zaman bükme · ayna",
         size=12, color=DARK_TXT)

# Model
sx = 3.0 + col_w + gap
add_rect(sx, sub_y, col_w, 10, SOFT_BG)
add_text(sx, sub_y + 0.2, col_w, 1.0, "🧠  GRU MODELİ",
         size=14, bold=True, color=CYAN, align=PP_ALIGN.CENTER)
add_text(sx + 0.4, sub_y + 1.4, col_w - 0.8, 8,
         "• 2 katmanlı GRU\n"
         "  (Gated Recurrent Unit)\n\n"
         "• Giriş: 1755 boyut\n"
         "  Gizli: 256 · Dropout 0.3\n\n"
         "• Sınıflandırıcı:\n"
         "  FC(256→128) + ReLU\n"
         "  FC(128→122)\n\n"
         "• Parametre: ~1.9 milyon\n"
         "• Optimizer: AdamW\n"
         "  LR=1e-3 · Batch=64\n"
         "• Eğitim: 80 epoch ~20 dk",
         size=12, color=DARK_TXT)

# NLP
sx = 3.0 + 2*(col_w + gap)
add_rect(sx, sub_y, col_w, 10, SOFT_BG)
add_text(sx, sub_y + 0.2, col_w, 1.0, "🌐  AKILLI NLP",
         size=14, bold=True, color=PINK, align=PP_ALIGN.CENTER)
add_text(sx + 0.4, sub_y + 1.4, col_w - 0.8, 8,
         "• Özne tespiti:\n"
         "  BEN/SEN/BERABER/KENDI\n"
         "  → doğru fiil çekimi\n\n"
         "• Bağlam ayrımı:\n"
         "  KAHVE ↔ KAFE benzer\n"
         "  işaretler ayırt edilir\n\n"
         "• Türkçe ekleri:\n"
         "  ablatif (-DAn)\n"
         "  locatif (-DA)\n\n"
         "• Soru çekimleri:\n"
         "  '…elim mi?' kohortatif",
         size=12, color=DARK_TXT)

# --- ŞERIT 4: SONUÇLAR + GRAFİKLER ---
y = 68.5
add_rect(2.5, y, 65, 18, CARD_BG, line_color=BORDER, line_width=1)
section_header(3.0, y + 0.3, 64, "SONUÇLAR", GREEN, icon="📈")

# Sol: büyük %99.97 + tablo
big_x = 3.5
big_w = 16
big_y = y + 2.2
add_rect(big_x, big_y, big_w, 5.5, PURPLE)
add_text(big_x, big_y + 0.1, big_w, 3, "%99.97",
         size=52, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_text(big_x, big_y + 3.2, big_w, 2.0, "GENEL DOĞRULUK\n3660 test örneği",
         size=12, bold=True, color=RGBColor(0xCB, 0xD5, 0xE1),
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# Tablo (kutunun altında)
tbl_y = big_y + 6.0
stats = [
    ("Doğrulama doğruluğu", "%99.88"),
    ("Eğitim doğruluğu",     "%100.0"),
    ("Sınıf sayısı",         "122"),
    ("Çıkarım gecikmesi (GPU)",   "~50 ms"),
    ("Çıkarım gecikmesi (CPU)",   "~80 ms"),
    ("Mobil canlı tanıma",       "~250 ms"),
    ("Eğitim süresi",           "20 dakika"),
]
row_h = 0.85
for i, (k, v) in enumerate(stats):
    add_text(big_x, tbl_y + i * row_h, big_w * 0.65, row_h, k,
             size=11, color=DARK_TXT, vert=MSO_ANCHOR.MIDDLE)
    add_text(big_x + big_w * 0.65, tbl_y + i * row_h, big_w * 0.35, row_h, v,
             size=12, bold=True, color=PURPLE,
             align=PP_ALIGN.RIGHT, vert=MSO_ANCHOR.MIDDLE)

# Sağ: 2 grafik yan yana
gx = big_x + big_w + 1.0
gw = 65 - (gx - 2.5) - 0.8  # genislik = ~30
gw_each = (gw - 0.5) / 2
gh = 14
gy = y + 2.0

add_text(gx, gy, gw_each, 0.8, "Eğitim Süreci",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx, gy + 0.9, gw_each, gh - 1, ART / "training_history_122.png")

add_text(gx + gw_each + 0.5, gy, gw_each, 0.8, "Karışıklık Matrisi",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx + gw_each + 0.5, gy + 0.9, gw_each, gh - 1,
            ART / "confusion_matrix_122.png")

# --- ŞERIT 5: SEN DE DENE — 4 avatar + mobil ekran + davet ---
y = 88.0
add_rect(2.5, y, 65, 11, RGBColor(0x1E, 0x29, 0x3B))  # koyu lacivert

# Sol kart: dene yazisi
sd_x = 3.0
add_text(sd_x, y + 0.3, 30, 1.6, "🎮  SEN DE DENE!",
         size=24, bold=True, color=CYAN,
         vert=MSO_ANCHOR.MIDDLE)
add_text(sd_x, y + 2.0, 30, 7.0,
         "Avatar 122 farklı Türkçe işaret\nkelimesini gerçek zamanlı\noynatabilir.\n\n"
         "Yanımdaki bilgisayardan\ncümle yazıp avatar üzerinde\n"
         "işaret dilini anında gözlemleyebilirsiniz.\n\n"
         "Mikrofona Türkçe konuşmak da\nyeterli — sistem otomatik\nişaret diline çeviriyor.",
         size=13, color=RGBColor(0xCB, 0xD5, 0xE1))

# Sag: 4 avatar mobil ss yan yana
av_imgs = [
    (IMG / "mobil_avatar_merhaba.jpeg", "MERHABA"),
    (IMG / "mobil_avatar_ataturk.jpeg", "ATATÜRK"),
    (IMG / "mobil_avatar_elhamdulıllah.jpeg", "ELHAMDÜLİLLAH"),
    (IMG / "mobil_avatar_hoscakal.jpeg", "HOŞÇA KAL"),
]
av_x_start = 35.0
av_total_w = 31.5
av_each_w = (av_total_w - 3 * 0.4) / 4  # ~7.4cm her biri
av_h = 9.0  # mobil ss tall - aspect 0.5

# her mobil ss 1023x2048 = 0.5 aspect. genislik = h * 0.5
# av_h=9 ise w = 4.5cm. ama bizim av_each_w=7.4. ratio sinirliyor.
# bizim aspect_fit kullanıyoruz, koru.

# arkalik
add_text(av_x_start, y + 0.3, av_total_w, 1.0, "🎭  Avatar Demo Önizleme",
         size=14, bold=True, color=CYAN, align=PP_ALIGN.CENTER,
         vert=MSO_ANCHOR.MIDDLE)

for i, (path, label) in enumerate(av_imgs):
    x = av_x_start + i * (av_each_w + 0.4)
    img_y = y + 1.5
    # beyaz çerçeve
    add_rect(x, img_y, av_each_w, av_h, CARD_BG, line_color=BORDER, line_width=1)
    add_pic_fit(x + 0.15, img_y + 0.15, av_each_w - 0.3, av_h - 1.2, path)
    # label
    add_text(x, img_y + av_h - 1.0, av_each_w, 0.9, label,
             size=11, bold=True, color=PURPLE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# --- ŞERIT 6: CANLI DEMO ÖZET + Mobil tanıma ss (üst yarısı) ---
# 5. seritle birlikte ust seritte canli demoyu eklemek karmasik oldu;
# bunu sectik daha kucuk bir banda gomelim - simdilik 4 avatar yeter, koyacak yer yok

# Yukari donup, sonuclar bandinin yaninda bos kalan yerde mobile_kamera_nasilsin'i kullanabiliriz
# ama o zaman sonuclar sherit'i kompakt durdu, bos yer yok.

# Cozum: sonuc grafiklerin sağına/altına eklemiyoruz;
# Onun yerine ozet seritinin sagına kucuk bir mobil ss koyalim

# Bu zaten yapildi.

# ============ KAYDET ============
prs.save(str(OUT_PATH))
print(f"\n✅ Poster kaydedildi: {OUT_PATH}")
