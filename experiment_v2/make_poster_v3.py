"""SignBridge posteri v3 - sablonu koru, daha buyuk yazi, tum kamera ss'leri ekle.
Yaratici tasarim: hero kamera goruntusu, renkli kart sistemi, sen-de-dene banner.
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
LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
SOFT_BG  = RGBColor(0xEE, 0xEC, 0xFF)
CARD_BG  = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TXT = RGBColor(0x0F, 0x17, 0x2A)
DARK_2   = RGBColor(0x1E, 0x29, 0x3B)
GREY_TXT = RGBColor(0x47, 0x55, 0x69)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
BORDER   = RGBColor(0xCB, 0xD5, 0xE1)
ICE      = RGBColor(0xE0, 0xF7, 0xFA)

FONT = "Calibri"

prs = Presentation(TEMPLATE)
slide = prs.slides[0]

# --- ADIM 1: Slide disindaki talimat balonlarini sil ---
to_remove = []
for sh in slide.shapes:
    x_cm = sh.left / 360000
    if x_cm < 0 or x_cm > 65:
        to_remove.append(sh)
for sh in to_remove:
    sh._element.getparent().remove(sh._element)

# --- Yardimcilar ---
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

def section_header(x, y, w, title, color=PURPLE, icon="", size=24):
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

# --- ADIM 2: Mevcut basliklari guncelle ---
for sh in list(slide.shapes):
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text.strip()
    if "PROJE BAŞLIĞI" in t:
        # Bu shape'i sil; yerine yeni temiz textboxlar ekleyeceğiz
        sh._element.getparent().remove(sh._element)
        # SignBridge baslik
        add_text(5.5, 10.5, 59, 4.5, "SignBridge", size=80, bold=True,
                 color=PURPLE, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
        # subtitle (bullet'siz, temiz)
        add_text(5.5, 15.4, 59, 2.0,
                 "Türk İşaret Dili için Çift Yönlü Çeviri Sistemi",
                 size=28, bold=True, color=DARK_TXT,
                 align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
        continue  # for sonraki shape'e devam et
    elif "Proje Sahibi" in t and "Adı SOYADI" in t:
        tf = sh.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
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

# ============ ALT İÇERİK ============
# Slide alani: 70 x 100 cm. Ust ~21cm kullanildi, alt 22..98 arası kullanılabilir.

# ============ ŞERİT 1: ÖZET + HERO MOBIL ============
y = 21.0
ozet_w = 39
add_rect(2.5, y, ozet_w, 11, SOFT_BG, rounding=0.05)
section_header(3.0, y + 0.3, ozet_w - 1, "ÖZET", PURPLE, icon="📝", size=22)

abstract = ("Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmasına rağmen Türk "
            "İşaret Dili için son kullanıcıya yönelik çeviri çözümleri yok denecek "
            "kadar azdır. Bu çalışmada TİD ile Türkçe arasında ÇİFT YÖNLÜ çeviri yapan "
            "bütünleşik bir sistem geliştirilmiştir.\n\n"
            "Sistem üç ana bileşenden oluşur: MediaPipe Holistic ile gerçek zamanlı "
            "işaret noktası çıkarımı, 122 kelimelik GRU sinir ağı ile dizi "
            "sınıflandırma ve akıllı Türkçe NLP modülü ile dilbilgisi doğru cümle "
            "üretimi. Tersine yönde ise kullanıcının yazdığı/söylediği Türkçe cümle "
            "3B avatar üzerinde işaret dili olarak gösterilmektedir.\n\n"
            "Model, 122 sınıf üzerinde %99.88 doğrulama doğruluğuna ulaşmış; sistem "
            "hem mobil (Flutter) hem web (Flask + Three.js) platformlarında eş "
            "zamanlı çalışabilmektedir.")
add_text(3.2, y + 2.0, ozet_w - 0.4, 9.0, abstract, size=15, color=DARK_TXT,
         align=PP_ALIGN.JUSTIFY)

# Sag: HERO MOBIL CANLI TAHMIN
hero_x = 2.5 + ozet_w + 0.8
hero_w = 70 - hero_x - 2.5
add_rect(hero_x, y, hero_w, 11, DARK_2)
add_text(hero_x + 0.3, y + 0.3, hero_w - 0.6, 1.0,
         "🎬  CANLI MOBİL TAHMİN", size=15, bold=True, color=CYAN,
         vert=MSO_ANCHOR.MIDDLE)
# Mobil ss aspect 1023/2048=0.5 (tall). 10cm yuksek koyalim => w=5cm
add_pic_fit(hero_x + 0.3, y + 1.4, hero_w - 0.6, 9.3,
            IMG / "mobil_kamera_nasılsın.jpeg")

# ============ ŞERİT 2: AMAÇ (sol) + MİMARİ (sag) ============
y = 33.0

# AMAÇ
amac_w = 26
add_rect(2.5, y, amac_w, 18, CARD_BG, line_color=BORDER)
section_header(3.0, y + 0.3, amac_w - 1, "AMAÇ", PINK, icon="🎯", size=22)

goals = [
    ("TİD → Türkçe",
     "Kameradan canlı görüntü alıp\nişaretleri gramer açısından doğru\nTürkçe cümlelere çevirme.", PURPLE),
    ("Türkçe → TİD",
     "Yazılı/sözlü Türkçe metni 3B\navatar üzerinde işaret dili\nolarak gösterme.", CYAN),
    ("Çift Platform",
     "Hem mobil (Flutter) hem web\n(tarayıcı) üzerinde eş zamanlı\nçalışma.", GREEN),
    ("Düşük Gecikme",
     "Gerçek zamanlı kullanım için\nişaret başına 1.5 saniyenin\naltında tanıma süresi.", ORANGE),
]
gy = y + 2.2
for i, (head, body, color) in enumerate(goals):
    # numara dairesi
    add_rect(3.2, gy, 1.8, 1.8, color, rounding=0.5)
    add_text(3.2, gy, 1.8, 1.8, str(i+1), size=24, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE,
             font=FONT)
    add_text(5.3, gy, amac_w - 3, 1.0, head, size=16, bold=True, color=color)
    add_text(5.3, gy + 1.1, amac_w - 3, 2.5, body, size=12, color=DARK_TXT)
    gy += 3.8

# MIMARI
mim_x = 30
mim_w = 37.5
add_rect(mim_x, y, mim_w, 18, CARD_BG, line_color=BORDER)
section_header(mim_x + 0.5, y + 0.3, mim_w - 1, "SİSTEM MİMARİSİ", CYAN,
               icon="🏗️", size=22)
add_pic_fit(mim_x + 0.5, y + 2.0, mim_w - 1, 15.5,
            ART / "architecture_diagram.png")

# ============ ŞERİT 3: YÖNTEM (3 kolon ayrıntı) ============
y = 52.5
add_rect(2.5, y, 65, 14, CARD_BG, line_color=BORDER)
section_header(3.0, y + 0.3, 64, "YÖNTEM", PURPLE, icon="🧠", size=22)

sub_y = y + 2.2
col_w = 20.7
gap = 0.6

# Veri Seti kart
sx = 3.0
add_rect(sx, sub_y, col_w, 11, SOFT_BG, rounding=0.06)
add_text(sx, sub_y + 0.25, col_w, 1.0, "📊  VERİ SETİ",
         size=15, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_text(sx + 0.3, sub_y + 1.4, col_w - 0.6, 9.4,
         "•  122 TİD kelimesi\n\n"
         "•  105.182 örnek dizisi\n   (augment edilmiş)\n\n"
         "•  Her örnek:\n   30 frame × 1755 boyut\n   1629 landmark\n   + 126 hız bileşeni\n\n"
         "•  Augmentation:\n   gürültü · ölçek · ayna\n   çevirme · döndürme\n   zaman bükme",
         size=13, color=DARK_TXT)

# Model kart
sx = 3.0 + col_w + gap
add_rect(sx, sub_y, col_w, 11, SOFT_BG, rounding=0.06)
add_text(sx, sub_y + 0.25, col_w, 1.0, "🧠  GRU MODELİ",
         size=15, bold=True, color=CYAN, align=PP_ALIGN.CENTER)
add_text(sx + 0.3, sub_y + 1.4, col_w - 0.6, 9.4,
         "•  2 katmanlı GRU\n   (Gated Recurrent Unit)\n\n"
         "•  Giriş: 1755 boyut\n   Gizli: 256 · Dropout 0.3\n\n"
         "•  Sınıflandırıcı:\n   FC(256→128) + ReLU\n   FC(128→122)\n\n"
         "•  ~1.9 M parametre\n\n"
         "•  Optimizer: AdamW\n   LR=1e-3 · Batch=64\n   Eğitim: 80 epoch ~20 dk",
         size=13, color=DARK_TXT)

# NLP kart
sx = 3.0 + 2*(col_w + gap)
add_rect(sx, sub_y, col_w, 11, SOFT_BG, rounding=0.06)
add_text(sx, sub_y + 0.25, col_w, 1.0, "🌐  AKILLI NLP",
         size=15, bold=True, color=PINK, align=PP_ALIGN.CENTER)
add_text(sx + 0.3, sub_y + 1.4, col_w - 0.6, 9.4,
         "•  Özne tespiti:\n   BEN/SEN/BERABER/KENDI\n   → doğru fiil çekimi\n\n"
         "•  Bağlam ayrımı:\n   KAHVE ↔ KAFE benzer\n   işaretler ayırt edilir\n\n"
         "•  Türkçe ekler:\n   ablatif (-DAn)\n   locatif (-DA)\n\n"
         "•  Kohortatif soru:\n   '…elim mi?' yapısı",
         size=13, color=DARK_TXT)

# ============ ŞERİT 4: SONUÇLAR ============
y = 67.5
add_rect(2.5, y, 65, 16.5, CARD_BG, line_color=BORDER)
section_header(3.0, y + 0.3, 64, "SONUÇLAR", GREEN, icon="📈", size=22)

# Sol: %99.97 buyuk + tablo
big_x = 3.0
big_w = 14
big_y = y + 2.2
add_rect(big_x, big_y, big_w, 5.5, PURPLE)
add_text(big_x, big_y + 0.2, big_w, 3.0, "%99.97",
         size=58, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_text(big_x, big_y + 3.4, big_w, 2.0, "GENEL DOĞRULUK\n3.660 örnekle ölçüm",
         size=12, bold=True, color=RGBColor(0xCB, 0xD5, 0xE1),
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# Performans tablosu
tbl_y = big_y + 6.2
add_text(big_x, tbl_y - 0.6, big_w, 0.7, "PERFORMANS METRİKLERİ",
         size=11, bold=True, color=GREY_TXT)
stats = [
    ("Doğrulama doğruluğu",       "%99.88"),
    ("Eğitim doğruluğu",          "%100.0"),
    ("Sınıf sayısı",              "122"),
    ("Çıkarım (GPU)",             "~50 ms"),
    ("Çıkarım (CPU)",             "~80 ms"),
    ("Mobil canlı",               "~250 ms"),
    ("Eğitim süresi",             "20 dk"),
]
row_h = 0.75
for i, (k, v) in enumerate(stats):
    bg = SOFT_BG if i % 2 == 0 else None
    if bg:
        add_rect(big_x - 0.1, tbl_y + i * row_h, big_w + 0.2, row_h, bg, rounding=0.0)
    add_text(big_x + 0.2, tbl_y + i * row_h, big_w * 0.65, row_h, k,
             size=11, color=DARK_TXT, vert=MSO_ANCHOR.MIDDLE)
    add_text(big_x + big_w * 0.6, tbl_y + i * row_h, big_w * 0.35, row_h, v,
             size=12, bold=True, color=PURPLE,
             align=PP_ALIGN.RIGHT, vert=MSO_ANCHOR.MIDDLE)

# Orta: 2 grafik yan yana
gx = big_x + big_w + 0.8
gy_start = y + 2.2
gw_total = 65 - (gx - 2.5) - 0.8
gw_each = (gw_total - 0.6) / 2
gh = 14

add_text(gx, gy_start, gw_each, 0.7, "Eğitim Süreci  (80 epoch)",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx, gy_start + 0.8, gw_each, gh - 0.8,
            ART / "training_history_122.png")

add_text(gx + gw_each + 0.6, gy_start, gw_each, 0.7,
         "Karışıklık Matrisi  (30 sık kelime)",
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
add_pic_fit(gx + gw_each + 0.6, gy_start + 0.8, gw_each, gh - 0.8,
            ART / "confusion_matrix_122.png")

# ============ ŞERİT 5: WEB DEMO + Zorluklar ============
y = 84.5
add_rect(2.5, y, 65, 4.0, CARD_BG, line_color=BORDER)
# Web kamera ss (genis aspect ~2.2) - sol
add_text(3.0, y + 0.1, 64, 0.7, "💻  WEB ARAYÜZÜ — Çift Kamera (Landmark + Ham)",
         size=13, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_pic_fit(3.0, y + 0.9, 31, 3.0, IMG / "web_kamera.jpeg", center=False)

# Sag: Zorluklar (kompakt)
add_rect(36, y + 0.7, 31.5, 3.1, SOFT_BG, rounding=0.06)
add_text(36.3, y + 0.85, 31, 0.7, "🔧  Karşılaşılan Zorluklar & Çözümler",
         size=12, bold=True, color=PINK)
add_text(36.3, y + 1.5, 31, 2.3,
         "•  Benzer işaretler (KAHVE↔KAFE): per-kelime güven eşiği\n"
         "•  MediaPipe paralel segfault: threading.Lock ile çözüldü\n"
         "•  Mobil kamera donması: zoom stream sonrası ertelendi\n"
         "•  Geçiş gürültüsü: 400ms cooldown + buffer reset",
         size=10, color=DARK_TXT)

# ============ ŞERİT 6: SEN DE DENE — KOYU HERO BANNER ============
# Web kamera serit'ini kucult, sen-de-dene'yi buyut
y = 89.0   # eski 91.3 yerine biraz yukari kaydir
add_rect(2.5, y, 65, 9.0, DARK_2)

# Sol: davet + web avatar ss
left_w = 17
add_text(3.3, y + 0.4, left_w, 1.6, "🎮  SEN DE DENE!",
         size=22, bold=True, color=CYAN, vert=MSO_ANCHOR.MIDDLE)
add_text(3.3, y + 1.8, left_w, 6.8,
         "Avatar 122 farklı Türkçe\nişaret kelimesini gerçek\nzamanlı oynatabilir.\n\n"
         "Yanımdaki bilgisayardan\nkomutla deneyebilirsiniz!",
         size=12, color=RGBColor(0xCB, 0xD5, 0xE1))

# Sag: 4 avatar TELEFON MOCKUP gibi - her biri 10cm x 8cm
av_imgs = [
    (IMG / "mobil_avatar_merhaba.jpeg", "MERHABA"),
    (IMG / "mobil_avatar_ataturk.jpeg", "ATATÜRK"),
    (IMG / "mobil_avatar_elhamdulıllah.jpeg", "ELHAMDÜLİLLAH"),
    (IMG / "mobil_avatar_hoscakal.jpeg", "HOŞÇA KAL"),
]
av_x_start = 21.0
av_total_w = 46
av_each_w = (av_total_w - 3 * 0.4) / 4  # ~11.2cm her
av_h = 8.0   # daha yuksek - mobil 0.5 aspect = w=4cm yuksek 8cm

for i, (path, label) in enumerate(av_imgs):
    x = av_x_start + i * (av_each_w + 0.4)
    img_y = y + 0.5
    # beyaz mockup arka plan
    add_rect(x, img_y, av_each_w, av_h, CARD_BG, rounding=0.06,
             line_color=BORDER)
    # mobil ss - aspect korunarak ortalanmis
    add_pic_fit(x + 0.2, img_y + 0.2, av_each_w - 0.4, av_h - 1.3, path)
    # label - renkli badge gibi
    add_rect(x + 1.0, img_y + av_h - 1.0, av_each_w - 2.0, 0.85, PURPLE,
             rounding=0.4)
    add_text(x, img_y + av_h - 1.0, av_each_w, 0.85, label,
             size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# ============ FOOTER ============
add_text(2.5, 98.5, 65, 1.2,
         "SignBridge © 2026  ·  Nevşehir Hacı Bektaş Veli Üniversitesi  ·  Bilgisayar Mühendisliği Bitirme Projesi",
         size=10, color=GREY_TXT, align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

prs.save(str(OUT_PATH))
print(f"\n✅ Poster v3 kaydedildi: {OUT_PATH}")
