"""SignBridge bitirme projesi posteri — 70x100cm dikey, Turkce."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pptx import Presentation
from pptx.util import Cm, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path

# ============ AYARLAR ============
W_CM, H_CM = 70, 100  # poster boyutu (dikey)
OUT_PATH = Path(r"C:\Users\leven\Downloads\SignBridge_poster.pptx")
ART = Path(r"C:/Projects/sign_bridge/experiment_v2/poster_artifacts")
IMG = Path(r"C:/Users/leven/Downloads/poster-images")

# Renkler (Soft Midnight tema)
PURPLE   = RGBColor(0x6C, 0x63, 0xFF)
CYAN     = RGBColor(0x22, 0xD3, 0xEE)
PINK     = RGBColor(0xEC, 0x48, 0x99)
GREEN    = RGBColor(0x10, 0xB9, 0x81)
LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
CARD_BG  = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TXT = RGBColor(0x0F, 0x17, 0x2A)
GREY_TXT = RGBColor(0x47, 0x55, 0x69)
ACCENT_BG = RGBColor(0xE0, 0xE7, 0xFF)

# Inter font (poster icin)
FONT_HEAD = "Inter"
FONT_BODY = "Calibri"  # daha yaygin

# ============ POSTER OLUSTUR ============
prs = Presentation()
prs.slide_width  = Cm(W_CM)
prs.slide_height = Cm(H_CM)
blank = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank)

# --- Helper: BG renk doldur ---
def fill_bg(shape, color):
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = color

def no_line(shape):
    shape.line.fill.background()

def add_rect(x, y, w, h, color=None, fill=True, line_color=None):
    """x,y,w,h Cm cinsinden."""
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Cm(x), Cm(y), Cm(w), Cm(h))
    sh.adjustments[0] = 0.05  # az koselik
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = color
    else:
        sh.fill.background()
    if line_color:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(2)
    else:
        sh.line.fill.background()
    return sh

def add_text(x, y, w, h, text, size=18, bold=False, color=DARK_TXT,
             font=FONT_BODY, align=PP_ALIGN.LEFT, vert=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Cm(0.2)
    tf.margin_top = tf.margin_bottom = Cm(0.1)
    tf.vertical_anchor = vert
    # text birden cok paragraf olabilir
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb

def add_section_title(x, y, w, text, color=PURPLE):
    """Bolum baslik (renkli cubuk + buyuk baslik)."""
    add_rect(x, y, 0.6, 1.5, color)
    add_text(x + 0.8, y, w - 0.8, 1.5, text, size=32, bold=True,
             color=DARK_TXT, font=FONT_HEAD,
             vert=MSO_ANCHOR.MIDDLE)

def add_pic(x, y, w, h, path):
    slide.shapes.add_picture(str(path), Cm(x), Cm(y), Cm(w), Cm(h))

# ===== ARKA PLAN: acik renk =====
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                             prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = LIGHT_BG
bg.line.fill.background()

# Renkli ust band
top_band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                   prs.slide_width, Cm(9))
top_band.fill.solid()
top_band.fill.fore_color.rgb = DARK_TXT
top_band.line.fill.background()

# Sag ust dekoratif daire
deco = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                               Cm(50), Cm(-12), Cm(30), Cm(30))
deco.fill.solid()
deco.fill.fore_color.rgb = PURPLE
deco.line.fill.background()
# Transparency yapamiyoruz dogrudan ama renkli daire posteri suslesin

# ===== UST BLOK: BASLIK + YAZARLAR =====
add_text(2, 1.2, 66, 2.2,
         "SignBridge",
         size=80, bold=True, color=CYAN, font=FONT_HEAD,
         align=PP_ALIGN.LEFT)

add_text(2, 3.5, 66, 1.6,
         "Türk İşaret Dili için Çift Yönlü Çeviri Sistemi",
         size=36, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), font=FONT_HEAD)

add_text(2, 5.0, 66, 1.0,
         "Derin Öğrenme Tabanlı Gerçek Zamanlı Tanıma ve 3B Avatar ile Üretim",
         size=22, bold=False, color=RGBColor(0xA8, 0xAE, 0xC1), font=FONT_HEAD)

# Yazar/danisman/kurum bilgisi
add_text(2, 6.5, 66, 0.9,
         "Esma Sıla ŞAHİNCİ    ·    Danışman: Öğr. Gör. Kadir HALTAŞ",
         size=22, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), font=FONT_HEAD)
add_text(2, 7.5, 66, 0.8,
         "Nevşehir Hacı Bektaş Veli Üniversitesi  ·  Mühendislik-Mimarlık Fak.  ·  Bilgisayar Mühendisliği  ·  2026",
         size=16, color=RGBColor(0xCB, 0xD5, 0xE1), font=FONT_HEAD)

# ===== SOL SUTUN =====
col_l_x = 2.0
col_l_w = 32.0

# ÖZET
y = 10.5
add_section_title(col_l_x, y, col_l_w, "ÖZET", PURPLE)
y += 1.8
abstract = ("Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmasına rağmen Türk "
            "İşaret Dili (TİD) için son kullanıcıya yönelik çeviri çözümleri yok denecek "
            "kadar azdır. Bu çalışmada TİD ↔ Türkçe çift yönlü çeviri yapan bir sistem "
            "geliştirilmiştir. Sistem üç bileşenden oluşur: MediaPipe Holistic ile işaret "
            "noktası çıkarımı, 122 kelimelik GRU sinir ağı ile gerçek zamanlı tanıma, ve "
            "akıllı Türkçe NLP modülü ile dilbilgisi açısından doğru cümle üretimi. "
            "Tersine yönde, kullanıcının yazdığı/söylediği Türkçe cümle 3B avatar üzerinde "
            "işaret dili olarak gösterilir. Model 122 sınıf üzerinde %99.88 doğrulama "
            "doğruluğu elde etmiş; sistem hem mobil (Flutter) hem web (Flask + Three.js) "
            "üzerinden eş zamanlı çalışmaktadır.")
add_text(col_l_x, y, col_l_w, 9.5, abstract, size=15, color=DARK_TXT)

# GİRİŞ
y += 9.8
add_section_title(col_l_x, y, col_l_w, "GİRİŞ", CYAN)
y += 1.8
intro = ("İşitme engelli bireyler günlük iletişimde, yazılı Türkçe ile aralarındaki köprü "
         "eksikliğinden kaynaklanan engellerle karşılaşır. Mevcut işaret dili çevirmenleri "
         "sınırlı sayıda ve 7/24 erişilebilir değildir.\n\n"
         "Son yıllarda dizi modelleme (LSTM, GRU, Transformer) ve görüntüden işaret noktası "
         "çıkarımı (MediaPipe, OpenPose) alanlarındaki gelişmeler, gerçek zamanlı işaret "
         "dili tanımayı uygulanabilir hâle getirmiştir (Koller vd., 2020). Ancak "
         "literatürdeki çalışmaların çoğu Amerikan İşaret Dili (ASL) üzerinedir; Türk "
         "İşaret Dili için yeterli büyüklükte etiketli veri seti ve son-kullanıcı "
         "uygulaması bulunmamaktadır. Bu açığı kapatmak amacıyla SignBridge geliştirilmiştir.")
add_text(col_l_x, y, col_l_w, 9.0, intro, size=15, color=DARK_TXT)

# AMAÇ
y += 9.3
add_section_title(col_l_x, y, col_l_w, "ÇALIŞMANIN AMACI", PINK)
y += 1.8
# 4 madde - icon + metin
goals = [
    ("🎯", "TİD → Türkçe:", "Kameradan canlı görüntü alıp işaretleri gramer açısından doğru cümlelere çevirme"),
    ("🎭", "Türkçe → TİD:", "Yazılı/sözlü Türkçe metni 3B avatar üzerinde işaret dili olarak gösterme"),
    ("📱", "Çift Platform:", "Hem mobil (Flutter) hem web (tarayıcı) üzerinde eşzamanlı çalışma"),
    ("⚡", "Düşük Gecikme:", "Gerçek zamanlı kullanım için işaret başına < 1.5 sn tanıma"),
]
for icon, lead, body in goals:
    add_text(col_l_x, y, 1.5, 1.2, icon, size=22, font=FONT_BODY)
    add_text(col_l_x + 1.5, y + 0.05, col_l_w - 1.5, 0.8,
             lead, size=15, bold=True, color=PURPLE)
    add_text(col_l_x + 1.5, y + 0.85, col_l_w - 1.5, 1.0,
             body, size=14, color=DARK_TXT)
    y += 1.85

# YÖNTEM
y += 0.4
add_section_title(col_l_x, y, col_l_w, "YÖNTEM", PURPLE)
y += 1.8

# Veri kutusu
add_rect(col_l_x, y, col_l_w, 2.8, CARD_BG, fill=True)
add_text(col_l_x + 0.3, y + 0.2, col_l_w - 0.6, 0.9,
         "📊 Veri Seti", size=16, bold=True, color=PURPLE)
add_text(col_l_x + 0.3, y + 1.1, col_l_w - 0.6, 1.7,
         "• 122 TİD kelimesi  ·  105.182 örnek (augment edilmiş)\n"
         "• Her örnek: 30 frame × 1755 boyutlu vektör\n"
         "  (1629 ham landmark + 126 hız bileşeni)\n"
         "• Augmentation: gürültü · ölçek · çevirme · döndürme · zaman bükme · ayna",
         size=13, color=DARK_TXT)
y += 3.1

# Model tablosu
add_rect(col_l_x, y, col_l_w, 5.0, CARD_BG, fill=True)
add_text(col_l_x + 0.3, y + 0.2, col_l_w - 0.6, 0.9,
         "🧠 Model Mimarisi (GRU)", size=16, bold=True, color=CYAN)
table_text = ("Tip:                  2 katmanlı GRU (Gated Recurrent Unit)\n"
              "Giriş:                1755 boyut (poz + el + yüz + hız)\n"
              "Gizli boyut:          256  ·  Dropout: 0.3\n"
              "Sınıflandırıcı:      FC(256→128) → ReLU → FC(128→122)\n"
              "Parametre:            ~1.9 milyon\n"
              "Optimizasyon:         AdamW · LR=1e-3 · Batch=64\n"
              "Eğitim:               80 epoch · ~20 dk (NVIDIA GPU)")
add_text(col_l_x + 0.3, y + 1.1, col_l_w - 0.6, 3.8, table_text,
         size=12, color=DARK_TXT, font="Consolas")
y += 5.3

# NLP kutusu
add_rect(col_l_x, y, col_l_w, 3.8, CARD_BG, fill=True)
add_text(col_l_x + 0.3, y + 0.2, col_l_w - 0.6, 0.9,
         "🌐 Akıllı Türkçe NLP", size=16, bold=True, color=PINK)
add_text(col_l_x + 0.3, y + 1.1, col_l_w - 0.6, 2.7,
         "• Özne tespiti (BEN/SEN/BERABER/KENDI) → fiil için doğru kişi eki\n"
         "• Bağlam ayrımı (KAHVE↔KAFE benzer işaretleri ayırt eder)\n"
         "• Soru ekleri ('Beraber kahve içelim mi?' kohortatif)\n"
         "• Türkçe gramer ekleri (ablatif -DAn, locatif -DA, attributiv -lI)\n\n"
         "Örnek:  [BEN, KAHVE, ICMEK] → 'Ben kahve içiyorum.'",
         size=13, color=DARK_TXT)
y += 4.1

# ===== SAĞ SUTUN =====
col_r_x = 36.0
col_r_w = 32.0

y = 10.5
# Mimari diyagram (üstte, gözden geçirme)
add_section_title(col_r_x, y, col_r_w, "SİSTEM MİMARİSİ", CYAN)
y += 1.8
arch_h = 8.5
add_pic(col_r_x, y, col_r_w, arch_h, ART / "architecture_diagram.png")
y += arch_h + 0.5

# DENEYSEL DÜZENEK + ekran goruntuleri
add_section_title(col_r_x, y, col_r_w, "DENEYSEL DÜZENEK", PURPLE)
y += 1.8

# 2 sutunlu - Mobil ss + web ss yan yana
ss_w = (col_r_w - 0.6) / 2
ss_h = 9.5
add_text(col_r_x, y, ss_w, 0.7, "📱 Mobil (Flutter)", size=14, bold=True, color=PURPLE)
add_text(col_r_x + ss_w + 0.6, y, ss_w, 0.7, "💻 Web (Tarayıcı)", size=14, bold=True, color=CYAN)
y += 0.8
add_pic(col_r_x, y, ss_w, ss_h, IMG / "mobil_kamera_merhaba.jpeg")
add_pic(col_r_x + ss_w + 0.6, y, ss_w, ss_h * 0.5,
        IMG / "web_kamera.jpeg")
add_pic(col_r_x + ss_w + 0.6, y + ss_h * 0.5 + 0.2, ss_w, ss_h * 0.5 - 0.2,
        IMG / "web_avatar.jpeg")
y += ss_h + 0.4

# Yazilim yigini
add_rect(col_r_x, y, col_r_w, 2.5, ACCENT_BG, fill=True)
add_text(col_r_x + 0.3, y + 0.2, col_r_w - 0.6, 0.7,
         "⚙️ Yazılım Yığını", size=14, bold=True, color=PURPLE)
add_text(col_r_x + 0.3, y + 1.0, col_r_w - 0.6, 1.5,
         "Backend: Python · Flask · Flask-SocketIO · PyTorch · MediaPipe\n"
         "Mobil:   Flutter · Dart · Camera · WebSocket · Flutter TTS\n"
         "Web:     HTML5 · CSS3 · Three.js · Web Speech API",
         size=12, color=DARK_TXT, font="Consolas")
y += 2.8

# SONUÇLAR
add_section_title(col_r_x, y, col_r_w, "SONUÇLAR", PINK)
y += 1.8

# Büyük doğruluk kutusu
big_box_w = col_r_w * 0.35
add_rect(col_r_x, y, big_box_w, 4.0, PURPLE, fill=True)
add_text(col_r_x + 0.3, y + 0.3, big_box_w - 0.6, 1.5,
         "%99.97", size=54, bold=True,
         color=RGBColor(0xFF, 0xFF, 0xFF), font=FONT_HEAD,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)
add_text(col_r_x + 0.3, y + 2.2, big_box_w - 0.6, 1.5,
         "Genel Doğruluk\n(3660 örnek)",
         size=14, color=RGBColor(0xCB, 0xD5, 0xE1),
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# Yan tablo
tbl_x = col_r_x + big_box_w + 0.4
tbl_w = col_r_w - big_box_w - 0.4
add_rect(tbl_x, y, tbl_w, 4.0, CARD_BG, fill=True)
add_text(tbl_x + 0.3, y + 0.15, tbl_w - 0.6, 0.6,
         "Model Performansı", size=13, bold=True, color=PURPLE)
stats = ("Doğrulama doğruluğu (eğitim):      %99.88\n"
         "Eğitim doğruluğu (son epoch):      %100.0\n"
         "Sınıf sayısı:                       122\n"
         "Çıkarım gecikmesi (GPU):           ~50 ms\n"
         "Çıkarım gecikmesi (CPU):           ~80 ms\n"
         "Eğitim süresi:                      20 dakika\n"
         "Mobil canlı gecikme:               ~250 ms")
add_text(tbl_x + 0.3, y + 0.8, tbl_w - 0.6, 3.1, stats,
         size=11, color=DARK_TXT, font="Consolas")
y += 4.3

# Eğitim grafiği + Confusion matrix yan yana
gw = (col_r_w - 0.4) / 2
gh = 7.5
add_text(col_r_x, y, gw, 0.7, "Eğitim Süreci (80 epoch)", size=13, bold=True, color=PURPLE)
add_text(col_r_x + gw + 0.4, y, gw, 0.7, "Karışıklık Matrisi (30 sık kelime)",
         size=13, bold=True, color=PURPLE)
y += 0.7
add_pic(col_r_x, y, gw, gh, ART / "training_history_122.png")
add_pic(col_r_x + gw + 0.4, y, gw, gh, ART / "confusion_matrix_122.png")
y += gh + 0.5

# CANLI DEMO ekran görüntüsü (mobil_kamera_nasilsin - en güzel ss)
add_text(col_r_x, y, col_r_w, 0.7,
         "💬 Canlı Demo — Cümle Oluşturma",
         size=14, bold=True, color=GREEN)
y += 0.8
demo_h = 8.0
add_pic(col_r_x, y, col_r_w * 0.55, demo_h, IMG / "mobil_kamera_nasılsın.jpeg")
# Yanına avatar fotoğrafları grid
ax = col_r_x + col_r_w * 0.55 + 0.4
aw = col_r_w * 0.45 - 0.4
ah = (demo_h - 0.4) / 2
add_pic(ax, y, aw, ah, IMG / "mobil_avatar_merhaba.jpeg")
add_pic(ax, y + ah + 0.4, aw, ah, IMG / "mobil_avatar_ataturk.jpeg")
y += demo_h + 0.5

# Karşılaşılan zorluklar
add_rect(col_r_x, y, col_r_w, 3.5, ACCENT_BG, fill=True)
add_text(col_r_x + 0.3, y + 0.2, col_r_w - 0.6, 0.7,
         "🔧 Karşılaşılan Zorluklar & Çözümler", size=14, bold=True, color=PURPLE)
add_text(col_r_x + 0.3, y + 0.9, col_r_w - 0.6, 2.5,
         "• Benzer işaretlerin karışması (KAHVE↔KAFE, BEN↔IYI):\n"
         "  Per-kelime güven eşiği (%92-%97) + 3 frame yumuşatma\n"
         "• MediaPipe paralel çağrı segfault: threading.Lock ile sıralandı\n"
         "• Mobil kamera donması: zoom çağrısı stream sonrası ertelendi\n"
         "• Geçiş gürültüsü: 400ms cooldown + buffer reset",
         size=12, color=DARK_TXT)
y += 3.8

# ===== ALT KISIM: SONUÇ + KAYNAKLAR (tam genislikte) =====
y_alt = 88.0
add_section_title(2, y_alt, 66, "SONUÇ ve GELECEK ÇALIŞMALAR", GREEN)
y_alt += 1.8

# 2 kolonlu - sol sonuc, sag kaynaklar
add_rect(2, y_alt, 32, 7.5, CARD_BG, fill=True)
add_text(2.3, y_alt + 0.2, 31.4, 7.1,
         "✅ 122 kelime üzerinde %99.97 canlı tanıma doğruluğu\n"
         "✅ Çift yönlü çeviri: hem TİD okuma hem avatar ile üretim\n"
         "✅ Mobil + Web çift platform, cihaz fark etmeksizin\n"
         "✅ Akıllı NLP ile dilbilgisi açısından doğru Türkçe cümleler\n"
         "✅ 10 demo cümlesinde %100 başarı\n\n"
         "🚀 GELECEK:\n"
         "    • Kelime sayısı 500+\n"
         "    • Transformer tabanlı daha sağlam model\n"
         "    • Doğrudan cümle düzeyi tanıma\n"
         "    • iOS desteği",
         size=14, color=DARK_TXT)

# Kaynaklar
add_rect(36, y_alt, 32, 7.5, CARD_BG, fill=True)
add_text(36.3, y_alt + 0.2, 31.4, 0.8,
         "KAYNAKLAR", size=14, bold=True, color=PURPLE)
refs = ("[1] Koller, O., Camgoz, N. C., Ney, H., & Bowden, R. (2020). Weakly\n"
        "    Supervised Learning with Multi-Stream CNN-LSTM-HMMs. IEEE TPAMI.\n"
        "[2] Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building\n"
        "    Perception Pipelines. arXiv:1906.08172.\n"
        "[3] Cho, K. et al. (2014). On the Properties of Neural Machine\n"
        "    Translation: Encoder–Decoder Approaches. arXiv:1409.1259.\n"
        "[4] Camgoz, N. C. et al. (2018). Neural Sign Language Translation. CVPR.\n"
        "[5] Şahin, M. (2018). Türk İşaret Dili Sözlüğü. Aile, Çalışma ve Sosyal\n"
        "    Hizmetler Bakanlığı, Ankara.\n"
        "[6] PyTorch Team. (2019). PyTorch: An Imperative Style, High-Performance\n"
        "    Deep Learning Library. NeurIPS.")
add_text(36.3, y_alt + 1.0, 31.4, 6.3, refs, size=10, color=GREY_TXT, font="Consolas")

# Alt footer
add_text(2, 97.5, 66, 1.5,
         "SignBridge © 2026  ·  Nevşehir Hacı Bektaş Veli Üniversitesi  ·  Bilgisayar Mühendisliği Bitirme Projesi",
         size=11, color=GREY_TXT, font=FONT_HEAD,
         align=PP_ALIGN.CENTER, vert=MSO_ANCHOR.MIDDLE)

# ============ KAYDET ============
prs.save(str(OUT_PATH))
print(f"\n✅ Poster kaydedildi: {OUT_PATH}")
print(f"   Boyut: {W_CM}x{H_CM} cm (dikey)")
