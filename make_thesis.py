# -*- coding: utf-8 -*-
"""
SignBridge bitirme odevi - Nevsehir HBV Univ. yazim kilavuzuna gore doldurulmus tez.
Sablonun bolum yapisi aynen korunmustur. Icerik genisletilmistir (~40 sayfa hedefi).
Cikti: C:\\Users\\leven\\Downloads\\SignBridge-Bitirme-Odevi.docx
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

sec = doc.sections[0]
sec.left_margin   = Cm(3.0)
sec.right_margin  = Cm(2.0)
sec.top_margin    = Cm(2.5)
sec.bottom_margin = Cm(2.5)


def new_section():
    """Yeni sayfadan baslayan bir bolum (section) ekler, kenar bosluklarini korur."""
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    s.left_margin = Cm(3.0)
    s.right_margin = Cm(2.0)
    s.top_margin = Cm(2.5)
    s.bottom_margin = Cm(2.5)
    return s


def _set_pgnum_format(section, fmt, start=1):
    """sectPr icine pgNumType (bicim + baslangic) ekler. fmt: 'lowerRoman' / 'decimal'."""
    sectPr = section._sectPr
    for el in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(el)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))
    cols = sectPr.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(pg)
    else:
        sectPr.append(pg)


def add_page_number(section, fmt, start=1):
    """Bolumun alt ortasina PAGE alani ekler (kilavuz md.1.7)."""
    section.footer.is_linked_to_previous = False
    _set_pgnum_format(section, fmt, start)
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
    i = OxmlElement("w:instrText"); i.set(qn("xml:space"), "preserve"); i.text = "PAGE"
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
    run._r.append(b); run._r.append(i); run._r.append(e)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
pf = normal.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
pf.space_after = Pt(6)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def body(text, indent=True):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.25)
    return p


def h1(num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run((f"{num}. " if num else "") + text)
    r.bold = True; r.font.size = Pt(12); r.font.name = "Times New Roman"
    return p


def h2(num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{num} {text}")
    r.bold = True; r.font.size = Pt(12); r.font.name = "Times New Roman"
    return p


def h3(num, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{num} {text}")
    r.bold = True; r.italic = True; r.font.size = Pt(12); r.font.name = "Times New Roman"
    return p


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    r.bold = True; r.font.size = Pt(11); r.font.name = "Times New Roman"
    return p


def make_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        rp = hdr[i].paragraphs[0].add_run(htext)
        rp.bold = True; rp.font.size = Pt(11); rp.font.name = "Times New Roman"
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            rp = cells[i].paragraphs[0].add_run(str(val))
            rp.font.size = Pt(11); rp.font.name = "Times New Roman"
    return t


def page_break():
    doc.add_page_break()


IMG = r"C:\Users\leven\Downloads\poster-images\framed"
ART = r"C:\Projects\sign_bridge\experiment_v2\poster_artifacts"
REP = r"C:\Projects\sign_bridge\report_images"
GEN = r"C:\Users\leven\Downloads\thesis-figures"


def add_image_abs(path, width_cm):
    """Mutlak yoldan sekil ekler, ortalar. Caption ALTTA cagrilmalidir."""
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_image(filename, width_cm):
    """Sekli ortalanmis olarak ekler. Sekil basligi (caption) ALTTA cagrilmalidir."""
    import os
    path = os.path.join(IMG, filename)
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def two_images_row(file_left, file_right, width_cm=6.0):
    """Iki gorseli yan yana borderless tablo ile yerlestirir."""
    import os
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for cell, fn in zip(t.rows[0].cells, (file_left, file_right)):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(os.path.join(IMG, fn), width=Cm(width_cm))


def two_images_row_abs(path_left, path_right, width_cm=7.0):
    """Iki gorseli (mutlak yol) yan yana borderless tablo ile yerlestirir."""
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for cell, pth in zip(t.rows[0].cells, (path_left, path_right)):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(pth, width=Cm(width_cm))


def cen(txt, bold=True, size=14, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(txt)
    r.bold = bold; r.italic = italic; r.font.size = Pt(size); r.font.name = "Times New Roman"
    return p


# ============================================================ KAPAK
for _ in range(2):
    doc.add_paragraph()
cen("T.C.")
cen("NEVŞEHİR HACI BEKTAŞ VELİ ÜNİVERSİTESİ")
cen("MÜHENDİSLİK-MİMARLIK FAKÜLTESİ")
cen("BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ")
for _ in range(4):
    doc.add_paragraph()
cen("SIGNBRIDGE: TÜRK İŞARET DİLİ İLE TÜRKÇE ARASINDA", size=16)
cen("ÇİFT YÖNLÜ GERÇEK ZAMANLI ÇEVİRİ SİSTEMİ", size=16)
for _ in range(2):
    doc.add_paragraph()
cen("BİTİRME ÖDEVİ")
for _ in range(4):
    doc.add_paragraph()
cen("Hazırlayan", bold=False, size=12)
cen("Esma Sıla ŞAHİNCİ", size=13)
for _ in range(3):
    doc.add_paragraph()
cen("Danışman", bold=False, size=12)
cen("Öğr. Gör. Dr. Kadir HALTAŞ", size=13)
for _ in range(5):
    doc.add_paragraph()
cen("NEVŞEHİR", size=12)
cen("2026", size=12)
# Kapak ayri bir bolum: sayfa numarasi yok (kilavuz md.1.7). On sozden itibaren
# yeni bolum baslar ve roma rakamiyla numaralanir.
new_section()

# ============================================================ ONSOZ
h1("", "ÖNSÖZ")
body("Bu bitirme ödevinde, işitme engelli bireyler ile işiten bireyler arasındaki "
     "iletişim engelini azaltmayı amaçlayan, Türk İşaret Dili (TİD) ile Türkçe "
     "arasında çift yönlü çeviri yapabilen SignBridge adlı sistem geliştirilmiştir. "
     "Çalışma kapsamında; günlük yaşamda sık kullanılan 122 kelimelik bir işaret "
     "sözlüğü için bir işaret tanıma modeli, tanınan kelimeleri akıcı Türkçeye çeviren "
     "kurala dayalı bir doğal dil işleme katmanı, Türkçeyi işaret diline çeviren üç "
     "boyutlu bir avatar ve hem web hem de mobil platformda çalışan bir uygulama "
     "tasarlanıp gerçeklenmiştir.")
body("İşaret dili tanıma, görüntü işleme, derin öğrenme ve doğal dil işleme gibi "
     "birçok alanı bir araya getiren disiplinler arası bir çalışmadır. Bu süreçte hem "
     "teorik bilgi hem de uygulamalı mühendislik becerileri kazanılmıştır. Geliştirilen "
     "sistemin, ileride daha kapsamlı çalışmalara temel oluşturması hedeflenmiştir.")
body("Çalışma süresince bilgi, tecrübe ve yönlendirmeleriyle bana destek olan "
     "danışmanım Öğr. Gör. Dr. Kadir HALTAŞ'na, bölümümüzün değerli öğretim "
     "üyelerine ve bu süreçte maddi-manevi desteklerini esirgemeyen aileme "
     "teşekkürlerimi sunarım.")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
r = p.add_run("Esma Sıla ŞAHİNCİ"); r.font.name = "Times New Roman"; r.font.size = Pt(12)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
r = p.add_run("Nevşehir, 2026"); r.font.name = "Times New Roman"; r.font.size = Pt(12)
page_break()

# ============================================================ ICINDEKILER
h1("", "İÇİNDEKİLER")


def toc_entry(text, page, level=0):
    """Icindekiler satiri: noktali (dot leader) ve saga hizali sayfa numarasi."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(3)
    indent = Cm(0.0) if level == 0 else (Cm(0.9) if level == 1 else Cm(1.8))
    p.paragraph_format.left_indent = indent
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(Cm(15.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    r = p.add_run(f"{text}\t{page}")
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
    if level == 0:
        r.bold = True
    return p


toc = [
    ("ÖNSÖZ", "i", 0),
    ("İÇİNDEKİLER", "ii", 0),
    ("ŞEKİLLER LİSTESİ", "iv", 0),
    ("TABLOLAR LİSTESİ", "v", 0),
    ("KISALTMALAR", "vi", 0),
    ("ÖZET", "vii", 0),
    ("1.  GİRİŞ", "1", 0),
    ("1.1.  Problemin Tanımı ve Önemi", "1", 1),
    ("1.2.  İşaret Dilinin Doğası", "2", 1),
    ("1.3.  Ödevin Amacı", "3", 1),
    ("1.4.  Ödevin Kapsamı", "3", 1),
    ("1.5.  Ödevin Katkıları", "4", 1),
    ("1.6.  Raporun Organizasyonu", "4", 1),
    ("2.  GENEL KISIMLAR", "5", 0),
    ("2.1.  Türk İşaret Dili", "5", 1),
    ("2.2.  İşaret Dili Tanıma Yaklaşımları", "6", 1),
    ("2.3.  İlgili Çalışmalar", "8", 1),
    ("2.4.  MediaPipe ve Holistic Çözümü", "9", 1),
    ("2.5.  Yapay Sinir Ağları ve Derin Öğrenme", "10", 1),
    ("2.6.  Doğal Dil İşleme", "13", 1),
    ("2.7.  Ses Teknolojileri", "14", 1),
    ("2.8.  Üç Boyutlu Karakter Animasyonu", "14", 1),
    ("2.9.  Model Dağıtımı ve ONNX", "15", 1),
    ("3.  MATERYAL VE YÖNTEM", "16", 0),
    ("3.1.  Genel Sistem Mimarisi", "16", 1),
    ("3.2.  Kullanılan Yazılım ve Araçlar", "17", 1),
    ("3.3.  Veri Seti Oluşturma", "18", 1),
    ("3.4.  Veri Artırma", "21", 1),
    ("3.5.  Öznitelik Çıkarımı ve Normalleştirme", "22", 1),
    ("3.6.  Model Mimarisi", "24", 1),
    ("3.7.  Model Eğitimi", "25", 1),
    ("3.8.  Gerçek Zamanlı Tahmin ve Yumuşatma", "26", 1),
    ("3.9.  Kural Tabanlı Doğal Dil İşleme Katmanı", "27", 1),
    ("3.10.  Üç Boyutlu Avatar", "29", 1),
    ("3.11.  Ses Bileşenleri", "30", 1),
    ("3.12.  Web Uygulaması", "30", 1),
    ("3.13.  Mobil Uygulama", "31", 1),
    ("3.14.  Demo Modu", "33", 1),
    ("4.  BULGULAR", "34", 0),
    ("5.  TARTIŞMA VE SONUÇ", "38", 0),
    ("KAYNAKLAR", "41", 0),
    ("EKLER", "42", 0),
]
for text, page, lvl in toc:
    toc_entry(text, page, lvl)
page_break()

# ============================================================ SEKILLER LISTESI
h1("", "ŞEKİLLER LİSTESİ")
figs = [
    ("Şekil 3.1.", "SignBridge sisteminin genel mimarisi ve çift yönlü veri akışı"),
    ("Şekil 3.2.", "Veri toplama arayüzünde MediaPipe ile çıkarılan anahtar noktalar"),
    ("Şekil 3.3.", "Bir işaret dizisinin 30 kareye bölünmesi"),
    ("Şekil 3.4.", "İki aşamalı normalleştirme (omuz ve bilek merkezli)"),
    ("Şekil 3.5.", "GRU tabanlı sınıflandırma modelinin katman yapısı"),
    ("Şekil 3.6.", "Gerçek zamanlı tahmin ve yumuşatma akış şeması"),
    ("Şekil 3.7.", "Üç boyutlu avatarın poz editörü ve kaydedilmiş bir işaretin oynatılması"),
    ("Şekil 3.8.", "Mobil uygulama kamera ve avatar ekranları"),
    ("Şekil 4.1.", "Eğitim ve doğrulama doğruluk eğrileri"),
    ("Şekil 4.2.", "En sık 30 kelime için karışıklık matrisi"),
    ("Şekil E.1.", "Web arayüzü – kamera sekmesi"),
    ("Şekil E.2.", "Web arayüzü – avatar sekmesi"),
    ("Şekil E.3.", "Mobil uygulama – kamera ekranı"),
    ("Şekil E.4.", "Mobil uygulama – avatar ekranı (MERHABA, HOŞÇA KAL)"),
    ("Şekil E.5.", "Mobil uygulama – avatar ekranı (ATATÜRK, ELHAMDÜLİLLAH)"),
]
for k, v in figs:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.5))
    r = p.add_run(f"{k} {v}\t"); r.font.name = "Times New Roman"; r.font.size = Pt(12)
page_break()

# ============================================================ TABLOLAR LISTESI
h1("", "TABLOLAR LİSTESİ")
tbls = [
    ("Tablo 3.1.", "122 kelimelik sözlüğün kategorilere göre dağılımı"),
    ("Tablo 3.2.", "Uygulanan veri artırma teknikleri"),
    ("Tablo 3.3.", "Öznitelik vektörünün bileşenleri"),
    ("Tablo 3.4.", "Model mimarisi ve eğitim hiperparametreleri"),
    ("Tablo 3.5.", "Gerçek zamanlı tahmin eşik değerleri"),
    ("Tablo 3.6.", "Kural tabanlı NLP katmanı örnek dönüşümleri"),
    ("Tablo 4.1.", "Veri seti ve model başarım özeti"),
    ("Tablo 4.2.", "Web ve mobil platform karşılaştırması"),
    ("Tablo E.1.", "122 kelimelik sözlüğün tam listesi"),
]
for k, v in tbls:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.5))
    r = p.add_run(f"{k} {v}\t"); r.font.name = "Times New Roman"; r.font.size = Pt(12)
page_break()

# ============================================================ KISALTMALAR
h1("", "KISALTMALAR")
make_table(["Kısaltma", "Açıklama"], [
    ["TİD", "Türk İşaret Dili"],
    ["GRU", "Gated Recurrent Unit (Geçitli Tekrarlayan Birim)"],
    ["LSTM", "Long Short-Term Memory (Uzun-Kısa Süreli Bellek)"],
    ["RNN", "Recurrent Neural Network (Tekrarlayan Sinir Ağı)"],
    ["CNN", "Convolutional Neural Network (Evrişimli Sinir Ağı)"],
    ["NLP", "Natural Language Processing (Doğal Dil İşleme)"],
    ["TTS", "Text-to-Speech (Metinden Sese)"],
    ["STT", "Speech-to-Text (Sesten Metne)"],
    ["ONNX", "Open Neural Network Exchange"],
    ["FPS", "Frames Per Second (Saniyedeki Kare Sayısı)"],
    ["API", "Application Programming Interface (Uygulama Programlama Arayüzü)"],
    ["JSON", "JavaScript Object Notation"],
    ["ReLU", "Rectified Linear Unit (Düzeltilmiş Doğrusal Birim)"],
    ["GPU", "Graphics Processing Unit (Grafik İşlem Birimi)"],
])
page_break()

# ============================================================ OZET
h1("", "ÖZET")
body("Bu çalışmada, Türk İşaret Dili (TİD) ile Türkçe arasında çift yönlü ve gerçek "
     "zamanlı çeviri yapabilen SignBridge adlı bir sistem geliştirilmiştir. Sistem iki "
     "ana yönde çalışmaktadır. Birinci yönde, kamera önünde yapılan işaret hareketleri "
     "yapay zekâ modeliyle tanınarak Türkçe yazıya ve sese dönüştürülür. İkinci yönde "
     "ise yazılan veya söylenen Türkçe ifadeler üç boyutlu bir avatar aracılığıyla "
     "işaret diline çevrilir.")
body("İşaret tanıma bileşeni için günlük yaşamda sık kullanılan 122 kelimelik bir "
     "sözlük belirlenmiş ve toplam 14.974 adet ham örnek dizisi toplanmıştır. Altı "
     "farklı veri artırma tekniğiyle bu sayı 105.182 örneğe çıkarılmıştır. Her örnek, "
     "MediaPipe Holistic kütüphanesiyle çıkarılan 543 anahtar noktanın 30 karelik "
     "dizisinden oluşmakta ve kare başına 1755 boyutlu bir öznitelik vektörüne "
     "dönüştürülmektedir. Sınıflandırma için iki katmanlı, 256 gizli birimli bir GRU "
     "tabanlı tekrarlayan sinir ağı kullanılmış ve bağımsız doğrulama kümesinde %99,88 "
     "doğruluk elde edilmiştir.")
body("Eğitilen model hem masaüstü/web platformunda (Python, Flask, PyTorch) hem de "
     "ONNX formatına dönüştürülerek Android tabanlı mobil uygulamada (Flutter) "
     "çalıştırılmıştır. Mobil uygulama, internet bağlantısı olmadan tamamen cihaz "
     "üzerinde tahmin yapabilmektedir. İşaret dizisinin akıcı Türkçe cümleye "
     "dönüştürülmesi için kurala dayalı bir doğal dil işleme katmanı geliştirilmiştir. "
     "Sonuçlar, sınırlı bir sözlük üzerinde TİD tanımanın gerçek zamanlı ve cihaz "
     "üzerinde uygulanabilir olduğunu göstermektedir.")
body("Anahtar Kelimeler: Türk İşaret Dili, işaret dili tanıma, derin öğrenme, GRU, "
     "MediaPipe, doğal dil işleme, mobil uygulama, çift yönlü çeviri.", indent=False)
# Govde (Giris'ten itibaren) yeni bolum: Arap rakamiyla 1'den baslar (kilavuz md.1.7).
new_section()

# ============================================================ 1. GIRIS
h1("1", "GİRİŞ")

h2("1.1.", "Problemin Tanımı ve Önemi")
body("İşitme engeli, dünya genelinde yüz milyonlarca insanı etkileyen yaygın bir "
     "durumdur. Dünya Sağlık Örgütü'nün verilerine göre dünya nüfusunun önemli bir "
     "bölümü işitme kaybıyla yaşamakta; Türkiye'de de yüz binlerce işitme engelli "
     "vatandaş bulunmaktadır. Bu bireylerin büyük bölümü, birincil iletişim aracı "
     "olarak işaret dilini kullanır.")
body("Ne var ki işiten toplumun büyük çoğunluğu işaret dili bilmemektedir. Bu durum, "
     "işitme engelli bireylerin hastane, okul, resmî kurum, alışveriş ve günlük sosyal "
     "etkileşim gibi pek çok ortamda iletişim güçlüğü yaşamasına neden olur. İşaret "
     "dili öğrenmek hem uzun zaman almakta hem de düzenli pratik gerektirmektedir. "
     "Profesyonel işaret dili tercümanlarına erişim ise sınırlıdır; tercüman desteği "
     "çoğu zaman hem maliyetli hem de her an ve her yerde ulaşılabilir değildir.")
body("Bu iletişim engeli, işitme engelli bireylerin eğitim, istihdam ve sosyal "
     "yaşama tam katılımının önünde önemli bir engel oluşturmaktadır. Bilgisayarlı "
     "görü ve derin öğrenme alanındaki gelişmeler, bu engelin teknoloji yardımıyla "
     "azaltılabileceği fikrini gündeme getirmiştir. Kameralı bir cihazın işaretleri "
     "tanıyıp yazıya ve sese çevirebilmesi, ya da yazılı/sesli bir metni işaret diline "
     "dönüştürebilmesi, taraflar arasında doğrudan ve aracısız bir iletişim imkânı "
     "sunabilir.")
body("Bu bitirme ödevi kapsamında geliştirilen SignBridge sistemi, söz konusu "
     "ihtiyaçtan doğmuştur. Sistemin amacı, sınırlı ancak günlük yaşamda sık kullanılan "
     "bir kelime kümesi üzerinde, gerçek zamanlı ve iki yönde çalışabilen, taşınabilir "
     "ve düşük maliyetli bir çeviri aracı ortaya koymaktır.")

h2("1.2.", "İşaret Dilinin Doğası")
body("İşaret dili, yalnızca el hareketlerinden ibaret değildir. Bir işaret; el şekli, "
     "elin uzaydaki konumu, yönelimi, hareketin yönü ve hızı ile yüz ifadeleri ve vücut "
     "duruşu gibi bileşenlerin eş zamanlı birleşiminden oluşur. Bu nedenle işaret dili, "
     "konuşma dilinden farklı olarak çok kanallı (multimodal) ve uzamsal-zamansal bir "
     "dildir.")
body("İşaret dilinin en belirgin özelliklerinden biri, hareket içermesidir. Bir kelime "
     "çoğu zaman tek bir duruşla değil, belirli bir süre boyunca yapılan bir hareketle "
     "ifade edilir. Bu durum, işaret tanımayı statik bir görüntü sınıflandırma "
     "probleminden ayırır: tek bir fotoğraf bir kelimeyi anlatmaya yetmez. Bu nedenle "
     "bu çalışmada, işaretin zaman içindeki değişimini yakalayan kısa görüntü dizileri "
     "kullanılmıştır.")
body("Ayrıca Türk İşaret Dili'nin söz dizimi, Türkçeden farklıdır. Tanınan kelimeler "
     "ham hâlleriyle (gloss) art arda sıralandığında doğrudan akıcı bir Türkçe cümle "
     "oluşturmaz. Bu da tanıma adımının ardından ayrı bir dil işleme adımını zorunlu "
     "kılar.")

h2("1.3.", "Ödevin Amacı")
body("Bu çalışmanın temel amacı; günlük yaşamda sık kullanılan 122 kelimelik bir TİD "
     "sözlüğü üzerinde, gerçek zamanlı çalışabilen, hem masaüstü/web hem de mobil "
     "platformda kullanılabilen, internet bağlantısı olmadan da çalışabilen çift yönlü "
     "bir işaret dili çeviri sistemi tasarlamak ve gerçeklemektir. Alt amaçlar şunlardır:")
body("a) Anahtar nokta tabanlı, ışık ve arka plan değişimlerine dayanıklı bir işaret "
     "tanıma yaklaşımı geliştirmek; b) sınırlı veriyle yüksek doğruluk elde etmek için "
     "veri artırma ve normalleştirme yöntemleri uygulamak; c) tanınan kelimeleri akıcı "
     "Türkçeye çeviren bir dil işleme katmanı oluşturmak; d) Türkçeyi işaret diline "
     "çeviren üç boyutlu bir avatar geliştirmek; e) sistemi gerçek bir mobil uygulamada "
     "cihaz üzerinde çalışacak biçimde dağıtmak.", indent=True)

h2("1.4.", "Ödevin Kapsamı")
body("Çalışma kapsamında 122 kelimelik bir TİD veri seti toplanmış ve veri artırma ile "
     "genişletilmiştir. MediaPipe tabanlı bir anahtar nokta çıkarımı ve GRU tabanlı bir "
     "dizi sınıflandırma modeli geliştirilmiştir. Tanınan kelimeleri akıcı Türkçe "
     "cümleye dönüştüren kurala dayalı bir doğal dil işleme katmanı yazılmıştır. "
     "Türkçeyi işaret diline çeviren üç boyutlu bir avatar oluşturulmuştur. Sistem hem "
     "bir web sunucusu hem de bir Android mobil uygulaması olarak uygulanmıştır.")
body("Çalışma, sınırlı bir sözlük (122 kelime) ile sınırlandırılmıştır; tüm TİD "
     "sözcük dağarcığını kapsamak bu ödevin kapsamı dışındadır. Benzer biçimde, cümle "
     "çeviri katmanı istatistiksel bir modele değil, dilbilgisi kurallarına "
     "dayanmaktadır; bunun gerekçesi ikinci bölümde açıklanmıştır.")

h2("1.5.", "Ödevin Katkıları")
body("Bu çalışmanın başlıca katkıları şöyle özetlenebilir: (i) 122 kelimelik, anahtar "
     "nokta tabanlı ve gizliliği koruyan bir TİD veri seti oluşturulmuş; (ii) küçük "
     "veriyle yüksek doğruluk sağlayan, iki aşamalı normalleştirme ve altı tip veri "
     "artırma içeren bir işleme hattı önerilmiş; (iii) gerçek zamanlı ve cihaz üzerinde "
     "çalışabilen, web ve mobil platformlarda tutarlı davranan çift yönlü bir sistem "
     "geliştirilmiş; (iv) paralel veri seti bulunmayan TİD için pratik, kurala dayalı "
     "bir cümle oluşturma katmanı tasarlanmıştır.")

h2("1.6.", "Raporun Organizasyonu")
body("Raporun ikinci bölümünde, işaret dili tanıma alanındaki temel kavramlar, "
     "kullanılan teknolojiler ve ilgili çalışmalar açıklanmaktadır. Üçüncü bölümde "
     "kullanılan materyal ve yöntem; sistem mimarisi, veri seti, öznitelik çıkarımı, "
     "model mimarisi, doğal dil işleme katmanı, avatar, web ve mobil uygulama ayrıntılı "
     "olarak ele alınmaktadır. Dördüncü bölümde elde edilen bulgular sunulmakta; beşinci "
     "bölümde ise sonuçlar tartışılmakta ve gelecek çalışmalara yönelik öneriler "
     "verilmektedir. Raporun sonunda kaynaklar ve ekler yer almaktadır.")
page_break()

# ============================================================ 2. GENEL KISIMLAR
h1("2", "GENEL KISIMLAR")

h2("2.1.", "Türk İşaret Dili")
body("Türk İşaret Dili (TİD), Türkiye'deki işitme engelli topluluğun kullandığı, "
     "kendine özgü dilbilgisi yapısına sahip doğal bir dildir. Konuşma dillerinden "
     "bağımsız bir gramere sahiptir; yani Türkçenin işaretlerle harfi harfine "
     "kodlanmış bir biçimi değildir. TİD'in kendi söz dizimi, çekim mantığı ve "
     "deyimsel yapıları vardır.")

h3("2.1.1.", "İşaretin Bileşenleri")
body("Bir işaret beş temel bileşenden oluşur: el şekli (parmakların durumu), elin "
     "konumu (vücuda göre nerede yapıldığı), elin yönelimi (avuç içinin yönü), hareket "
     "(yön, hız ve tekrar) ve el dışı işaretler (yüz ifadesi, kaş ve baş hareketleri, "
     "vücut duruşu). Bu bileşenlerin tümü anlam taşır; örneğin aynı el şekli farklı "
     "konum veya hareketle farklı kelimeler üretebilir. Bu nedenle başarılı bir tanıma "
     "sisteminin yalnızca elleri değil, vücut ve yüz bilgisini de dikkate alması "
     "gerekir.")

h3("2.1.2.", "Söz Dizimi Farkı ve Çeviri Gereksinimi")
body("TİD'de cümle ögelerinin sıralanışı Türkçeden farklı olabilir ve birçok dilbilgisi "
     "öğesi (ekler, bağlaçlar, zaman kipleri) işaretle birebir karşılanmaz. Bu yüzden "
     "tanınan işaretlerin ham listesi, doğrudan okunabilir bir Türkçe cümle vermez. "
     "Örneğin ardışık olarak BERABER, KAHVE ve İÇMEK işaretleri tanındığında, akıcı "
     "Türkçe karşılığı \"Beraber kahve içelim mi?\" olmalıdır. Bu dönüşüm, tanıma "
     "adımından ayrı bir dil işleme adımını gerektirir.")

h2("2.2.", "İşaret Dili Tanıma Yaklaşımları")
body("Literatürde işaret dili tanıma için kullanılan başlıca üç yaklaşım vardır: "
     "sensör/eldiven tabanlı, ham görüntü (CNN) tabanlı ve iskelet/anahtar nokta "
     "tabanlı yaklaşımlar.")

h3("2.2.1.", "Sensör ve Eldiven Tabanlı Yaklaşımlar")
body("Bu yaklaşımda kullanıcı, parmak bükülmesini ve el hareketini ölçen sensörlerle "
     "donatılmış özel bir eldiven giyer. Ölçüm doğruluğu yüksek olabilir; ancak özel "
     "donanım gerektirmesi, maliyeti ve giyilebilir olması nedeniyle günlük ve yaygın "
     "kullanım için elverişli değildir. Bu çalışmada herhangi bir özel donanım "
     "gerektirmeyen, yalnızca kamera kullanan bir çözüm hedeflendiğinden bu yaklaşım "
     "tercih edilmemiştir.")

h3("2.2.2.", "Görüntü (CNN) Tabanlı Yaklaşımlar")
body("Bu yaklaşımda ham görüntü veya video kareleri doğrudan evrişimli sinir ağlarına "
     "(CNN) verilir ve ağ, pikseller üzerinden öğrenir. Bu yöntem çok güçlü olabilir "
     "ancak büyük miktarda etiketli veri ve yüksek hesaplama gücü gerektirir. Ayrıca "
     "ışık, arka plan, kıyafet ve kişi değişimlerinden kolayca etkilenir; modelin bu "
     "değişkenlere karşı dayanıklı olabilmesi için çok daha geniş ve çeşitli veri "
     "kümeleri gerekir. Mobil cihazlarda gerçek zamanlı çalıştırmak da görece zordur.")

h3("2.2.3.", "İskelet / Anahtar Nokta Tabanlı Yaklaşımlar")
body("Bu yaklaşımda önce görüntüden el, vücut ve yüz anahtar noktaları (landmark) "
     "çıkarılır; ardından bu düşük boyutlu nokta dizileri bir dizi modeline verilir. "
     "Anahtar nokta çıkarımı, MediaPipe gibi önceden eğitilmiş ve optimize edilmiş "
     "kütüphanelerle gerçek zamanlı yapılabilir. Bu yaklaşımın avantajları: daha az "
     "veriyle çalışabilmesi, ışık ve arka plan değişimlerine karşı dayanıklı olması, "
     "düşük hesaplama yüküyle mobil cihazlarda dahi çalışabilmesi ve ham görüntü "
     "saklanmadığı için gizliliği korumasıdır. Bu nedenlerle SignBridge'de anahtar "
     "nokta tabanlı yaklaşım benimsenmiştir.")

h2("2.3.", "İlgili Çalışmalar")
body("İşaret dili tanıma, son yıllarda derin öğrenmenin yaygınlaşmasıyla hızla "
     "gelişen bir araştırma alanıdır. Yapılan çalışmalar genel olarak iki gruba "
     "ayrılabilir: izole (tek tek) kelime tanıma ve sürekli (cümle düzeyinde) işaret "
     "tanıma. İzole kelime tanıma, sabit bir sözlükteki kelimeleri tek tek "
     "sınıflandırmaya odaklanır ve bu çalışmanın da ana yaklaşımıdır. Sürekli işaret "
     "tanıma ise kesintisiz bir işaret akışını bölütleyip çevirmeyi hedefler ve daha "
     "zordur.")
body("Anahtar nokta tabanlı çalışmalarda, MediaPipe veya OpenPose gibi araçlarla "
     "çıkarılan iskelet verileri üzerinde LSTM, GRU veya Transformer tabanlı modeller "
     "yaygın olarak kullanılmaktadır. Bu çalışmalar, ham görüntü tabanlı yöntemlere "
     "göre daha az veriyle benzer doğruluklara ulaşabildiklerini göstermiştir. "
     "SignBridge, bu literatürdeki anahtar nokta + tekrarlayan sinir ağı yaklaşımını "
     "temel almakta; ek olarak çift yönlü çeviri (avatar) ve mobil cihaz üzerinde "
     "çalışma özellikleriyle bütüncül bir uygulama sunmaktadır.")

h2("2.4.", "MediaPipe ve Holistic Çözümü")
body("MediaPipe, Google tarafından geliştirilen, gerçek zamanlı bilgisayarlı görü "
     "işlem hatları kurmaya yarayan açık kaynaklı bir çatıdır (Lugaresi ve diğ., 2019). "
     "Bu çalışmada kullanılan "
     "MediaPipe Holistic çözümü, tek bir kamera görüntüsünden eş zamanlı olarak yüz, "
     "vücut (poz) ve iki el iskeletini çıkarır. Toplam 543 anahtar nokta elde edilir: "
     "468 yüz noktası, 33 vücut (poz) noktası, 21 sol el ve 21 sağ el noktası. Her "
     "nokta için x, y ve z koordinatları üretilir; x ve y görüntü düzlemindeki konumu, "
     "z ise göreli derinliği temsil eder.")
body("MediaPipe'ın bu noktaları çıkarması, sistemin görüntüyü doğrudan saklamak yerine "
     "yalnızca sayısal koordinatlarla çalışmasını sağlar. Bu durum, hem veri boyutunu "
     "küçültür hem de kişinin yüzünün veya görüntüsünün kaydedilmemesi sayesinde "
     "gizliliği korur. Ayrıca MediaPipe gerçek zamanlı çalışacak biçimde optimize "
     "edildiği için, hem masaüstünde hem de mobil cihazlarda kullanılabilir.")

h2("2.5.", "Yapay Sinir Ağları ve Derin Öğrenme")
body("Derin öğrenme, çok katmanlı yapay sinir ağlarının verideki örüntüleri "
     "öğrenmesine dayanan bir makine öğrenmesi alt alanıdır. Bu çalışmada, işaretin "
     "zaman içindeki değişimini öğrenebilen tekrarlayan sinir ağları kullanılmıştır.")

h3("2.5.1.", "Tekrarlayan Sinir Ağları (RNN)")
body("Tekrarlayan sinir ağları (RNN), dizisel verilerle çalışmak üzere tasarlanmıştır. "
     "Bir RNN, her adımda hem o anki girdiyi hem de önceki adımdan gelen \"gizli "
     "durumu\" (hidden state) işler; böylece geçmiş bilgiyi bir tür bellekte taşır. Bu "
     "özellik, işaret hareketi gibi zaman içinde anlam kazanan verilerde önemlidir. "
     "Ancak klasik RNN'ler, uzun dizilerde gradyanların aşırı küçülmesi (kaybolan "
     "gradyan) sorunu nedeniyle uzun vadeli bağımlılıkları öğrenmekte zorlanır.")

h3("2.5.2.", "Uzun-Kısa Süreli Bellek (LSTM)")
body("LSTM, kaybolan gradyan sorununu çözmek için geliştirilmiş bir RNN türüdür "
     "(Hochreiter ve Schmidhuber, 1997). Bir "
     "hücre durumu (cell state) ve üç geçit (giriş, unutma ve çıkış geçitleri) "
     "içerir. Bu geçitler, hangi bilginin belleğe yazılacağına, hangisinin silineceğine "
     "ve hangisinin çıkışa aktarılacağına karar verir. Bu sayede LSTM, uzun dizilerde "
     "bilgiyi daha iyi koruyabilir.")

h3("2.5.3.", "Geçitli Tekrarlayan Birim (GRU)")
body("GRU, LSTM'in daha sade bir alternatifidir (Cho ve diğ., 2014). Ayrı bir hücre "
     "durumu yerine yalnızca "
     "gizli durumu kullanır ve iki geçit içerir: güncelleme geçidi (update gate) ve "
     "sıfırlama geçidi (reset gate). Güncelleme geçidi, önceki bilginin ne kadarının "
     "korunacağını; sıfırlama geçidi ise yeni girdiyle geçmişin ne ölçüde "
     "birleştirileceğini belirler. GRU da LSTM gibi geçit mekanizmasına sahip olduğu "
     "için bilgiyi seçici biçimde tutabilir; yani \"unutmadan\" çok, neyi unutup neyi "
     "tutacağına karar verebilir.")

h3("2.5.4.", "GRU ve LSTM Karşılaştırması")
body("GRU, LSTM'e göre daha az geçit ve dolayısıyla daha az parametre içerir. Bu durum "
     "iki önemli avantaj sağlar: birincisi, daha az parametre küçük veri kümelerinde "
     "aşırı öğrenme (overfitting) riskini azaltır; ikincisi, daha az hesaplama "
     "gerektirdiği için eğitim ve çıkarım daha hızlıdır; bu da gerçek zamanlı ve mobil "
     "kullanım için kritiktir. Pek çok çalışmada GRU ve LSTM'in benzer doğruluklar "
     "verdiği gösterilmiştir (Chung ve diğ., 2014). Bu çalışmada, veri setinin "
     "büyüklüğü ve gerçek zamanlılık "
     "ihtiyacı göz önüne alınarak GRU tercih edilmiştir.")

h2("2.6.", "Doğal Dil İşleme")
body("Doğal dil işleme (NLP), insan dilinin bilgisayarlar tarafından işlenmesini konu "
     "alan alandır. Bu çalışmada NLP, tanınan ham işaret kelimelerini akıcı Türkçe "
     "cümleye dönüştürmek için kullanılır.")

h3("2.6.1.", "Paralel Veri Seti Sorunu")
body("Bir makine çeviri modeli eğitmenin standart yolu, kaynak ve hedef dildeki "
     "eşleşmiş örneklerden oluşan büyük bir paralel veri seti kullanmaktır. İşaret dili "
     "için bu, çok sayıda \"işaret dizisi - Türkçe cümle\" çiftinden oluşan bir veri "
     "seti anlamına gelir. Ancak Türk İşaret Dili için bu ölçekte, kamuya açık ve "
     "yeterince büyük bir paralel veri seti bulunmamaktadır. Böyle bir veri setini "
     "sıfırdan oluşturmak, bu bitirme ödevinin kapsamını aşan, ayrı ve büyük bir "
     "çalışmadır.")

h3("2.6.2.", "Kural Tabanlı Yaklaşım")
body("Paralel veri eksikliği nedeniyle bu çalışmada istatistiksel/öğrenmeli bir çeviri "
     "modeli yerine kural tabanlı bir NLP katmanı tercih edilmiştir. Bu katman, Türkçe "
     "dilbilgisi kurallarını (ek getirme, fiil çekimi, özne tespiti, bağlama göre "
     "düzeltme vb.) elle kodlanmış kurallar biçiminde uygular. Kural tabanlı yaklaşım, "
     "sınırlı sözlükte öngörülebilir ve denetlenebilir sonuçlar verir; ayrıca veri "
     "gerektirmez. Dezavantajı, her yeni yapı için yeni kural yazılması gerekmesidir.")

h2("2.7.", "Ses Teknolojileri")
body("Metinden sese (TTS) teknolojisi, yazılı metni doğal bir insan sesine "
     "dönüştürür; sesten metne (STT) teknolojisi ise konuşmayı yazıya çevirir. "
     "SignBridge'de TTS, tanınan ve cümleye dönüştürülen işaretleri sesli okumak için; "
     "STT ise işiten kullanıcının konuşmasını yazıya çevirip avatara aktarmak için "
     "kullanılır. Web tarafında tarayıcının yerleşik Web Speech API'si, mobil tarafta "
     "ise flutter_tts ve speech_to_text paketleri kullanılmıştır.")

h2("2.8.", "Üç Boyutlu Karakter Animasyonu")
body("Türkçe→işaret yönünde, işaretlerin görsel olarak gösterilmesi için üç boyutlu "
     "insansı bir karakter (avatar) kullanılır. Üç boyutlu karakterler, bir iskelet "
     "(armature) ve bu iskelete bağlı kemiklerden (bone) oluşur. Her kemiğin uzaydaki "
     "dönüşü değiştirilerek karakterin duruşu (poz) belirlenir. Bir işaret, ilgili "
     "kemiklerin belirli dönüş açılarına getirilmesiyle elde edilir. Bu çalışmada "
     "avatar, web tarayıcısında çalışan Three.js kütüphanesiyle gerçeklenmiştir "
     "(Three.js, 2023).")

h2("2.9.", "Model Dağıtımı ve ONNX")
body("Bir derin öğrenme modeli genellikle PyTorch veya TensorFlow gibi bir çatıda "
     "eğitilir. Ancak modelin mobil cihaz gibi farklı bir ortamda çalıştırılması "
     "gerektiğinde, taşınabilir bir formata dönüştürülmesi gerekir. ONNX (Open Neural "
     "Network Exchange), farklı çatılar arasında model taşınabilirliği sağlayan açık "
     "bir standarttır (ONNX, 2019). Bu çalışmada, PyTorch ile eğitilen GRU modeli ONNX "
     "formatına "
     "dönüştürülerek mobil uygulamada cihaz üzerinde (internet olmadan) "
     "çalıştırılmıştır.")
page_break()

# ============================================================ 3. MATERYAL VE YONTEM
h1("3", "MATERYAL VE YÖNTEM")

h2("3.1.", "Genel Sistem Mimarisi")
body("SignBridge, çift yönlü çalışan modüler bir mimariye sahiptir. Sistem iki ana "
     "akıştan oluşur. İşaret→Türkçe akışında: kamera görüntüsü alınır, MediaPipe "
     "Holistic ile anahtar noktalar çıkarılır, öznitelik vektörü oluşturulur ve "
     "normalleştirilir, GRU modeli kelime tahmini yapar, tahmin yumuşatma (smoothing) "
     "ile kararlı kelimeler belirlenir, kurala dayalı NLP katmanı bu kelimeleri akıcı "
     "Türkçe cümleye dönüştürür ve sonuç hem ekrana yazılır hem de TTS ile seslendirilir.")
body("Türkçe→İşaret akışında ise: kullanıcı Türkçe metni yazar veya mikrofona "
     "konuşur (STT), metin kelimelere ayrılır ve her kelimenin kökü bulunur, her kelime "
     "için önceden kaydedilmiş işaret pozu sözlükten getirilir ve üç boyutlu avatar bu "
     "pozları sırayla oynatarak işareti gösterir. Şekil 3.1'de bu iki yönlü akış "
     "şematik olarak gösterilmiştir.")
add_image_abs(ART + r"\architecture_diagram.png", 16.0)
caption("Şekil 3.1. SignBridge sisteminin genel mimarisi ve çift yönlü veri akışı")

h2("3.2.", "Kullanılan Yazılım ve Araçlar")
body("Sistemin sunucu/web tarafı Python diliyle geliştirilmiştir. Web çatısı olarak "
     "Flask, gerçek zamanlı çift yönlü iletişim için Socket.IO kullanılmıştır. Gerçek "
     "zamanlı kamera işlemede WebSocket kararlılığı için Werkzeug 2.3.8 sürümü tercih "
     "edilmiş ve sunucu iş parçacığı (threading) tabanlı çalışacak biçimde "
     "yapılandırılmıştır. Anahtar nokta çıkarımı MediaPipe ile, görüntü işleme OpenCV "
     "ile yapılmıştır (Bradski, 2000).")
body("Derin öğrenme modeli PyTorch ile eğitilmiş (Paszke ve diğ., 2019), sayısal "
     "işlemler NumPy ile gerçekleştirilmiştir. Model, mobil dağıtım için ONNX formatına "
     "dönüştürülmüştür. "
     "Mobil uygulama Flutter/Dart ile Android için geliştirilmiştir. Üç boyutlu avatar, "
     "web tarayıcısında Three.js kütüphanesiyle gerçeklenmiştir. Tablo 3.4'te ve ekte "
     "bileşenlerin teknoloji eşlemesi özetlenmiştir.")

h2("3.3.", "Veri Seti Oluşturma")

h3("3.3.1.", "Sözlüğün Belirlenmesi")
body("İşaret tanıma modeli için günlük yaşamda sık kullanılan kelimelerden oluşan 122 "
     "kelimelik bir sözlük belirlenmiştir. Sözlük; aile bireyleri, sayılar, haftanın "
     "günleri, ayların adları, selamlaşma ve nezaket ifadeleri, sık kullanılan fiiller, "
     "soru sözcükleri, yiyecek-içecekler, meslekler, yerler, nesneler ve sıfatlar gibi "
     "geniş bir yelpazeyi kapsayacak biçimde seçilmiştir. Sözlüğün kategorilere göre "
     "dağılımı Tablo 3.1'de, tam listesi ise EK-1'de (Tablo E.1) verilmiştir.")
caption("Tablo 3.1. 122 kelimelik sözlüğün kategorilere göre dağılımı")
make_table(["Kategori", "Örnek Kelimeler", "Yaklaşık Adet"], [
    ["Aile ve kişiler", "ANNE, BABA, ABİ, ABLA, AMCA, AKRABA, ARKADAŞ, BEN, SEN, KENDİ", "10"],
    ["Sayılar", "SIFIR, BİR, İKİ, ÜÇ, DÖRT, BEŞ, ALTI, YEDİ, SEKİZ, DOKUZ", "10"],
    ["Haftanın günleri", "PAZARTESİ, SALI, ÇARŞAMBA, PERŞEMBE, CUMA, CUMARTESİ, PAZAR", "7"],
    ["Aylar", "OCAK, ŞUBAT, MART, NİSAN, MAYIS, HAZİRAN, TEMMUZ, AĞUSTOS, EYLÜL, EKİM, KASIM, ARALIK", "12"],
    ["Zaman", "GÜN, SAAT, SONRA, SABAH, AKŞAM, BUGÜN, DÜN, KIŞ", "8"],
    ["Fiiller", "BAKMAK, BİTMEK, BULUŞMAK, ÇALIŞMAK, DOYMAK, EVLENMEK, GÖRÜŞMEK, İÇMEK, YAPMAK, YATMAK, YEMEK, YORULMAK, BOŞ VERMEK", "13"],
    ["Soru sözcükleri", "NE, KAÇ, NEREDE, HANGİ", "4"],
    ["Yiyecek-içecek", "AYRAN, BAKLAVA, ÇAY, ÇİKOLATA, EKMEK, ERİK, KAHVE, SU, YEMEK, ORUÇ", "10"],
    ["Yerler", "EV, OKUL, PARK, KAFE, TUVALET, YAN", "6"],
    ["Meslek", "DOKTOR, HEMŞİRE, ÖĞRETMEN", "3"],
    ["Nesne/teknoloji", "BİLGİSAYAR, CEP TELEFONU, TV, CD, FİNAL", "5"],
    ["Spor", "FUTBOL, HENTBOL, VOLEYBOL", "3"],
    ["Sıfat/durum", "AKILLI, APTAL, BÜYÜK, KÜÇÜK, ÇOK, BOŞ, ZENGİN, İYİ, KÖTÜ", "9"],
    ["Selamlaşma/nezaket", "MERHABA, NASILSIN, TEŞEKKÜR, HOŞÇA KAL, AFERİN, TAMAM, EVET, AYIP, DİKKAT, YETER, BU KADAR, YARDIM", "12"],
    ["Kültürel/dini/diğer", "ALLAH, AMİN, ELHAMDÜLİLLAH, ATATÜRK, AŞK, BARIŞ, ÇÜNKÜ, BERABER, CEZA, ALERJİ, ARAP", "11"],
])

body("Veri toplama, MediaPipe'ın çıkardığı anahtar noktaların canlı olarak görüntü "
     "üzerine bindirildiği bir arayüz üzerinden yapılmıştır (Şekil 3.2). Bu sayede her "
     "kaydın doğru biçimde alınıp alınmadığı anlık olarak denetlenebilmiştir.")
add_image_abs(REP + r"\veri_toplama.png", 14.0)
caption("Şekil 3.2. Veri toplama arayüzünde MediaPipe ile çıkarılan anahtar noktalar")

h3("3.3.2.", "\"Dizi\" Kavramı")
body("İşaret dili hareket içerdiğinden, tek bir fotoğraf bir kelimeyi anlatmaya "
     "yetmez. Bu nedenle bir kelime işaret edildiğinde, kamera yaklaşık bir saniye "
     "boyunca 30 ardışık kare yakalar. Bu 30 kare birlikte bir \"dizi\" (sequence) "
     "oluşturur. Yani bir dizi, bir kelimenin tek bir gösterimine karşılık gelir ve "
     "hareketin film şeridi gibi kaydıdır. Modelin girdisi tek bir görüntü değil, bu "
     "30 karelik dizidir; böylece model, hareketin zaman içindeki değişimini öğrenebilir.")
add_image_abs(GEN + r"\fig_3_3_dizi.png", 15.0)
caption("Şekil 3.3. Bir işaret dizisinin 30 kareye bölünmesi")

h3("3.3.3.", "Veri Toplama Süreci")
body("Her kelime, kamera önünde defalarca işaret edilerek kaydedilmiştir. Her kayıt, "
     "MediaPipe ile çıkarılan anahtar noktaların 30 karelik bir dizisi olarak bir .npy "
     "dosyasına yazılmıştır. Toplamda 14.974 adet ham örnek dizisi toplanmıştır. Kelime "
     "başına örnek sayısı sabit değildir; ortanca değer kelime başına yaklaşık 150 örnek "
     "olup, kelimelere göre 50 ile 300 arasında değişmektedir. Kelimelerin zorluğuna ve "
     "modelin ilk denemelerdeki başarımına göre bazı kelimeler için daha fazla örnek "
     "toplanmıştır.")

h3("3.3.4.", "Veri Saklama ve Gizlilik")
body("Önemli bir tasarım kararı olarak, veri setinde hiçbir ham fotoğraf veya video "
     "saklanmamaktadır. Yalnızca MediaPipe'ın çıkardığı anahtar nokta koordinatları "
     "(sayısal değerler) saklanır. Bu yaklaşımın iki önemli faydası vardır: birincisi, "
     "dosya boyutları çok küçük kalır ve veri seti verimli biçimde işlenebilir; "
     "ikincisi, kişinin yüzü veya görüntüsü hiçbir aşamada kaydedilmediği için "
     "gizliliği korunur. Bu durum, gerçek kullanıcılarla yapılacak ileride çalışmalar "
     "açısından da etik bir avantaj sağlar.")

h2("3.4.", "Veri Artırma (Data Augmentation)")
body("Sınırlı sayıdaki ham örnekle modelin genelleme yeteneğini artırmak için altı "
     "farklı veri artırma tekniği uygulanmıştır. Her teknik, mevcut bir örnekten yeni "
     "ama gerçekçi varyantlar üretir; böylece model farklı koşullara karşı dayanıklı "
     "hâle gelir. Her ham örnekten birden çok varyant üretilerek veri seti 14.974 "
     "örnekten 105.182 örneğe çıkarılmıştır. Teknikler Tablo 3.2'de açıklanmıştır.")
caption("Tablo 3.2. Uygulanan veri artırma teknikleri")
make_table(["Teknik", "Açıklama", "Amaç"], [
    ["Gürültü (noise)", "Koordinatlara küçük rastgele sapmalar eklenir.", "Ölçüm gürültüsüne dayanıklılık"],
    ["Ölçekleme (scale)", "İşaret büyüklüğü hafifçe büyütülüp küçültülür.", "Kişi boyu/uzaklık farkları"],
    ["Öteleme (translate)", "Tüm noktalar çerçeve içinde kaydırılır.", "Kişinin konum farkı"],
    ["Döndürme (rotate)", "Koordinatlar küçük açılarla döndürülür.", "Kamera açısı farkları"],
    ["Zaman bükme (timewarp)", "Hareketin hızı hafifçe değiştirilir.", "Farklı işaret hızları"],
    ["Aynalama (mirror)", "Sağ-sol simetri uygulanır.", "Sağ/sol elli kullanım"],
])
body("Veri artırmanın yalnızca eğitim verisine uygulandığı, doğrulama verisinin ise "
     "gerçek (artırılmamış) örneklerden oluşacak biçimde ayrıldığına dikkat edilmiştir. "
     "Aksi hâlde, bir örneğin hem kendisinin hem de varyantının farklı kümelere düşmesi "
     "doğrulama doğruluğunu yapay olarak yükseltebilirdi.")

h2("3.5.", "Öznitelik Çıkarımı ve Normalleştirme")
body("Her kare için MediaPipe Holistic'ten elde edilen 543 anahtar noktanın x, y, z "
     "koordinatları kullanılır; bu, kare başına 543 × 3 = 1629 değer demektir. Bunlara "
     "ek olarak, ardışık kareler arasındaki el noktalarının yer değiştirmesini ifade "
     "eden 126 boyutlu bir hız (velocity) bileşeni hesaplanır. Hız bilgisi, hareketin "
     "yönü ve hızını açıkça modele aktararak benzer duruşlara sahip ama farklı "
     "hareketli işaretlerin ayırt edilmesine yardımcı olur. Sonuçta her kare 1629 + "
     "126 = 1755 boyutlu bir öznitelik vektörüyle temsil edilir. Bileşenler Tablo "
     "3.3'te özetlenmiştir.")
caption("Tablo 3.3. Öznitelik vektörünün bileşenleri")
make_table(["Bileşen", "Nokta Sayısı", "Boyut"], [
    ["Yüz koordinatları", "468", "1404"],
    ["Vücut (poz) koordinatları", "33", "99"],
    ["Sol el koordinatları", "21", "63"],
    ["Sağ el koordinatları", "21", "63"],
    ["Ara toplam (konum)", "543", "1629"],
    ["El hız bileşeni (velocity)", "—", "126"],
    ["Toplam (kare başına)", "—", "1755"],
])
body("Kameraya olan uzaklıktan ve kişinin çerçevedeki konumundan bağımsız tanıma "
     "yapabilmek için iki aşamalı bir normalleştirme uygulanır. Birinci aşamada tüm "
     "noktalar omuz merkezine göre yeniden konumlandırılır ve ölçeklenir; böylece "
     "kişinin kameraya yakın ya da uzak olması fark etmez. İkinci aşamada el noktaları "
     "bilek merkezine göre yeniden ölçeklenir; böylece elin görüntüdeki büyüklüğü değil, "
     "parmakların birbirine göre durumu öne çıkar. Bu iki aşamalı normalleştirme, aynı "
     "işaretin farklı koşullarda benzer öznitelik değerleri üretmesini sağlar.")
add_image_abs(GEN + r"\fig_3_4_normallestirme.png", 15.0)
caption("Şekil 3.4. İki aşamalı normalleştirme (omuz ve bilek merkezli)")
body("Mobil uygulamada dikey (portre) çekim ile eğitimdeki yatay görüntü en-boy oranı "
     "arasındaki farktan kaynaklanan bir sapma gözlenmiştir. Bu sorun, görüntü "
     "MediaPipe'a verilmeden önce 4:3 oranına tamamlanacak biçimde kenar dolgusu "
     "(padding) eklenerek giderilmiş ve mobildeki tanıma, eğitimdeki koşullarla "
     "uyumlu hâle getirilmiştir.")

h2("3.6.", "Model Mimarisi")
body("Sınıflandırma modeli, iki katmanlı ve katman başına 256 gizli birimli bir GRU "
     "ağıdır. Giriş olarak 1755 boyutlu kare vektörlerinden oluşan bir dizi alır. "
     "GRU'nun ürettiği son gizli durum, 128 nöronlu tam bağlı bir katmana, ardından "
     "ReLU aktivasyon fonksiyonuna ve son olarak 122 sınıflı çıkış katmanına verilir. "
     "Çıkış katmanı, her kelime için bir olasılık değeri üretir ve en yüksek olasılıklı "
     "kelime tahmin olarak seçilir. Model, değişken uzunluktaki dizilerle "
     "çalışabilmektedir; bu, gerçek zamanlı kullanımda tampondaki kare sayısı 30'a "
     "ulaşmadan da tahmin üretilebilmesini sağlar. Mimari Tablo 3.4'te ve Şekil 3.5'te "
     "gösterilmiştir.")
caption("Tablo 3.4. Model mimarisi ve eğitim hiperparametreleri")
make_table(["Parametre", "Değer"], [
    ["Model türü", "GRU (tekrarlayan sinir ağı)"],
    ["Gizli birim sayısı", "256"],
    ["GRU katman sayısı", "2"],
    ["Tam bağlı katman", "Linear(256→128) → ReLU → Linear(128→122)"],
    ["Giriş boyutu (kare başına)", "1755"],
    ["Dizi uzunluğu", "30 kare (değişken uzunluk desteklenir)"],
    ["Sınıf sayısı", "122"],
    ["Eğitim/doğrulama ayrımı", "%80 / %20"],
    ["Eniyileyici (optimizer)", "Adam"],
    ["Kayıp fonksiyonu", "Çapraz entropi (Cross-Entropy)"],
])
add_image_abs(GEN + r"\fig_3_5_model.png", 9.5)
caption("Şekil 3.5. GRU tabanlı sınıflandırma modelinin katman yapısı")

h2("3.7.", "Model Eğitimi")
body("Eğitim sırasında veri %80 eğitim ve %20 bağımsız doğrulama olarak ayrılmıştır. "
     "Model, çapraz entropi kayıp fonksiyonu ve Adam eniyileyici (Kingma ve Ba, 2015) "
     "kullanılarak eğitilmiştir. Her devir (epoch) sonunda doğrulama kümesindeki "
     "doğruluk ölçülmüş ve "
     "en iyi doğrulama doğruluğunu veren model ağırlıkları kaydedilmiştir (best_model). "
     "Bu yaklaşım, modelin aşırı öğrenmeye başladığı noktayı geçmeden en iyi genelleme "
     "yapan hâlinin saklanmasını sağlar.")
body("Bağımsız doğrulama doğruluğu, modelin eğitim sırasında hiç görmediği örnekler "
     "üzerinde ölçülen başarımı ifade eder. Bu ölçüt, modelin veriyi ezberleyip "
     "ezberlemediğini değil, gerçekten genelleme yapıp yapmadığını gösterdiği için "
     "önemlidir. Eğitim ve doğrulama doğruluk eğrileri Şekil 4.1'de sunulmuştur.")
page_break()

h2("3.8.", "Gerçek Zamanlı Tahmin ve Yumuşatma")
body("Gerçek zamanlı kullanımda, kameradan gelen kareler sürekli olarak MediaPipe ile "
     "işlenir ve öznitelik vektörleri bir tampona (buffer) biriktirilir. Tamponda "
     "yeterli sayıda kare (en az 15) biriktiğinde model tahmin üretmeye başlar. Ham "
     "tahminler kare kare dalgalanabileceğinden, yanlış ve titrek sonuçları engellemek "
     "için bir güven eşiği ve tahmin yumuşatma (smoothing) mekanizması uygulanır.")
body("Bir kelimenin kabul edilebilmesi için modelin o kelimeye verdiği güven değerinin "
     "belirli bir eşiği aşması ve aynı tahminin ardışık karelerde yeterince "
     "tekrarlanması gerekir. 122 sınıflı bir problemde kelimeler arası karışma riski "
     "daha yüksek olduğundan, eşik değerleri karışmayı azaltacak biçimde sıkı "
     "tutulmuştur. Çok yüksek güvenli tahminler için ise daha hızlı kabul sağlayan bir "
     "\"hızlı izleme\" (fast-track) yolu tanımlanmıştır. Ayrıca aynı kelimenin arka "
     "arkaya tekrar yazılmasını önlemek için bir bekleme (cooldown) süresi "
     "kullanılmıştır. Bu değerler Tablo 3.5'te verilmiştir.")
caption("Tablo 3.5. Gerçek zamanlı tahmin eşik değerleri")
make_table(["Parametre", "Değer", "Açıklama"], [
    ["Güven eşiği", "0,88", "Kabul için gereken minimum güven"],
    ["Hızlı izleme eşiği", "0,95", "Tek karede kabul için yüksek güven"],
    ["Yumuşatma tekrarı", "2 kare", "Normal kabul için ardışık tekrar"],
    ["Hızlı yumuşatma", "1 kare", "Yüksek güvende ardışık tekrar"],
    ["Bekleme süresi (cooldown)", "0,4 sn", "Aynı kelimenin tekrarını önler"],
    ["Minimum tampon", "15 kare", "Tahmine başlamak için gereken kare"],
])
body("Bu mekanizma, web ve mobil platformlarda aynı parametrelerle uygulanarak iki "
     "platform arasında tutarlı bir davranış sağlanmıştır. Akış Şekil 3.6'da "
     "gösterilmiştir.")
add_image_abs(GEN + r"\fig_3_6_akis.png", 15.0)
caption("Şekil 3.6. Gerçek zamanlı tahmin ve yumuşatma akış şeması")

h2("3.9.", "Kural Tabanlı Doğal Dil İşleme Katmanı")
body("Tanınan kelimeler ham gloss biçimindedir (örneğin BERABER KAHVE İÇMEK). Bu "
     "dizinin akıcı Türkçeye dönüştürülmesi için kurala dayalı bir NLP katmanı "
     "geliştirilmiştir. Bu katman; fiil çekim tabloları, ek (sonek) işleme, özne tespiti "
     "(BEN/SEN), sayı + SONRA yapısının ablatif çekimi ve bağlama göre düzeltme "
     "kuralları içerir. Katman, hem web tarafında (JavaScript) hem de mobil tarafta "
     "(Dart) aynı mantıkla uygulanmıştır.")
body("Örnek kurallar şunlardır: BERABER kelimesi varsa fiil istek/öneri kipine "
     "çevrilir (\"içelim mi?\"); KENDİ kelimesi varsa fiil emir kipine getirilir "
     "(\"bak\"); cümlede BEN veya SEN varsa özne tespit edilip fiil uygun şahısta "
     "çekilir; sayı ile SONRA birlikte geçtiğinde ablatif ek uygulanır (\"beşten "
     "sonra\"); KAHVE ile KAFE karışıklığı bağlamdan düzeltilir (BERABER bağlamında "
     "KAHVE, YAN/PARK bağlamında KAFE). Bazı örnek dönüşümler Tablo 3.6'da verilmiştir.")
caption("Tablo 3.6. Kural tabanlı NLP katmanı örnek dönüşümleri")
make_table(["Tanınan kelimeler (gloss)", "Üretilen Türkçe cümle"], [
    ["BERABER KAHVE İÇMEK", "Beraber kahve içelim mi?"],
    ["BEŞ SAAT SONRA BULUŞMAK", "Beşten sonra buluşalım."],
    ["BEN OKUL GİTMEK", "Ben okula gidiyorum."],
    ["KENDİ BAKMAK", "Kendine iyi bak."],
    ["SEN NASILSIN", "Sen nasılsın?"],
])

h2("3.10.", "Üç Boyutlu Avatar")
body("Türkçe→işaret yönünde, kullanıcının yazdığı ya da mikrofonla söylediği Türkçe "
     "ifade önce kelimelere ayrılır ve her kelimenin kökü bulunur. Kök bulma sırasında, "
     "kullanıcı \"bak\" yazsa bile sistem bunu sözlükteki mastar biçimiyle (BAKMAK) "
     "eşleştirir. Türkçenin sondan eklemeli yapısı nedeniyle ekler tekrarlı biçimde "
     "soyularak kelime köküne ulaşılır; ayrıca yakın eşleşme için bir benzerlik "
     "araması yapılır. Bu sırada yanlış eşleşmeleri önlemek için mastar eşleşmesine "
     "öncelik verilir (örneğin \"bak\" kelimesinin yanlışlıkla BAKLAVA ile eşleşmesi "
     "engellenmiştir).")
body("Bulunan her kelime için önceden kaydedilmiş işaret pozu sözlükten getirilir ve "
     "avatar bu pozları sırayla oynatır. Avatar, insansı bir model (rain.glb) kullanır "
     "ve Three.js ile gerçeklenmiştir. İşaret pozları, modelin kemiklerinin (bone) x, "
     "y, z dönüş açıları olarak saved_poses.json dosyasında saklanır. Bu dosyada 123 "
     "kelime için poz tanımı bulunmaktadır; her poz, ilgili kemiklerin dönüş "
     "değerlerinden oluşur (örneğin bir işaret 21 kemiğin belirli açılara getirilmesiyle "
     "tanımlanabilir). Bir işaret kaydedilirken avatar elle istenen duruşa getirilir ve "
     "kemik açıları bu dosyaya yazılır.")
two_images_row_abs(REP + r"\poz_editoru.png", REP + r"\web_avatar_baba.png", width_cm=7.4)
caption("Şekil 3.7. Üç boyutlu avatarın poz editörü (solda: kemik eksen "
        "denetimleri) ve kaydedilmiş bir işaretin oynatılması (sağda: BABA işareti)")

h2("3.11.", "Ses Bileşenleri")
body("İşaret→Türkçe yönünde oluşturulan cümle, metinden sese (TTS) teknolojisiyle "
     "sesli okunur. Web tarafında tarayıcının Web Speech API'si, mobil tarafta "
     "flutter_tts paketi kullanılır. Cümle tamamlandığında otomatik seslendirme "
     "tetiklenir. Ters yönde ise kullanıcının konuşması sesten metne (STT) ile yazıya "
     "çevrilir; web tarafında webkitSpeechRecognition, mobil tarafta speech_to_text "
     "kullanılır. Böylece sistem hem yazılı hem de sesli olarak kullanılabilir; "
     "özellikle işiten kullanıcı için mikrofonla konuşup avatara aktarma kolaylık "
     "sağlar.")

h2("3.12.", "Web Uygulaması")
body("Web uygulaması, tek bir portal üzerinden erişilen iki sekmeden oluşur: kamera "
     "(işaret tanıma) ve avatar (Türkçe→işaret) sekmeleri. Kamera sekmesinde kullanıcı "
     "kamerasını açar, işaret yapar ve tanınan kelimeler ile oluşturulan cümle ekranda "
     "görüntülenip seslendirilir. Avatar sekmesinde ise kullanıcı metin yazar veya "
     "mikrofonla konuşur ve avatar işareti gösterir; ayrıca hazır günlük cümlelerden "
     "seçim yapılabilen bir hızlı erişim çubuğu bulunur. Sunucu, kameradan gelen "
     "kareleri WebSocket üzerinden alır, MediaPipe ve GRU ile işler ve sonucu istemciye "
     "geri gönderir.")

h2("3.13.", "Mobil Uygulama")
body("Sistemin taşınabilir ve yaygın kullanılabilir olması amacıyla, web "
     "uygulamasıyla aynı yetenekleri sunan bir Android mobil uygulaması "
     "geliştirilmiştir. Uygulama, Flutter/Dart platformu kullanılarak yazılmıştır "
     "(Google, 2023). "
     "Flutter, tek bir kod tabanından yerel (native) performansa yakın uygulamalar "
     "üretebilmesi ve zengin arayüz bileşenleri sunması nedeniyle tercih edilmiştir.")

h3("3.13.1.", "Uygulama Yapısı ve Ekranlar")
body("Uygulamada iki ana ekran bulunur. Kamera ekranı (camera_screen.dart) işaret→"
     "Türkçe yönünü; avatar ekranı (avatar_screen.dart) ise Türkçe→işaret yönünü "
     "gerçekler. Kamera ekranında kullanıcı, ön ve arka kamera arasında geçiş yapabilir "
     "(kamera değişiminde kaynakların güvenli biçimde serbest bırakılması sağlanmıştır), "
     "demo modunu açıp kapatabilir ve tanınan kelimelerin canlı olarak ekrana "
     "yazılmasını izleyebilir. Tanınan kelimeler, web ile aynı mantığı uygulayan bir "
     "doğal dil işleme servisi (nlp_service.dart) ile akıcı Türkçe cümleye dönüştürülür "
     "ve seslendirilir. Avatar ekranında ise kullanıcı metin yazabilir veya mikrofonla "
     "konuşabilir; girilen ifade işaret diliyle gösterilir. Ayrıca günlük cümlelerden "
     "oluşan hazır ifade düğmeleri bulunur.")

h3("3.13.2.", "Cihaz Üzerinde Çıkarım (ONNX)")
body("Modelin mobilde çalışabilmesi için PyTorch modeli ONNX formatına "
     "dönüştürülmüştür. Anahtar nokta çıkarımı için MediaPipe, sınıflandırma için ise "
     "ONNX modeli doğrudan cihaz üzerinde çalıştırılmaktadır. Böylece uygulama, internet "
     "bağlantısı olmadan tamamen çevrimdışı tahmin yapabilmektedir; bu, hem gizlilik "
     "(görüntünün cihazdan çıkmaması) hem de erişilebilirlik (her ortamda çalışma) "
     "açısından önemli bir avantajdır. Doğru sınıf etiketlerinin kullanılabilmesi için "
     "122 sınıflık etiket eşlemesi (label_map.json) modelle birlikte uygulamaya "
     "gömülmüştür.")
body("Gerçek zamanlı tahmin için kullanılan güven eşiği, hızlı izleme eşiği, yumuşatma "
     "tekrarı ve bekleme süresi gibi parametreler web tarafıyla bire bir aynı tutulmuş; "
     "böylece iki platformda tutarlı bir davranış sağlanmıştır. Kamera kare işleme hızı, "
     "cihazı yormayacak ve gecikme oluşturmayacak biçimde yaklaşık 20 FPS hedeflenecek "
     "şekilde ayarlanmıştır. Tanınan kelimeler, otomatik seslendirme (flutter_tts) ile "
     "okunur; avatar ekranındaki mikrofon, konuşmayı yazıya çeviren sesten metne "
     "(speech_to_text) bileşenini kullanır.")

h3("3.13.3.", "Geliştirme Süreci ve Çözülen Sorunlar")
body("Mobil uygulamanın geliştirilmesinde, sunucu ile cihaz arasında işin nasıl "
     "paylaştırılacağına ilişkin aşamalı bir yaklaşım izlenmiştir. İlk aşamada cihaz "
     "yalnızca kamera görüntüsünü sunucuya gönderip tüm işlemi sunucuya bırakırken, ara "
     "aşamada anahtar nokta çıkarımı sunucuda, sınıflandırma cihazda yapılmış; son "
     "aşamada ise hem MediaPipe hem de ONNX modeli doğrudan cihaz üzerinde "
     "çalıştırılarak tam çevrimdışı çalışmaya geçilmiştir. Bu sayede ağ gecikmesi "
     "ortadan kalkmış ve tepki süresi iyileşmiştir.")
body("Geliştirme sürecinde iki önemli sorun çözülmüştür. Birincisi, mobil kameranın "
     "dikey (portre) görüntüsü ile modelin eğitildiği yatay görüntünün en-boy oranı "
     "farkından kaynaklanan tanıma sapmasıdır; bu sorun, görüntü MediaPipe'a "
     "verilmeden önce 4:3 oranına kenar dolgusu (padding) eklenerek giderilmiştir. "
     "İkincisi, başlangıçta uygulamaya gömülen etiket eşlemesinin yalnızca 100 sınıf "
     "içermesinden kaynaklanan bir uyumsuzluktur; etiket eşlemesi 122 sınıfı kapsayacak "
     "biçimde yeniden üretilerek modelle birebir hizalanmış ve uygulama yeniden "
     "derlenmiştir.")
body("Uygulama, sürüm (release) modunda derlenerek dağıtılabilir bir APK paketi "
     "olarak üretilmiştir. Böylece sistem, bir geliştirme ortamına ihtiyaç duymadan "
     "doğrudan bir Android cihaza kurulabilmektedir.")
two_images_row("phone_mobil_kamera_merhaba.png", "phone_mobil_avatar_merhaba.png", width_cm=6.2)
caption("Şekil 3.8. Mobil uygulama kamera (solda) ve avatar (sağda) ekranları")

h2("3.14.", "Demo Modu")
body("Tanıtım ve gösterim sırasında kelime karışmasını azaltmak için her iki "
     "platformda da bir demo modu eklenmiştir. Bu modda, önceden belirlenmiş 10 örnek "
     "günlük cümlede geçen yaklaşık 28 kelimeden oluşan bir beyaz liste (whitelist) "
     "etkinleştirilir ve modelin tahminleri yalnızca bu kelimelerle sınırlandırılır. "
     "Böylece olası 122 kelime yerine küçük bir alt küme içinden seçim yapılır ve "
     "gösterim sırasında daha kararlı, öngörülebilir sonuçlar elde edilir. Demo modu, "
     "kullanıcı tarafından bir düğmeyle açılıp kapatılabilir.")
page_break()

# ============================================================ 4. BULGULAR
h1("4", "BULGULAR")

h2("4.1.", "Eğitim ve Doğrulama Başarımı")
body("122 sınıflı GRU modeli, %80 eğitim ve %20 bağımsız doğrulama ayrımıyla "
     "eğitilmiştir. Eğitim ilerledikçe hem eğitim hem de doğrulama doğruluğu artmış ve "
     "model, %20'lik bağımsız doğrulama kümesinde %99,88 doğruluk değerine ulaşmıştır. "
     "Bu yüksek doğruluk, veri artırmanın ve iki aşamalı normalleştirmenin etkili "
     "olduğunu; modelin veriyi ezberlemek yerine genellenebilir örüntüler öğrendiğini "
     "göstermektedir. Eğitim ve doğrulama doğruluk eğrileri Şekil 4.1'de verilmiştir.")
add_image_abs(ART + r"\training_history_122.png", 16.0)
caption("Şekil 4.1. Eğitim ve doğrulama doğruluk eğrileri (final %99,88 doğruluk)")
caption("Tablo 4.1. Veri seti ve model başarım özeti")
make_table(["Ölçüt", "Değer"], [
    ["Kelime (sınıf) sayısı", "122"],
    ["Ham örnek dizisi sayısı", "14.974"],
    ["Veri artırma sonrası örnek sayısı", "105.182"],
    ["Örnek başına kare sayısı", "30"],
    ["Kare başına öznitelik boyutu", "1755"],
    ["Eğitim/doğrulama ayrımı", "%80 / %20"],
    ["Bağımsız doğrulama doğruluğu", "%99,88"],
])

body("Modelin kelimeler arası karışma davranışını incelemek için karışıklık matrisi "
     "(confusion matrix) oluşturulmuştur. Şekil 4.2'de en sık kullanılan 30 kelime için "
     "elde edilen matris görülmektedir. Köşegen üzerindeki yoğunluk, tahminlerin büyük "
     "çoğunluğunun doğru sınıfa düştüğünü; köşegen dışındaki değerlerin neredeyse "
     "sıfır olması ise kelimeler arası karışmanın çok düşük olduğunu göstermektedir.")
add_image_abs(ART + r"\confusion_matrix_122.png", 13.0)
caption("Şekil 4.2. En sık 30 kelime için karışıklık matrisi")

h2("4.2.", "Gerçek Zamanlı Çalışma")
body("Sistem hem web tarayıcısında hem de Android cihazında gerçek zamanlı olarak "
     "çalışmıştır. Kelime tahmini, tampon en az 15 kare biriktirdiğinde üretilmekte; "
     "güven eşiği ve yumuşatma sayesinde kararlı sonuçlar vermektedir. Kullanıcı bir "
     "kelimeyi işaret ettiğinde, sistem yaklaşık bir saniyelik hareketi değerlendirip "
     "kelimeyi ekrana yazmakta ve seslendirmektedir. Mobil uygulama, ONNX modeli "
     "sayesinde internet bağlantısı olmadan tamamen cihaz üzerinde çalışabilmektedir.")

h2("4.3.", "Çift Yönlü Çeviri ve Cümle Oluşturma")
body("Kural tabanlı NLP katmanı, tanınan ham kelime dizilerini akıcı Türkçe cümlelere "
     "dönüştürebilmiştir. Örneğin BERABER KAHVE İÇMEK dizisi \"Beraber kahve içelim "
     "mi?\" biçimine; sayı ve SONRA içeren diziler ise uygun ablatif çekimle "
     "cümlelere dönüştürülmüştür. Ters yönde avatar, yazılan veya söylenen Türkçe "
     "ifadeleri kelime kelime işaret diliyle göstermiş; kök bulma ve mastar eşleştirme "
     "sayesinde çekimli kelimeler de doğru işaretlere bağlanmıştır.")

h2("4.4.", "Web ve Mobil Platform Karşılaştırması")
body("Sistem iki platformda da tutarlı parametrelerle çalıştırılmıştır. Temel farklar "
     "ve ortak noktalar Tablo 4.2'de özetlenmiştir.")
caption("Tablo 4.2. Web ve mobil platform karşılaştırması")
make_table(["Özellik", "Web", "Mobil (Android)"], [
    ["Anahtar nokta çıkarımı", "Sunucuda MediaPipe", "Cihazda MediaPipe"],
    ["Model formatı", "PyTorch (.pt)", "ONNX"],
    ["Çalışma", "Sunucu gerektirir", "Çevrimdışı (cihaz üzerinde)"],
    ["TTS / STT", "Web Speech API", "flutter_tts / speech_to_text"],
    ["Güven eşiği", "0,88 / 0,95", "0,88 / 0,95"],
    ["Demo modu", "Var", "Var"],
])

h2("4.5.", "Karşılaşılan Sınırlılıklar")
body("Tüm kelimeler aynı kolaylıkta tanınmamaktadır. Bazı kelimelerde (örneğin YAPMAK) "
     "modelin güven değeri zaman zaman eşiğin altında kalmış ve bu kelimeler daha zor "
     "tanınmıştır. Bunun nedeni, ilgili kelimeye ait eğitim örneklerinin görece az ya "
     "da birbirine çok benzer olması ve kelimenin işaretinin başka işaretlere "
     "benzemesidir. Bu durum, ilgili kelimeler için daha fazla ve daha çeşitli veri "
     "toplanmasının gerektiğini göstermektedir.")
body("Ayrıca, eğitim verisinin büyük bölümünün tek bir kişi tarafından toplanmış "
     "olması, bağımsız doğrulama doğruluğunun farklı kişilerdeki gerçek başarımı bir "
     "miktar iyimser yansıtmasına yol açabilir. Farklı kişiler, farklı işaret hızları "
     "ve küçük biçim farklarıyla işaret yaptığında başarımın bir miktar düşmesi "
     "beklenir; bu, gelecekte çok kişili veri toplamayı gerekli kılan bir bulgudur.")
page_break()

# ============================================================ 5. TARTISMA VE SONUC
h1("5", "TARTIŞMA VE SONUÇ")
body("Bu çalışmada, Türk İşaret Dili ile Türkçe arasında çift yönlü ve gerçek zamanlı "
     "çeviri yapabilen SignBridge sistemi başarıyla geliştirilmiştir. Anahtar nokta "
     "tabanlı öznitelik çıkarımı ve GRU tabanlı dizi sınıflandırması ile 122 kelimelik "
     "sözlük üzerinde %99,88 bağımsız doğrulama doğruluğu elde edilmiştir. Sistem hem "
     "web hem de mobil platformda; mobilde ise internet bağlantısı gerektirmeden "
     "çalışabilmektedir.")
body("Elde edilen sonuçlar, sınırlı bir sözlük üzerinde işaret dili tanımanın gerçek "
     "zamanlı ve cihaz üzerinde uygulanabilir olduğunu göstermektedir. Anahtar nokta "
     "tabanlı yaklaşımın seçilmesi; düşük veri ihtiyacı, ışık ve arka plan "
     "değişimlerine dayanıklılık ve gizliliğin korunması açısından isabetli olmuştur. "
     "İki aşamalı normalleştirme ve altı tip veri artırma, küçük veriyle yüksek "
     "doğruluk elde edilmesinde belirleyici olmuştur. Kural tabanlı NLP katmanı, paralel "
     "veri seti bulunmayan TİD için pratik ve denetlenebilir bir cümle oluşturma çözümü "
     "sunmuştur.")
body("Çalışmanın başlıca sınırlılıkları şunlardır: sözlüğün 122 kelimeyle sınırlı "
     "olması, eğitim verisinin büyük ölçüde tek kişiden toplanmış olması, cümle çeviri "
     "katmanının istatistiksel bir model yerine kurallara dayanması ve mobil "
     "uygulamanın yalnızca Android platformunda gerçeklenmiş olmasıdır. Ayrıca bazı "
     "kelimelerin tanıma başarımı diğerlerine göre düşüktür.")
body("Gelecek çalışmalar için öneriler şu şekilde sıralanabilir: (i) sözlüğün farklı "
     "yaş, cinsiyet ve işaret biçimine sahip çok sayıda kişiden toplanan verilerle "
     "genişletilmesi ve çeşitlendirilmesi; (ii) gerçek işitme engelli bireylerle saha "
     "testleri yapılarak sistemin gerçek kullanım koşullarında değerlendirilmesi; (iii) "
     "yeterli paralel veri toplanması durumunda kural tabanlı NLP katmanının "
     "istatistiksel veya öğrenmeli bir çeviri modeliyle değiştirilmesi; (iv) sürekli "
     "(cümle düzeyinde) işaret tanımaya geçilmesi; (v) avatar hareketlerinin ve cümle "
     "yapısının daha doğal hâle getirilmesi; (vi) iOS desteğinin eklenmesi.")
body("Sonuç olarak SignBridge, yalnızca bir bitirme ödevi olmanın ötesinde, işitme "
     "engelli bireyler ile işiten bireyler arasındaki iletişimi kolaylaştırmayı "
     "amaçlayan, gerçekten çalışan ve geliştirilmeye açık bir araç ortaya koymuştur. "
     "Önerilen geliştirmelerle sistemin kapsamı ve gerçek hayattaki kullanılabilirliği "
     "artırılabilir.")
page_break()

# ============================================================ KAYNAKLAR
h1("", "KAYNAKLAR")
# Harvard Referans Teknigi: kaynaklar yazar soyadina gore ALFABETIK siralanir
# (kilavuz md.2.3.1). Metin icinde (Soyad, yil) bicimiyle atif yapilmistir.
refs = [
    "BRADSKI, G., 2000, The OpenCV Library, Dr. Dobb's Journal of Software Tools, "
    "25 (11), 120-125.",
    "CHO, K., VAN MERRIENBOER, B., GULCEHRE, C., BAHDANAU, D., BOUGARES, F., "
    "SCHWENK, H. and BENGIO, Y., 2014, Learning Phrase Representations using RNN "
    "Encoder-Decoder for Statistical Machine Translation, Proceedings of EMNLP 2014, "
    "Doha, Qatar, 1724-1734.",
    "CHUNG, J., GULCEHRE, C., CHO, K. and BENGIO, Y., 2014, Empirical Evaluation of "
    "Gated Recurrent Neural Networks on Sequence Modeling, arXiv preprint "
    "arXiv:1412.3555.",
    "GOOGLE, 2023, Flutter Documentation [online], https://docs.flutter.dev "
    "[Ziyaret Tarihi: 31 Mayıs 2026].",
    "HOCHREITER, S. and SCHMIDHUBER, J., 1997, Long Short-Term Memory, Neural "
    "Computation, 9 (8), 1735-1780.",
    "KINGMA, D.P. and BA, J., 2015, Adam: A Method for Stochastic Optimization, "
    "3rd International Conference on Learning Representations (ICLR 2015), San Diego, USA.",
    "LUGARESI, C., TANG, J., NASH, H., MCCLANAHAN, C., UBOWEJA, E., HAYS, M., "
    "ZHANG, F., CHANG, C.L., YONG, M.G., LEE, J., CHANG, W.T., HUA, W., GEORG, M. "
    "and GRUNDMANN, M., 2019, MediaPipe: A Framework for Building Perception "
    "Pipelines, arXiv preprint arXiv:1906.08172.",
    "ONNX, 2019, Open Neural Network Exchange [online], https://onnx.ai "
    "[Ziyaret Tarihi: 31 Mayıs 2026].",
    "PASZKE, A., GROSS, S., MASSA, F. ve diğ., 2019, PyTorch: An Imperative Style, "
    "High-Performance Deep Learning Library, Advances in Neural Information "
    "Processing Systems 32 (NeurIPS 2019), 8024-8035.",
    "THREE.JS, 2023, Three.js JavaScript 3D Library [online], https://threejs.org "
    "[Ziyaret Tarihi: 31 Mayıs 2026].",
]
for r in refs:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    run = p.add_run(r); run.font.name = "Times New Roman"; run.font.size = Pt(12)
page_break()

# ============================================================ EKLER
h1("", "EKLER")
h2("EK-1.", "122 Kelimelik Sözlüğün Tam Listesi")
caption("Tablo E.1. 122 kelimelik sözlüğün tam listesi")
WORDS = ["ABİ","ABLA","AFERİN","AĞUSTOS","AKILLI","AKRABA","AKŞAM","ALERJİ","ALLAH",
"ALTI","AMCA","AMİN","ANNE","APTAL","ARALIK","ARAP","ARKADAŞ","AŞK","ATATÜRK","AYIP",
"AYRAN","BABA","BAKLAVA","BAKMAK","BARIŞ","BEN","BERABER","BEŞ","BİLGİSAYAR","BİR",
"BİTMEK","BOŞ","BOŞ VERMEK","BU KADAR","BUGÜN","BULUŞMAK","BÜYÜK","ÇALIŞMAK","ÇARŞAMBA",
"ÇAY","CD","CEP TELEFONU","CEZA","ÇİKOLATA","ÇOK","CUMA","CUMARTESİ","ÇÜNKÜ","DİKKAT",
"DOKTOR","DOKUZ","DÖRT","DOYMAK","DÜN","EKİM","EKMEK","ELHAMDÜLİLLAH","ERİK","EV","EVET",
"EVLENMEK","EYLÜL","FİNAL","FUTBOL","GÖRÜŞMEK","GÜN","HANGİ","HAZİRAN","HEMŞİRE","HENTBOL",
"HOŞÇA KAL","İÇMEK","İKİ","İYİ","KAÇ","KAFE","KAHVE","KASIM","KENDİ","KIŞ","KÖTÜ","KÜÇÜK",
"MART","MAYIS","MERHABA","NASILSIN","NE","NEREDE","NİSAN","OCAK","ÖĞRETMEN","OKUL","ORUÇ",
"PARK","PAZAR","PAZARTESİ","PERŞEMBE","SAAT","SABAH","SALI","SEKİZ","SEN","SIFIR","SONRA",
"SU","ŞUBAT","TAMAM","TEMMUZ","TEŞEKKÜR","TUVALET","TV","ÜÇ","VOLEYBOL","YAN","YAPMAK",
"YARDIM","YATMAK","YEDİ","YEMEK","YETER","YORULMAK","ZENGİN"]
rows4 = []
for i in range(0, len(WORDS), 4):
    chunk = WORDS[i:i+4]
    while len(chunk) < 4:
        chunk.append("")
    rows4.append(chunk)
make_table(["Kelime", "Kelime", "Kelime", "Kelime"], rows4)
body(f"Toplam kelime (sınıf) sayısı: {len(WORDS)}.", indent=False)

h2("EK-2.", "Sistem Bileşenleri ve Kullanılan Teknolojiler")
make_table(["Bileşen", "Teknoloji"], [
    ["Anahtar nokta çıkarımı", "MediaPipe Holistic (543 nokta)"],
    ["Görüntü işleme", "OpenCV"],
    ["Sınıflandırma modeli", "GRU (PyTorch ile eğitim)"],
    ["Mobil model formatı", "ONNX"],
    ["Sunucu", "Python, Flask, Socket.IO, Werkzeug 2.3.8"],
    ["Mobil uygulama", "Flutter / Dart (Android)"],
    ["Avatar", "Three.js (rain.glb), saved_poses.json (123 poz)"],
    ["Sesli çıktı / giriş", "Web Speech API, flutter_tts, speech_to_text"],
])

h2("EK-3.", "Uygulama Ekran Görüntüleri")
body("Aşağıda SignBridge sisteminin web ve mobil arayüzlerine ait ekran görüntüleri "
     "verilmiştir.", indent=False)

add_image("browser_web_kamera.png", 15.0)
caption("Şekil E.1. Web arayüzü – kamera (işaret tanıma) sekmesi")

add_image("browser_web_avatar.png", 15.0)
caption("Şekil E.2. Web arayüzü – avatar (Türkçe→işaret) sekmesi")

two_images_row("phone_mobil_kamera_merhaba.png", "phone_mobil_kamera_nasılsın.png", width_cm=6.2)
caption("Şekil E.3. Mobil uygulama – kamera ekranı (MERHABA ve NASILSIN tanıma)")

two_images_row("phone_mobil_avatar_merhaba.png", "phone_mobil_avatar_hoscakal.png", width_cm=6.2)
caption("Şekil E.4. Mobil uygulama – avatar ekranı (MERHABA ve HOŞÇA KAL işaretleri)")

two_images_row("phone_mobil_avatar_ataturk.png", "phone_mobil_avatar_elhamdulıllah.png", width_cm=6.2)
caption("Şekil E.5. Mobil uygulama – avatar ekranı (ATATÜRK ve ELHAMDÜLİLLAH işaretleri)")

# ---- Sayfa numaralama (kilavuz md.1.7) ----
# Bolum 0: kapak -> numara YOK.
# Bolum 1: on bolumler (Onsoz..Ozet) -> roma rakami (i, ii, ...), 1'den baslar.
# Bolum 2: govde (Giris..Ekler) -> Arap rakami (1, 2, ...), 1'den baslar.
secs = doc.sections
if len(secs) >= 2:
    add_page_number(secs[1], "lowerRoman", start=1)
if len(secs) >= 3:
    add_page_number(secs[2], "decimal", start=1)

out = r"C:\Users\leven\Downloads\SignBridge-Bitirme-Odevi.docx"
try:
    doc.save(out)
except PermissionError:
    import datetime
    out = r"C:\Users\leven\Downloads\SignBridge-Bitirme-Odevi-" + \
          datetime.datetime.now().strftime("%H%M%S") + ".docx"
    doc.save(out)
    print("[!] Ana dosya acik oldugu icin yeni isimle kaydedildi.")
print("Olusturuldu:", out)
print("Toplam kelime:", len(WORDS))
