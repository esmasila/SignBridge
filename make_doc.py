# -*- coding: utf-8 -*-
"""SignBridge teknik dokumantasyon dosyasi olustur."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
# Styles
styles = doc.styles
normal = styles['Normal']
normal.font.name = 'Calibri'
normal.font.size = Pt(11)

def h(text, level=1):
    p = doc.add_heading(text, level=level)
    p.style.font.color.rgb = RGBColor(0x5B, 0x21, 0xB6)
    return p

def p(text):
    return doc.add_paragraph(text)

def code(text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    return para

def bullet(text):
    return doc.add_paragraph(text, style='List Bullet')

# ─── BASLIK ───
title = doc.add_heading('SignBridge — Türk İşaret Dili Çift Yönlü AI Çevirmeni', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph('Teknik Dokümantasyon')
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(14)
sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

doc.add_paragraph()

# ─── 1. GENEL BAKIS ───
h('1. Proje Özeti', 1)
p('SignBridge, Türk İşaret Dili (TİD) ile yazılı/sözlü Türkçe arasında gerçek zamanlı, çift yönlü çeviri yapan bir yapay zeka sistemidir. Türkiye\'deki yaklaşık 700.000 işitme engelli vatandaşın iletişim engelini teknolojiyle aşmayı hedefler.')

h('İki Yön', 2)
bullet('Kamera → Türkçe: Kullanıcı işaret yapar, sistem Türkçe metne ve sese çevirir.')
bullet('Türkçe → Avatar: Kullanıcı Türkçe yazar veya sesle söyler, 3D avatar işaret dilini gösterir.')

h('Platformlar', 2)
bullet('Web uygulaması — http://localhost:5050 (Chrome, Edge)')
bullet('Mobil uygulama — Android (Flutter ile yazıldı)')
bullet('Sunucu — Python (Flask + PyTorch + MediaPipe)')

# ─── 2. MIMARI ───
h('2. Sistem Mimarisi', 1)

h('Üç Ana Bileşen', 2)
p('Sistem üç katmandan oluşur:')
bullet('Tanıma Modülü (Kamera → Metin): MediaPipe Holistic ile el ve vücut noktalarını çıkarır, GRU sinir ağı bunları kelimeye sınıflandırır.')
bullet('Üretim Modülü (Metin → İşaret): NLP modülü Türkçe cümleyi gloss\'lara çevirir, Three.js avatar bunları sırayla animasyon olarak oynatır.')
bullet('Sunucu Katmanı: Flask + Socket.IO ile gerçek zamanlı veri akışı sağlar.')

h('Veri Akışı (Kamera → Metin)', 2)
p('1) Kamera 30 FPS frame yakalar (web: doğrudan webcam, mobil: telefon kamerası JPEG olarak).')
p('2) MediaPipe Holistic her frame\'de 543 anahtar nokta çıkarır (yüz 468 + poz 33 + sağ el 21 + sol el 21).')
p('3) Bu noktalar X-Y-Z koordinatlarıyla 1629 boyutlu vektör halini alır.')
p('4) Buna ellerin hareket hızı (velocity) eklenir → 1755 boyutlu özellik.')
p('5) 30 frame\'lik dizi GRU\'ya verilir (1 saniyelik pencere).')
p('6) GRU 122 kelime arasından en yüksek olasılıklı olanı seçer.')
p('7) Smoothing + cooldown ile gürültü filtrelenir, cümleye eklenir.')
p('8) NLP modülü gloss dizisini doğal Türkçe cümleye çevirir.')
p('9) TTS (Text-to-Speech) cümleyi sesli okur.')

# ─── 3. MODEL ───
h('3. Yapay Zeka Modeli', 1)

h('Mimari', 2)
code('Input → GRU(256, 2 layer) → Linear(128) → ReLU → Linear(122 class)')
p('Toplam ~1.6 milyon parametre. PyTorch ile eğitildi, ONNX formatında mobile yüklendi.')

h('Veri Seti', 2)
bullet('122 Türkçe kelime (günlük kullanım: aile, sayılar, günler, sıfatlar, fiiller)')
bullet('Her kelime için ~150 örnek video (toplam ~18.000 dizi)')
bullet('6 augmentation: mirror, noise, scale, rotate, translate, timewarp → veri 7x büyütüldü')
bullet('Eğitim setinde ~%99.9 doğruluk')

h('Önemli Teknik Detaylar', 2)
bullet('2-katmanlı normalizasyon: omuza göre + bilek-merkezli (kullanıcının kamera mesafesinden bağımsız)')
bullet('Velocity feature: el ivmesi → hareket dinamiği yakalanır')
bullet('Clip [-5, 5]: outlier landmark\'lardan korunma')

# ─── 4. SUNUCU ───
h('4. Sunucu (Backend)', 1)

h('Teknolojiler', 2)
bullet('Python 3.10')
bullet('Flask + Flask-SocketIO (HTTP + WebSocket)')
bullet('PyTorch (model inference)')
bullet('MediaPipe Holistic (landmark çıkarma)')
bullet('OpenCV (görüntü işleme, JPEG decode)')
bullet('NumPy (vektör işlemleri)')

h('Dosya Yapısı', 2)
code('experiment_v2/web_app.py        ← Ana sunucu (~1700 satır)\n'
     'experiment_v2/checkpoints/      ← Model dosyaları\n'
     '  best_model.pt                  ← Aktif PyTorch modeli\n'
     '  model_v2.onnx                  ← Mobil için ONNX export\n'
     '  label_map.json                 ← 122 kelimenin index haritası\n'
     'experiment_v2/data/<KELIME>/    ← Ham landmark verileri (.npy)\n'
     'experiment_v2/data_aug_v2/      ← Augmentasyonlu veriler\n'
     'experiment_v2/finetune.py       ← Model eğitim scripti\n'
     'experiment_v2/collect.py        ← Veri toplama scripti')

h('Önemli Parametreler', 2)
bullet('CONF_THRESHOLD = 0.88: Cümleye eklemek için minimum güven (%88)')
bullet('CONF_FAST_TRACK = 0.95: Bu üzerinde anlık commit')
bullet('SMOOTH_NEEDED = 2: 2 frame ardışık aynı tahmin gerekli')
bullet('COOLDOWN_SEC = 0.4: Ardışık kelimeler arası bekleme')
bullet('SEQ_LEN = 30: Modelin beklediği frame sayısı')

# ─── 5. WEB ───
h('5. Web Uygulaması', 1)

h('Yapı', 2)
bullet('web/index.html — iki sekmeli portal (Kamera + Avatar)')
bullet('web/avatar/step4_blender_player.html — 3D avatar oynatıcı (Three.js)')
bullet('web/avatar/saved_poses.json — 123 kelime için kayıtlı pose verileri')
bullet('web/avatar/rain.glb — 3D karakter modeli (Blender export, 7 MB)')

h('Kamera Sekmesi', 2)
bullet('Canlı kamera görüntüsü')
bullet('Algılanan kelimeler chip\'ler halinde')
bullet('Türkçe çeviri kutusu (NLP işlenmiş)')
bullet('Demo Modu toggle (28 kelime whitelist)')
bullet('Otomatik Sesli Oku')
bullet('Hızlı Cümleler — 10 demo cümlesi tek tıkla')

h('Avatar Sekmesi', 2)
bullet('3D karakter (Three.js + WebGL)')
bullet('Türkçe metin input (yazılı veya sesli)')
bullet('Mikrofon butonu (STT — Speech to Text)')
bullet('Hız slayderı (0.3x – 2x)')
bullet('Quick phrases — hazır cümleler')

# ─── 6. MOBIL ───
h('6. Mobil Uygulama', 1)

h('Teknolojiler', 2)
bullet('Flutter 3.x')
bullet('Dart')
bullet('Camera plugin (YUV420 capture)')
bullet('Socket.IO client')
bullet('ONNX Runtime Mobile (yerel mod için)')
bullet('flutter_tts (sesli okuma)')
bullet('speech_to_text (sesle yazma)')
bullet('WebView (avatar tab için)')

h('Çalışma Modları', 2)
bullet('Sunucu modu (varsayılan): Telefon JPEG yollar → PC server inference → PC döner')
bullet('Telefonda modu: Tüm işlem cihazda (ONNX), internet gerektirmez')

h('Mobil Özellikleri', 2)
bullet('Kamera + Avatar tab\'ları')
bullet('Demo Modu toggle (kameranın altında)')
bullet('Ön/Arka kamera geçişi')
bullet('Sesli okuma (TTS) — kamera tahmininden sonra')
bullet('Mikrofon (STT) — avatar tab\'ında')
bullet('Hızlı cümleler')
bullet('Debug ekranı — sistem loglarını gösterir')

# ─── 7. NLP ───
h('7. NLP (Doğal Dil İşleme) Modülü', 1)

h('Gloss → Türkçe Çeviri', 2)
p('Model işaret dilinin "gloss" karşılığını verir (örn. BEN + CALISMAK). NLP modülü bunu doğal Türkçeye çevirir:')

bullet('Özne-fiil çekimi: BEN + CALISMAK → "Ben çalışıyorum"')
bullet('Soru tespiti: SEN + NE + YAPMAK → "Sen ne yapıyorsun?"')
bullet('Cohortative (BERABER): BERABER + KAHVE + ICMEK → "Beraber kahve içelim mi?"')
bullet('İmperatif (KENDI): KENDI + IYI + BAKMAK → "Kendine iyi bak."')
bullet('Ablativ ek (sayı + SONRA): BES + SONRA → "beşten sonra"')
bullet('Yer eki (PARK + YAN + isim): "parkın yanındaki kafede"')
bullet('Bağlam-temelli düzeltme: BERABER + ? + ICMEK varsa "?" = KAHVE')

h('Türkçe → Gloss', 2)
p('Avatar için: Kullanıcı yazdığı Türkçeyi gloss dizisine çevirir.')
bullet('ASCII normalize: "çalışıyorum" → "CALISIYORUM"')
bullet('Suffix stripping: "IYORUM" eki silinir → "CALIS"')
bullet('Mastar arama: "CALIS" + "MAK" = "CALISMAK" (kayıtlı pose)')
bullet('Stop words: "ve", "ile", "bir", "mı" gibi kelimeler atlanır')

# ─── 8. ONEMLI OZELLIKLER ───
h('8. Önemli Özellikler', 1)

h('Demo Modu (28-Kelime Whitelist)', 2)
p('Kullanıcı bir butonla aktifleştirir. Sistem yalnızca 10 demo cümlesinde geçen 28 kelime arasından tahmin yapar. Bu sayede:')
bullet('Karışıklık önemli ölçüde azalır')
bullet('Tez sunumu gibi kontrollü demoda %95+ doğruluk')
bullet('Server top-1 yerine "whitelist içinde en yüksek 122 sınıf top-K" araması')

h('Sesli Özellikler', 2)
bullet('TTS (Text-to-Speech): Cümle tamamlandığında otomatik sesli okur (Türkçe, Microsoft Tolga ses)')
bullet('STT (Speech-to-Text): Mikrofona konuşunca metne yazar, avatar oynatır')

h('Glassmorphic UI', 2)
bullet('Modern dark tema (deep navy + neon accents)')
bullet('Background glow orbs (animated)')
bullet('backdrop-filter blur cards')
bullet('Mor → cyan gradient vurgular')

# ─── 9. TEKNIK TERIMLER ───
h('9. Teknik Terimler Sözlüğü', 1)

terms = [
    ('Gloss', 'İşaret dilinin yazılı karşılığı (BEN, MERHABA, CALISMAK gibi büyük harfli kök kelime).'),
    ('Landmark', 'MediaPipe\'ın görüntüden çıkardığı anahtar nokta. Toplam 543 nokta: 468 yüz, 33 vücut, 21 sağ el, 21 sol el.'),
    ('MediaPipe Holistic', 'Google\'ın geliştirdiği, tek videoda yüz + vücut + iki el landmark\'larını birlikte çıkaran kütüphane.'),
    ('GRU (Gated Recurrent Unit)', 'Bir tür RNN. Zaman içindeki bağımlılıkları öğrenir. LSTM\'in sadeleştirilmiş, daha hızlı versiyonu.'),
    ('Softmax', 'Sinir ağı çıktısını olasılık dağılımına çeviren fonksiyon. Toplamı 1 olur.'),
    ('ONNX (Open Neural Network Exchange)', 'PyTorch modelini platform-bağımsız formata çevirme standardı. Mobilde ONNX Runtime ile çalışır.'),
    ('Buffer', 'Modelin beklediği son N frame\'i tutan kuyruk yapısı (deque). Burada N=30.'),
    ('Smoothing', 'Aynı tahminin N kere ardışık gelmesini bekleyerek yanıltıcı tahminleri filtreleyen mantık.'),
    ('Cooldown', 'İki ardışık kelime ekleme arasındaki minimum bekleme süresi.'),
    ('Confidence (güven)', 'Modelin tahminine ne kadar emin olduğu (0-1 arası). 0.95 = %95 emin.'),
    ('Fast-track', 'Çok yüksek güvende (>0.95) smoothing\'i atlayıp anlık ekleme.'),
    ('Whitelist (Demo modu)', 'Sadece belirli kelimeler arasından seçim yapma. 28 kelimelik küçük bir set.'),
    ('Velocity feature', 'Frame\'ler arası el hareket farkı (Δx, Δy, Δz). Hareket dinamiğini yakalar.'),
    ('Augmentation', 'Eğitim verisini büyütme: orijinali değiştirip aynı sınıf için ek örnekler üretme (mirror, rotate, noise vs).'),
    ('NLP', 'Doğal Dil İşleme. Burada gloss\'tan Türkçeye veya Türkçeden gloss\'a çeviri yapar.'),
    ('Cohortative', 'Türkçede "-elim/-alım" ekiyle yapılan davet/önerme (içelim, gidelim, buluşalım).'),
    ('Suffix stripping', 'Türkçe ekleri kök kelimeye ulaşmak için silme (çalışıyorum → çalış → CALISMAK).'),
    ('Glassmorphism', 'UI tasarım stili: yarı saydam camsı paneller + blur + neon vurgular.'),
    ('Socket.IO', 'WebSocket üzerine kurulu gerçek zamanlı iletişim kütüphanesi. Frame yollama için.'),
    ('JPEG quality 55', 'Mobil görüntü kalitesi ayarı. Düşük = küçük dosya + hızlı ağ, yüksek = net görüntü.'),
    ('Frame rate (FPS)', 'Saniyede gönderilen frame sayısı. Mobil 20 FPS, web ~30 FPS.'),
    ('Inference', 'Eğitilmiş modelin girdiyi alıp tahmin üretmesi.'),
]

for term, desc in terms:
    para = doc.add_paragraph()
    bold = para.add_run(term + ': ')
    bold.bold = True
    bold.font.color.rgb = RGBColor(0x5B, 0x21, 0xB6)
    para.add_run(desc)

# ─── 10. KONTROL ───
h('10. Sistem Kontrolleri', 1)

h('Sunucu Açma', 2)
code('cd C:\\Projects\\sign_bridge\\experiment_v2\npython web_app.py\n# → http://localhost:5050')

h('Mobil Bağlama', 2)
p('Settings → Sunucu IP: 192.168.1.28, Port: 5050 (PC ile aynı Wi-Fi).')

h('Model Yeniden Eğitme', 2)
code('python experiment_v2/collect.py        # Yeni kelime için veri topla\n'
     'python experiment_v2/precompute_v2.py  # Augment + velocity\n'
     'python experiment_v2/finetune.py       # GRU eğitim (80 epoch)')

h('Sayısal Özet', 2)
bullet('122 kelime — 28 demo whitelist')
bullet('~%99.9 eğitim doğruluğu')
bullet('Web: ~1 sn / kelime, Mobil: ~1.5 sn / kelime')
bullet('Web sunucu — Python 3.10 + Flask + PyTorch')
bullet('Mobil — Flutter Android (Dart) + ONNX Runtime')
bullet('Frontend — HTML + JavaScript + Three.js')

doc.add_paragraph()
end = doc.add_paragraph('SignBridge — 2026 Bitirme Projesi')
end.alignment = WD_ALIGN_PARAGRAPH.CENTER
end.runs[0].italic = True
end.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

# Save
out = r'C:\Users\leven\Downloads\SignBridge-Dokumantasyon.docx'
doc.save(out)
print(f'Olusturuldu: {out}')
