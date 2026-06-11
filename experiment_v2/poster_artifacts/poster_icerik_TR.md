# SignBridge — Poster İçeriği (Türkçe, 70x100cm)

## BAŞLIK
**SignBridge: Türk İşaret Dili için Çift Yönlü Çeviri Sistemi**
*Derin Öğrenme Tabanlı Gerçek Zamanlı Tanıma ve 3B Avatar ile Üretim*

## YAZARLAR
**Esma Sıla ŞAHİNCİ**
**Danışman:** Öğr. Gör. Kadir HALTAŞ

## KURUM
Nevşehir Hacı Bektaş Veli Üniversitesi
Mühendislik–Mimarlık Fakültesi · Bilgisayar Mühendisliği Bölümü · 2026

---

## ÖZET (ABSTRACT)
Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmasına rağmen Türk İşaret Dili (TİD) için son kullanıcıya yönelik çeviri çözümleri yok denecek kadar azdır. Bu çalışmada, **TİD ↔ Türkçe çift yönlü** çeviri yapan bir sistem geliştirilmiştir. Sistem üç bileşenden oluşur: (1) **Mediapipe Holistic** ile vücut/el/yüz işaret noktalarını çıkaran ön-işlem; (2) **122 kelimelik GRU sinir ağı** ile gerçek zamanlı işaret tanıma; (3) **akıllı Türkçe NLP modülü** ile öznelere göre fiil çekimi yapan cümle üretimi. Tersine yönde, kullanıcının yazdığı/söylediği Türkçe cümle **3B avatar** üzerinde işaret dili olarak gösterilir. Model 122 sınıf üzerinde **%99.88** doğrulama doğruluğu elde etmiş, sistem hem **mobil (Flutter)** hem **web (Flask + Three.js)** üzerinden eş zamanlı çalışmaktadır.

---

## GİRİŞ (INTRODUCTION)
İşitme engelli bireyler günlük iletişimde sıklıkla yazılı Türkçe ile aralarındaki köprü eksikliğinden kaynaklanan engellerle karşılaşır. Mevcut işaret dili çevirmenleri sınırlıdır ve 7/24 erişilebilir değildir. Son yıllarda derin öğrenme alanında özellikle **dizi modelleme** (LSTM, GRU, Transformer) ve **görüntüden işaret noktası çıkarımı** (MediaPipe, OpenPose) alanlarında elde edilen gelişmeler, gerçek zamanlı işaret dili tanımayı uygulanabilir hâle getirmiştir (Koller vd., 2020).

Ancak literatürdeki çalışmaların büyük çoğunluğu **Amerikan İşaret Dili (ASL)** üzerinedir; Türk İşaret Dili için yeterli büyüklükte etiketli veri seti ve son kullanıcı uygulaması bulunmamaktadır. Bu açığı kapatmak amacıyla **SignBridge** geliştirilmiştir.

**Sistemin temel bileşenleri:**
- **Kamera + MediaPipe Holistic**: 30 FPS hızında 1629 boyutlu işaret noktası vektörü çıkarımı
- **122 sınıflık GRU modeli**: 30 frame'lik dizi üzerinden işaret sınıflandırma
- **Akıllı NLP motoru**: 6 farklı zamir, fiil çekimi, soru ekleri, eş anlamlı kelime ayrımı
- **3B Avatar (Three.js)**: Yazılı Türkçe → işaret dili görsel çevirisi

---

## ÇALIŞMANIN AMACI (PURPOSE OF THE STUDY)
Bu çalışmanın temel amacı, **işitme engelli bireyler ile işiten bireyler arasında doğal bir iletişim köprüsü** kurmaktır. Spesifik hedefler:

1. **TİD → Türkçe**: Kameradan canlı görüntü alarak işaretleri tanıyıp gramer açısından doğru Türkçe cümlelere çevirme
2. **Türkçe → TİD**: Yazılı/sözlü Türkçe metni 3B avatar üzerinde işaret dili olarak gösterme
3. **Mobil + Web çift platform**: Hem akıllı telefonda hem de tarayıcıda çalışan eş zamanlı sistem
4. **Düşük gecikme**: Gerçek zamanlı kullanım için her işaret <1.5 saniye içinde tanınmalı

---

## YÖNTEM (THEORETICAL REVIEW)

### Veri Toplama ve Ön İşleme
- **122 TİD kelimesi** üzerinde toplam **105.182 örnek** (sınıf başına ortalama 862 augment edilmiş dizi)
- Her örnek **30 frame × 1755 boyutlu** vektör (1629 ham landmark + 126 hız bileşeni)
- Augmentation: gürültü, ölçek, çevirme, döndürme, zaman bükme, ayna

### Model Mimarisi
| Bileşen | Detay |
|---|---|
| **Tip** | 2 katmanlı GRU (Gated Recurrent Unit) |
| **Giriş boyutu** | 1755 (33 poz + 21×2 el + 468 yüz × 3 koordinat + hız) |
| **Gizli boyut** | 256 |
| **Sınıflandırıcı** | FC(256→128) → ReLU → Dropout → FC(128→122) |
| **Toplam parametre** | ~1.9 milyon |
| **Optimizasyon** | AdamW, LR=1e-3, Batch=64 |
| **Eğitim süresi** | 80 epoch × ~15s = ~20 dakika (NVIDIA GPU) |

### Cümle Üretimi (NLP)
Tanınan işaret etiketleri (örn. `[BEN, KAHVE, ICMEK]`) **akıllı çekim motoru**na verilir:
- **Özne tespiti**: BEN/SEN/BERABER/KENDI → fiil için doğru kişi eki
- **Bağlam ayrımı**: KAHVE/KAFE benzer işaretler, önceki kelimelere göre seçilir
- **Soru ekleri**: BERABER + fiil → kohortatif soru ("…elim mi?")
- **Türkçe gramer ekleri**: ablatif (`-DAn`), yer bildirme (`-DA`), bulunma (`-lI`)
- **Sonuç**: `[BEN, KAHVE, ICMEK]` → **"Ben kahve içiyorum."**

### Üretim Yönü: Avatar
- Three.js + Blender'da çıkarılmış 122 işaret pozu (rain.glb iskelet)
- Kullanıcı yazdığı Türkçe → kelime kelime ayrıştırılıp avatar üzerinde animasyon

---

## DENEYSEL DÜZENEK (EXPERIMENTAL SETUP)

**Donanım:**
- Eğitim: NVIDIA RTX serisi GPU
- Test: Android telefonlar (Samsung A serisi), Windows 11 + Chrome

**Yazılım Yığını:**
- **Backend**: Python 3.10 + Flask + Flask-SocketIO + PyTorch + MediaPipe
- **Mobil**: Flutter 3.x + Dart + Camera + WebSocket + Flutter TTS
- **Web**: HTML5 + CSS3 + Three.js + Web Speech API
- **Veritabanı**: JSON tabanlı pozlar (122 sınıf × ortalama 5 keyframe)

**Çalışma Modları:**
1. **Sunucu modu**: Telefon JPEG gönderir, sunucu MediaPipe + GRU çalıştırır
2. **Yerel mod**: ONNX export + telefonda MediaPipe + GRU (offline çalışma)
3. **Demo modu**: 10 önceden hazırlanmış cümle için 29 kelimelik beyaz liste

---

## SONUÇLAR VE TARTIŞMA (RESULTS AND DISCUSSION)

### Model Performansı
| Metrik | Değer |
|---|---|
| **Eğitim doğruluğu (son)** | %100.0 |
| **Doğrulama doğruluğu (en iyi)** | **%99.88** (Epoch 61) |
| **Bağımsız değerlendirme** | **%99.97** (3660 örnek üzerinde) |
| **Sınıf başına ortalama doğruluk** | %99.97 |
| **Eğitim süresi** | ~20 dakika (80 epoch) |
| **Çıkarım gecikmesi** | ~50ms (GPU) / ~80ms (CPU) |

### Sistem Performansı
| Metrik | Değer |
|---|---|
| **Mobil tahmin gecikmesi** (sunucu modu) | ~250ms (kelime başına) |
| **30 frame buffer doldurma** | ~1.0 saniye @ 30 FPS |
| **NLP çekim doğruluğu** (10 demo cümlesi) | %100 |
| **3B Avatar animasyon hızı** | 60 FPS |

### Karşılaşılan Zorluklar ve Çözümler
1. **Benzer işaretlerin karışması** (KAHVE↔KAFE, BEN↔IYI): Per-kelime güven eşiği (%92-%97) ve ardışık frame yumuşatması (3 frame) ile çözüldü
2. **MediaPipe paralel çağrı segfault**: `threading.Lock` ile sunucu tarafında işlem sıralandı
3. **Mobil kamera donması**: Zoom çağrısı stream başlatıldıktan sonra ertelendi
4. **Geçiş frame'lerinde gürültü**: Eller arası geçişte 400ms cooldown + buffer reset

---

## SONUÇ (CONCLUSIONS)
Geliştirilen **SignBridge** sistemi, Türk İşaret Dili için literatürdeki en kapsamlı son-kullanıcı uygulamalarından biridir:

✅ **122 kelime / 10 demo cümlesi** üzerinde %99+ doğrulukla canlı tanıma
✅ **Çift yönlü çeviri**: hem kameradan TİD okuma, hem de avatar ile TİD üretimi
✅ **Mobil + Web çift platform**: cihaz fark etmeksizin çalışır
✅ **Akıllı NLP** ile dilbilgisi açısından doğru cümleler

**Gelecek çalışmalar:**
- Kelime sayısını **500+** seviyesine çıkarma
- **Cümle düzeyi** doğrudan tanıma (kelime-kelime yerine)
- Transformer tabanlı **daha sağlam model**
- iOS desteği

---

## KAYNAKLAR (REFERENCES)

1. Koller, O., Camgoz, N. C., Ney, H., & Bowden, R. (2020). *Weakly Supervised Learning with Multi-Stream CNN-LSTM-HMMs to Discover Sequential Parallelism in Sign Language Videos*. IEEE TPAMI.

2. Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.

3. Cho, K., van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). *On the Properties of Neural Machine Translation: Encoder–Decoder Approaches*. arXiv:1409.1259.

4. Camgoz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018). *Neural Sign Language Translation*. CVPR.

5. Şahin, M. (2018). *Türk İşaret Dili Sözlüğü*. Aile, Çalışma ve Sosyal Hizmetler Bakanlığı, Ankara.

6. PyTorch Team. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.

---

## POSTER YERLEŞİM ÖNERİSİ (Şablonla uyumlu)

```
┌─────────────────────────────────────────────────────────────┐
│  [LOGO]    SIGNBRIDGE — TİD ÇİFT YÖNLÜ ÇEVİRİ      [LOGO]   │
│            Esma Sıla ŞAHİNCİ · Danışman: K. HALTAŞ          │
│            Nevşehir HBVÜ · Bilgisayar Mühendisliği          │
├──────────────────────────┬──────────────────────────────────┤
│  ÖZET                    │  SONUÇLAR                        │
│  [kısa paragraf]         │  📊 training_history_122.png     │
│                          │  📊 confusion_matrix_122.png     │
│  GİRİŞ                   │  Tablo: model metrikleri         │
│  [3 paragraf]            │                                  │
│                          │  DENEYSEL DÜZENEK                │
│  AMAÇ                    │  📷 Mobil ekran görüntüsü        │
│  [4 madde]               │  📷 Web çift kamera ss           │
│                          │  📷 Avatar 3B ss                 │
│  YÖNTEM                  │  📷 Demo modu ss                 │
│  • Veri (105k örnek)     │                                  │
│  • Model (GRU 1.9M)      │  SONUÇ + GELECEK                 │
│  • NLP motoru            │  ✅ 122 kelime, %99.97           │
│  • Avatar                │  ✅ Mobil + Web                  │
│  [mimari diyagram]       │  → 500+ kelime, transformer     │
│                          │                                  │
│                          │  KAYNAKLAR                       │
│                          │  [6 referans, küçük font]        │
└──────────────────────────┴──────────────────────────────────┘
```

## RENK PALETİ (Tema)
- **Birincil**: #6C63FF (mor — başlıklar, vurgular)
- **İkincil**: #22D3EE (cyan — veri görselleri, çerçeveler)
- **Vurgu**: #EC4899 (pembe — sayısal istatistikler)
- **Arka plan**: #FFFFFF veya #F8FAFC (beyaz/açık)
- **Metin**: #0F172A (koyu lacivert)

## YAZI TİPİ
- **Başlık**: Inter Bold 72-96pt
- **Bölüm başlıkları**: Inter SemiBold 36-44pt
- **Gövde**: Inter Regular 18-24pt
- **Tablolar**: Inter Medium 16pt
- **Kaynaklar**: Inter Regular 12-14pt
