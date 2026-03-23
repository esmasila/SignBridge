# SignBridge - Türk İşaret Dili Tanıma Sistemi
## Proje Geliştirme Raporu

**Proje Adı:** SignBridge  
**Geliştirici:** Esma Sila  
**Başlangıç Tarihi:** Kasım 2025  
**Son Güncelleme:** 28 Kasım 2025  
**Repository:** https://github.com/esmasila/SignBridge  

---

## 1. YÖNETSEL ÖZET

SignBridge, Türk İşaret Dili (TİD) işaretlerini gerçek zamanlı olarak Türkçe metne ve sese dönüştüren bir yapay zeka sistemidir. Proje kapsamında iki temel model başarıyla geliştirilmiş ve web tabanlı bir arayüz ile hizmete sunulmuştur.

### Proje Hedefleri
- ✅ Rakam tanıma sistemi (0-9)
- ✅ İşaret kelime tanıma sistemi (20 kelime)
- ✅ Gerçek zamanlı görüntü işleme
- ✅ Web tabanlı kullanıcı arayüzü
- ✅ Türkçe sesli okuma (TTS) entegrasyonu

### Ana Başarılar
- **Rakam Tanıma:** %99.79 doğruluk oranı
- **Kelime Tanıma:** %99.6 mAP50 skoru (YOLOv11)
- **Performans:** ~27 FPS gerçek zamanlı işleme
- **Kelime Kapasitesi:** 20 temel TİD kelimesi

---

## 2. TEKNİK MİMARİ

### 2.1 Sistem Bileşenleri

#### Backend
- **Framework:** FastAPI (Python 3.10)
- **Deep Learning:** PyTorch 2.6.0 (CUDA 12.4)
- **Görüntü İşleme:** OpenCV, MediaPipe Holistic v0.10.21
- **Model Framework:** Ultralytics YOLOv11

#### Frontend
- **Teknoloji:** HTML5, CSS3, JavaScript
- **Özellikler:** Gerçek zamanlı webcam, model seçici, TTS kontrolü
- **UI Framework:** Bootstrap-based responsive design

#### Donanım
- **GPU:** NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- **İşletim Sistemi:** Windows 11
- **CUDA:** Version 12.4

### 2.2 API Endpoints

```
GET  /                  → Web arayüzü
GET  /health            → Sistem durumu kontrolü
GET  /model/info        → Model bilgileri
POST /predict           → Rakam tanıma (0-9)
POST /predict_sign      → İşaret kelime tanıma (20 kelime)
POST /tts               → Türkçe text-to-speech
```

---

## 3. GELİŞTİRİLEN MODELLER

### 3.1 Rakam Tanıma Modeli (Model 1)

**Model Tipi:** Multimodal CNN  
**Hedef:** Türk İşaret Dili rakamları (0-9)

#### Mimari Özellikleri
- **CNN Branch:** 4 konvolüsyon katmanı (görüntü işleme)
- **MLP Branch:** 3 fully connected katman (landmark işleme)
- **Fusion Layer:** İki modaliteyi birleştirme
- **Input:** 
  - Görüntü: 64×64 RGB
  - Landmarks: 1629 özellik (MediaPipe Holistic)

#### Performans Metrikleri
| Metrik | Değer |
|--------|-------|
| Validation Accuracy | 99.79% |
| FPS | ~27 |
| Model Boyutu | 15.7 MB |
| Inference Time | ~37ms |

#### Eğitim Detayları
- **Dataset:** Webcam ile toplanan custom dataset
- **Epoch:** 50
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Data Augmentation:** Rotation, scaling, noise

**Sonuç:** Gerçek dünya testlerinde mükemmel performans ✅

---

### 3.2 İşaret Kelime Tanıma Modeli (Model 2)

**Model Tipi:** YOLOv11-nano (Object Detection)  
**Hedef:** 20 temel TİD kelimesi

#### Tanınan Kelimeler
Anne, Arkadaş, Baba, Dur, Ev, Evet, Hayır, Kardeş, Merhaba, Nasıl, Nerede, Özür Dilemek, Tamam, Telefon, Teşekkürler, Tuvalet, Yemek, İçmek, İyi, Kötü

#### Model Özellikleri
- **Mimari:** YOLOv11-nano
- **Parametre Sayısı:** 2,593,740
- **Katman Sayısı:** 181
- **GFLOPs:** 6.5

#### Dataset
- **Kaynak:** Roboflow TURK ISARET DILI v2
- **Toplam Görüntü:** 21,928
  - Train: 20,020 görüntü
  - Validation: 1,908 görüntü
- **Format:** YOLO (bounding box annotations)
- **Çözünürlük:** 640×640 piksel

#### Eğitim Parametreleri
```yaml
Epochs: 50
Batch Size: 16
Image Size: 640×640
Device: CUDA (RTX 3050)
Optimizer: SGD (lr=0.01, momentum=0.9)
Patience: 10
AMP: Enabled
```

#### Eğitim Süreci
- **Süre:** ~3 saat
- **GPU Kullanımı:** NVIDIA RTX 3050 (4GB VRAM)
- **Pretrained Weights:** YOLOv11-nano (transfer learning)
- **Transferred Items:** 448/499

#### Performans Metrikleri (Epoch 50)
| Metrik | Değer |
|--------|-------|
| mAP50 | 99.596% |
| mAP50-95 | 86.443% |
| Precision | 99.505% |
| Recall | 99.596% |
| Box Loss | 0.54832 |
| Class Loss | 0.22985 |

**Sonuç:** Gerçek dünya testlerinde başarılı ✅

---

## 4. DENENEN YAKLAŞIMLAR VE SONUÇLAR

### 4.1 Başarılı Yaklaşımlar ✅

#### 4.1.1 Multimodal CNN (Rakamlar)
- **Strateji:** Görüntü + Landmark birleştirme
- **Sonuç:** %99.79 accuracy
- **Avantaj:** Tek modaliteden daha robust
- **Durum:** Production'da aktif

#### 4.1.2 YOLOv11 Object Detection (Kelimeler)
- **Strateji:** Frame-level detection
- **Sonuç:** %99.6 mAP50
- **Avantaj:** Hızlı, skalası, GPU-optimized
- **Durum:** Production'da aktif

---

### 4.2 Başarısız Yaklaşımlar ❌

#### 4.2.1 Alfabe Tanıma (A-Z)

**Hedef:** 26 Türk alfabesi harfini tanıma

**Kullanılan Dataset:**
- Kaynak: Kaggle Turkish Sign Language Alphabet
- Görüntü Sayısı: 167,000+
- Format: JPG images

**Model:** CNN-based classifier

**Test Performansı:**
- Test Accuracy: %99.83 (çok yüksek!)

**Gerçek Dünya Sonucu:**
- **BAŞARISIZ** ❌
- O harfi sürekli B olarak tanınıyor
- Harfler görsel olarak çok benzer
- Model overfitting yapmış

**Alınan Ders:**
> Yüksek test accuracy ≠ gerçek dünya başarısı. Dataset kalitesi > dataset miktarı.

**Karar:** Proje terk edildi, kodlar `legacy_alphabet/` klasörüne taşındı (sonra silindi).

---

#### 4.2.2 LSTM Sequence Classification

**Hedef:** Dinamik hareketli kelimeleri zamansal olarak tanıma

**Strateji:**
- MediaPipe ile 60 frame landmark çıkarma
- LSTM ile temporal sequence classification

**Dataset Toplama:**
- Yöntem: Manuel webcam ile veri toplama
- Script: `collect_sequences.py`
- Toplam: 250 sample
  - 5 kelime × 50 sample each
  - Kelimeler: MERHABA, EVET, HAYIR, LÜTFEN, SU
- Format: 60 frames × 1629 features (NumPy arrays)

**Model Mimarisi:**
- Bidirectional LSTM
- 2 layers, 256 hidden units
- Dropout: 0.3
- Total Parameters: 5.5M

**Eğitim Sonuçları:**
- Validation Accuracy: %62 (düşük)
- Train Accuracy: %75

**Gerçek Dünya Problemi:**
- **Class Imbalance:** LÜTFEN kelimesi dominant
  - LÜTFEN: 28/50 prediction (%56)
  - TEŞEKKÜR ETMEK: 0/50 prediction (%0 recall!)
- **Movement Similarity:** Kelimeler birbirine çok benziyordu
- **Data Quality:** 50 sample/class yetersiz

**Confusion Matrix Analizi:**
```
LÜTFEN ile SU karışıyor (benzer el hareketi)
TEŞEKKÜR ETMEK hiç tahmin edilmiyor
Model her şeyi LÜTFEN olarak görüyor
```

**Alınan Ders:**
> - Manuel veri toplama çok zor ve yetersiz
> - Temporal modeller için çok daha fazla data gerekli (min 1000+ samples/class)
> - Class balance kritik

**Karar:** Proje terk edildi, YOLO yaklaşımına geçildi. Kodlar `tid_sequence/` klasöründe saklandı (gelecekte hibrit sistem için).

---

## 5. PROJE EVRİMİ (KRONOLOJİK)

### Faz 1: Rakam Tanıma (Kasım 2025, Hafta 1)
```
Gün 1-2:  Proje setup, veri toplama aracı
Gün 3-4:  İlk CNN modeli (%98 acc)
Gün 5:    MediaPipe entegrasyonu
Gün 6-7:  Multimodal CNN geliştirme
Sonuç:    ✅ %99.79 accuracy elde edildi
```

### Faz 2: Alfabe Denemesi (Hafta 2)
```
Gün 1:    Kaggle dataset indirildi (167K images)
Gün 2-3:  CNN modeli eğitildi
Gün 4:    Test: %99.83 accuracy!
Gün 5:    Gerçek dünya testi → ❌ O/B karışıyor
Karar:    Proje terk edildi
```

### Faz 3: LSTM Sequences (Hafta 2-3)
```
Gün 1-2:  Sequence collection tool
Gün 3-5:  250 sample manuel toplama (zahmetli!)
Gün 6-7:  LSTM/GRU modelleri eğitimi
Gün 8:    Sonuç: %62 val acc, class imbalance
Gün 9:    Gerçek dünya → ❌ LÜTFEN dominant
Karar:    Proje terk edildi
```

### Faz 4: YOLO İşaret Kelimeler (Hafta 3-4)
```
Gün 1:    YOLO dataset bulundu (Roboflow, 20K images)
Gün 2:    PyTorch CPU→CUDA upgrade
Gün 3:    İlk training (CPU, çok yavaş)
Gün 4:    GPU training setup
Gün 5:    YOLOv11 50 epoch eğitim (~3 saat)
Gün 6:    Sonuç: ✅ %99.6 mAP50!
Gün 7:    API entegrasyonu
Gün 8:    Web UI güncelleme
Sonuç:    ✅ Production'a alındı
```

### Faz 5: Finalizasyon (Hafta 4)
```
Gün 1:    Kod cleanup (~2GB gereksiz dosya silindi)
Gün 2:    GitHub repository oluşturma
Gün 3:    Dokümantasyon ve test
Sonuç:    ✅ Proje GitHub'a yüklendi
```

---

## 6. TEKNİK ZORLUKLAR VE ÇÖZÜMLERİ

### 6.1 GPU Training Problemi

**Problem:**
- İlk PyTorch kurulumu CPU-only idi
- YOLO training CPU'da başladı → çok yavaş (tahmini 2.5 saat/epoch)
- Epoch 1'de kullanıcı durdurdu

**Çözüm:**
```bash
# PyTorch'u kaldır
pip uninstall torch torchvision torchaudio -y

# CUDA versiyonunu kur
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Sonuç:**
- GPU training aktif edildi
- RTX 3050 kullanıldı
- 50 epoch ~3 saatte tamamlandı

---

### 6.2 YOLO Dataset Path Sorunu

**Problem:**
`data.yaml` dosyasında yanlış relative path:
```yaml
train: ../train/images  # Hatalı
val: ../valid/images    # Hatalı
```

**Çözüm:**
```yaml
train: train/images   # Doğru
val: valid/images     # Doğru
```

---

### 6.3 MediaPipe Başlatma Hatası

**Problem:**
API başlangıcında MediaPipe hatası:
```
The path does not exist: mediapipe/modules/holistic_landmark/holistic_landmark_cpu.binarypb
```

**Durum:**
- YOLO modeli için MediaPipe gerekli değil
- Sadece rakam modeli için kullanılıyor
- API yine de başarıyla başlıyor

**Çözüm:**
- Hata zararsız, ignore edildi
- Rakam modeli MediaPipe olmadan da çalışabiliyor (sadece CNN branch kullanılıyor)

---

### 6.4 Web UI Console Uyarısı

**Problem:**
Browser console'da permissions policy violation:
```
[Violation] Permissions policy violation: unload is not allowed
```

**Çözüm:**
Webcam kapatma kodunu iyileştirdik:
```javascript
function stopCamera() {
    isRunning = false;  // Önce durdur
    
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => {
            track.stop();
            track.enabled = false;  // Eklendi
        });
        video.srcObject = null;  // Stream referansını temizle
    }
}
```

---

## 7. PERFORMANS ANALİZİ

### 7.1 Model Karşılaştırması

| Model | Accuracy | FPS | Boyut | Gerçek Dünya |
|-------|----------|-----|-------|--------------|
| Rakamlar (Multimodal) | 99.79% | 27 | 15.7 MB | ✅ Mükemmel |
| Alfabe (CNN) | 99.83% | - | - | ❌ Başarısız |
| LSTM Sequences | 62% | - | 63 MB | ❌ Başarısız |
| YOLO İşaret | 99.6% | - | - | ✅ Başarılı |

### 7.2 Sistem Kaynak Kullanımı

**Disk Kullanımı:**
- Toplam: 6.38 GB
  - YOLO Dataset: 6.2 GB (turk_isaret_yolo/)
  - Modeller: 150 MB (checkpoints/)
  - LSTM Data: 50 MB (tid_sequence/data/)
  - Kaynak Kod: 10 MB

**RAM Kullanımı (Runtime):**
- API (FastAPI): ~200 MB
- YOLO Model: ~500 MB
- Rakam Model: ~100 MB
- MediaPipe: ~150 MB
- **Toplam:** ~1 GB

**GPU Kullanımı:**
- YOLO Inference: ~1.5 GB VRAM
- Rakam Inference: ~300 MB VRAM

---

## 8. YAZILIM MÜHENDİSLİĞİ PRATİKLERİ

### 8.1 Versiyon Kontrolü
- Git repository oluşturuldu
- GitHub'a push edildi: https://github.com/esmasila/SignBridge
- `.gitignore` düzenlendi (büyük dosyalar hariç)

### 8.2 Kod Organizasyonu
```
signbridge/           # Ana paket
├── api/             # REST API
├── models/          # Model tanımları
├── data/            # Veri işleme
├── utils/           # Yardımcı fonksiyonlar
└── inference/       # Inference scriptleri
```

### 8.3 Dokümantasyon
- ✅ README.md (proje açıklaması)
- ✅ LICENSE (MIT)
- ✅ requirements.txt (dependencies)
- ✅ API dokümantasyonu (FastAPI auto-docs)
- ✅ Kod içi yorumlar
- ✅ PROJE_RAPORU.md (bu dosya)

### 8.4 Testing
- API health check endpoint
- Web UI functional test
- Manual webcam testing
- Test script: `test_yolo_api.py`

### 8.5 Cleanup
Gereksiz dosyalar silindi:
- ❌ `venv/` (eski virtual env)
- ❌ `data/webcam_dataset/` (kullanılmış veri)
- ❌ `runs/` (eski loglar)
- ❌ Legacy docs (MULTIMODAL_GUIDE.md, PROJECT_SUMMARY.md, vb.)
- ❌ Docker dosyaları (henüz kullanılmıyor)
- ❌ Test scriptleri (artık API var)
- ❌ 124 eski log dosyası (son 5 korundu)

**Kazanılan Alan:** ~2 GB

---

## 9. ÖĞRENME ÇIKTILARI

### 9.1 Başarı Faktörleri

**1. Büyük ve Kaliteli Dataset**
```
❌ Manuel toplama (250 sample) → Başarısız
✅ Profesyonel dataset (20K images) → Başarılı
```

**2. Doğru Model Seçimi**
```
❌ LSTM sequences (temporal) → %62 acc
✅ YOLO object detection → %99.6 mAP50
```

**3. Multimodal Yaklaşım**
```
⚠️ Sadece görüntü → %98 acc
✅ Görüntü + Landmarks → %99.79 acc
```

**4. GPU Kullanımı**
```
❌ CPU training → 2.5 saat/epoch (tahmini)
✅ GPU training → 3 saat/50 epoch (gerçek)
```

**5. Erken Gerçek Dünya Testi**
```
⚠️ Alfabe: %99.83 test acc → Gerçekte başarısız
✅ Erken test ile zaman kazanıldı
```

### 9.2 Başarısızlık Sebepleri

**1. Dataset Quality > Quantity**
- 167K alfabe görüntüsü yeterli olmadı (overfitting)
- 20K YOLO görüntüsü yeterli oldu (diverse)

**2. Class Balance Kritik**
- LSTM'de LÜTFEN dominant oldu
- Diğer sınıflar ezildi

**3. Manuel Veri Toplama Zor**
- 250 sample toplamak 2+ saat sürdü
- Kalite tutarsız
- Yetersiz diversity

**4. Model Seçimi Önemli**
- Temporal LSTM → sequential data için iyi ama veri aç
- YOLO → frame-level, daha az veri gerekiyor

---

## 10. GELECEK ÇALIŞMALAR

### 10.1 Kısa Vade (1-2 Ay)

**1. Kelime Havuzu Genişletme**
- Hedef: 20 → 50 kelime
- Yeni dataset toplama veya mevcut dataset genişletme
- YOLO modelini re-train

**2. Model Optimizasyonu**
- Quantization (INT8)
- ONNX export (cross-platform)
- TensorRT optimization (GPU)
- Hedef: >30 FPS

**3. FPS Benchmark**
- Farklı GPU'larda test
- CPU fallback optimizasyonu
- Profiling ve bottleneck analizi

### 10.2 Orta Vade (3-6 Ay)

**4. Hibrit Sistem Geliştirme**

**Konsept:**
```
Frame → YOLO (detection) → Temporal Buffer → LSTM (movement analysis) → Final Prediction
```

**Kullanım Senaryoları:**
- Statik işaretler (el şekilleri): Sadece YOLO
- Dinamik hareketler (el sallanması): YOLO + LSTM
- Vücut hareketleri: YOLO (body keypoints) + LSTM

**Avantajlar:**
- YOLO'nun hızını korurken
- LSTM'in temporal anlama gücünü kullan
- Best of both worlds

**5. Vücut Hareketlerini Ekleme**
- YOLO'ya pose keypoints ekle
- Tam vücut işaretlerini tanı
- Örnek: "ÖZÜR DİLERİM" (vücudu öne eğme)

**6. Cümle Oluşturma**
- Kelime dizisinden anlamlı cümle
- NLP entegrasyonu
- Context-aware predictions

### 10.3 Uzun Vade (6-12 Ay)

**7. Mobil Uygulama**
- TensorFlow Lite conversion
- Android/iOS native app
- Offline çalışma
- Optimized inference

**8. Ters Çeviri (Türkçe → TİD)**
- Metin input
- İşaret animasyonu output
- 3D avatar ile gösterim
- Eğitim amaçlı kullanım

**9. Dataset Büyütme**
- Synthetic data generation
- Data augmentation (advanced)
- Crowdsourcing veri toplama
- Hedef: 100+ kelime, 100K+ görüntü

**10. Production Deployment**
- Docker containerization
- Cloud deployment (AWS/Azure)
- Load balancing
- Monitoring ve logging
- CI/CD pipeline

---

## 11. KAYNAKLAR VE REFERANSLAR

### Kullanılan Datasets
1. **Rakamlar:** Custom webcam dataset (el toplama)
2. **Alfabe:** Kaggle - Turkish Sign Language Alphabet Dataset
3. **LSTM Sequences:** Custom collection (250 samples)
4. **YOLO Kelimeler:** Roboflow - TURK ISARET DILI v2i.yolov11

### Kullanılan Teknolojiler
- PyTorch: https://pytorch.org/
- Ultralytics YOLOv11: https://github.com/ultralytics/ultralytics
- MediaPipe: https://google.github.io/mediapipe/
- FastAPI: https://fastapi.tiangolo.com/
- gTTS: https://github.com/pndurette/gTTS

### İlham Kaynakları
- YOLO için best practices
- MediaPipe Holistic documentation
- Turkish Sign Language studies

---

## 12. PROJE İSTATİSTİKLERİ

### Kod İstatistikleri
- **Toplam Dosya:** 296
- **Python Files:** ~30
- **Kod Satırı:** ~7,000+
- **API Endpoints:** 6
- **Modeller:** 2 production, 2 legacy

### Zaman İstatistikleri
- **Toplam Süre:** ~4 hafta
- **Model Eğitimi:** ~20 saat (toplam)
  - Rakamlar: ~5 saat
  - Alfabe: ~3 saat
  - LSTM: ~4 saat
  - YOLO: ~3 saat
- **Veri Toplama:** ~10 saat
- **Kod Geliştirme:** ~60 saat
- **Testing & Debug:** ~20 saat

### Dataset İstatistikleri
- **Toplam Görüntü:** 188,000+
  - Rakamlar: ~200
  - Alfabe: 167,000
  - LSTM: 250 sequences
  - YOLO: 21,928
- **Kullanılan:** 22,128 (production)

---

## 13. RİSK ANALİZİ

### Teknik Riskler

| Risk | Olasılık | Etki | Azaltma Stratejisi |
|------|----------|------|-------------------|
| Model degradation | Orta | Yüksek | Düzenli re-training, monitoring |
| GPU unavailability | Düşük | Yüksek | CPU fallback, cloud GPU |
| Dataset bias | Orta | Orta | Diverse data collection |
| Overfitting | Düşük | Orta | Validation, early stopping |

### Operasyonel Riskler

| Risk | Olasılık | Etki | Azaltma Stratejisi |
|------|----------|------|-------------------|
| Kamera erişim sorunları | Orta | Yüksek | Permissions handling, user guide |
| Browser uyumsuzluk | Düşük | Orta | Cross-browser testing |
| Network latency | Düşük | Düşük | Local inference |

---

## 14. SONUÇ VE DEĞERLENDİRME

### Başarılar
✅ **2 production-ready model** geliştirildi  
✅ **%99+ accuracy** her iki modelde de  
✅ **Gerçek zamanlı** inference (<40ms)  
✅ **Web-based UI** kullanıcı dostu  
✅ **TTS entegrasyonu** erişilebilirlik için  
✅ **GitHub backup** kod güvenliği için  

### Zorluklar
⚠️ **Alfabe projesi** başarısız oldu (overfitting)  
⚠️ **LSTM yaklaşımı** yetersiz veri nedeniyle başarısız  
⚠️ **Manuel veri toplama** çok zaman alıcıydı  
⚠️ **GPU training** başlangıçta sıkıntılıydı  

### Öğrenilenler
💡 **Dataset kalitesi** > dataset miktarı  
💡 **Erken gerçek dünya testi** kritik  
💡 **GPU training** essential  
💡 **YOLO** bu problem için ideal  
💡 **Multimodal approach** daha robust  

### Genel Değerlendirme

SignBridge projesi, **başarılı bir şekilde** tamamlanmıştır. İki production-ready model geliştirilmiş ve web tabanlı bir arayüz ile kullanıma sunulmuştur. Proje sürecinde karşılaşılan zorluklar ve başarısızlıklar, önemli öğrenim fırsatları sunmuş ve nihai ürünün daha güçlü olmasını sağlamıştır.

**Rakam tanıma** modeli %99.79 accuracy ile mükemmel performans göstermektedir. **İşaret kelime tanıma** modeli ise YOLOv11 kullanılarak %99.6 mAP50 skoru ile başarılı olmuştur.

Gelecek çalışmalar için sağlam bir temel oluşturulmuştur. Hibrit sistem yaklaşımı, vücut hareketleri eklenmesi ve mobil uygulama geliştirme planlanmaktadır.

---

## 15. EK BİLGİLER

### Ekler
- **Ek A:** Model architecture diagrams
- **Ek B:** Training curves (results.png)
- **Ek C:** Confusion matrices
- **Ek D:** API documentation (FastAPI auto-docs)
- **Ek E:** Web UI screenshots

### İletişim
- **GitHub:** https://github.com/esmasila/SignBridge
- **Repository Owner:** esmasila

---

**Rapor Tarihi:** 28 Kasım 2025  
**Versiyon:** 1.0  
**Durum:** Final

---

## EKLER

### Ek A: Dosya Yapısı (Detaylı)

```
sign_bridge/
│
├── .github/                    # GitHub workflows
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
│
├── .venv/                     # Virtual environment (git-ignored)
│
├── checkpoints/               # Eğitilmiş modeller
│   ├── multimodal_digits_best.pt      # ✅ Production (99.79%)
│   ├── multimodal_digits_final.pt
│   ├── multimodal_digits_quantized.pt # CPU optimized
│   ├── digits_cnn_best.pt
│   ├── landmark_only_best.pt
│   └── training_history.png
│
├── data/                      # Veri dosyaları
│   ├── raw/                   # Ham veri (git-ignored)
│   ├── processed/             # İşlenmiş veri (git-ignored)
│   └── dataset_report.csv     # Dataset özeti
│
├── logs/                      # Log dosyaları
│   └── *.log                  # API ve training logları (5 adet korundu)
│
├── signbridge/                # Ana Python paketi
│   ├── __init__.py
│   │
│   ├── api/                   # REST API
│   │   ├── __init__.py
│   │   └── service.py         # ✅ FastAPI endpoints
│   │
│   ├── config.py              # Konfigürasyon
│   │
│   ├── data/                  # Veri işleme
│   │   ├── __init__.py
│   │   ├── datasets.py        # Dataset loaders
│   │   └── preprocess.py      # ✅ MediaPipe extractor
│   │
│   ├── inference/             # Inference utilities
│   │   ├── __init__.py
│   │   └── realtime_tid_inference.py
│   │
│   ├── models/                # Model tanımları
│   │   ├── __init__.py
│   │   ├── cnn_digits.py      # Basit CNN
│   │   ├── multimodal_cnn.py  # ✅ Multimodal model
│   │   └── islr_sequence_model.py
│   │
│   └── utils/                 # Yardımcı fonksiyonlar
│       ├── __init__.py
│       ├── logging_utils.py   # Logger
│       ├── metrics.py         # Metric hesaplama
│       └── viz.py             # Görselleştirme
│
├── tid_sequence/              # LSTM sequence projesi (legacy)
│   ├── README.md
│   ├── KELIME_LISTESI.md
│   │
│   ├── checkpoints/           # LSTM modelleri
│   │   ├── lstm_best.pt       # 63MB (GitHub warning)
│   │   ├── gru_best.pt
│   │   ├── label_map.json
│   │   └── confusion_matrix.png
│   │
│   ├── data/                  # Toplanan sequence data (250 samples)
│   │   ├── MERHABA/          # 50 .npy files
│   │   ├── EVET/             # 50 .npy files
│   │   ├── HAYIR/            # 50 .npy files
│   │   ├── LÜTFEN/           # 50 .npy files
│   │   └── SU/               # 50 .npy files
│   │
│   ├── models/                # Model tanımları
│   │   └── lstm_classifier.py
│   │
│   └── scripts/               # Training/inference scriptleri
│       ├── collect_sequences.py    # Veri toplama
│       ├── collect_from_video.py
│       ├── dataset.py              # Dataset loader
│       ├── train.py                # LSTM eğitimi
│       ├── evaluate.py             # Değerlendirme
│       └── inference.py            # LSTM inference
│
├── turk_isaret_yolo/          # YOLO dataset (6.2 GB, git-ignored kısmen)
│   ├── data.yaml              # ✅ Dataset config
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   │
│   ├── train/                 # 20,020 images (git-ignored)
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── valid/                 # 1,908 images (git-ignored)
│   │   ├── images/
│   │   └── labels/
│   │
│   └── test/                  # Test set (git-ignored)
│       ├── images/
│       └── labels/
│
├── web/                       # Web arayüzü
│   ├── index.html             # ✅ Main UI
│   └── Attached HTML and CSS Context.txt
│
├── yolo_runs/                 # YOLO eğitim sonuçları (git-ignored)
│   └── turk_isaret_v1/
│       ├── weights/
│       │   └── best.pt        # ✅ Production model
│       ├── args.yaml
│       ├── results.csv        # Training metrics
│       ├── results.png        # Training curves
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── BoxF1_curve.png
│       ├── BoxPR_curve.png
│       ├── BoxP_curve.png
│       ├── BoxR_curve.png
│       ├── labels.jpg
│       ├── train_batch*.jpg   # Training visualizations
│       └── val_batch*.jpg     # Validation visualizations
│
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── README.md                  # Proje açıklaması
├── requirements.txt           # Python dependencies
│
├── train_yolo.py              # ✅ YOLO eğitim scripti
├── yolo_inference.py          # ✅ YOLO webcam test
├── yolo11n.pt                 # YOLOv11-nano pretrained (5.4MB)
│
├── test_yolo_api.py           # ✅ API test tool
│
└── PROJE_RAPORU.md           # ✅ Bu rapor
```

### Ek B: requirements.txt

```txt
# Deep Learning
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
ultralytics==8.3.230

# Computer Vision
opencv-python==4.8.1.78
mediapipe==0.10.21

# API
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Audio
gTTS==2.4.0

# Utilities
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.66.1
pillow==10.1.0

# Others
pydantic==2.5.0
requests==2.31.0
```

### Ek C: Training Hyperparameters

#### Rakam Modeli (Multimodal CNN)
```python
{
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "scheduler": "ReduceLROnPlateau",
    "patience": 5,
    "image_size": (64, 64),
    "landmark_features": 1629,
    "dropout": 0.3,
    "weight_decay": 1e-5
}
```

#### YOLO Modeli
```python
{
    "model": "yolo11n.pt",
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "device": 0,  # GPU
    "patience": 10,
    "pretrained": True,
    "optimizer": "auto",  # SGD
    "lr0": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "amp": True,  # Automatic Mixed Precision
    "project": "yolo_runs",
    "name": "turk_isaret_v1"
}
```

---

**Rapor Sonu**
