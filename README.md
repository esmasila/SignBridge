# 🤟 SignBridge - Türk İşaret Dili Gerçek Zamanlı Çevirici

**SignBridge**, gerçek zamanlı olarak **Türk İşaret Dili (TİD) → Türkçe Metin/Ses** çevirisi yapan, tamamen cihaz üzerinde (on-device) çalışan bir MVP projesidir.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Kurulum](#kurulum)
- [Veri Setlerinin Hazırlanması](#veri-setlerinin-hazırlanması)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Model Eğitimi](#model-eğitimi)
- [API Kullanımı](#api-kullanımı)
- [Sınıf Haritalama](#sınıf-haritalama)
- [Gelişmiş Özellikler](#gelişmiş-özellikler)
- [Sorun Giderme](#sorun-giderme)
- [Katkıda Bulunma](#katkıda-bulunma)

---

## ✨ Özellikler

- ✅ **Gerçek Zamanlı Çeviri**: Webcam ile anlık TİD → Türkçe çeviri
- ✅ **On-Device İşleme**: Tüm işlemler cihazda, internet gerektirmez
- ✅ **MediaPipe Holistic**: 543 landmark (yüz, vücut, eller) çıkarımı
- ✅ **Çoklu Model Desteği**: CNN, LSTM, GRU, Transformer
- ✅ **Türkçe Çıktı**: Tüm tahminler Türkçe metin olarak
- ✅ **Text-to-Speech**: İsteğe bağlı sesli okuma (Türkçe)
- ✅ **REST API**: FastAPI ile web servisi
- ✅ **Gizlilik**: Ham video hiçbir zaman diske kaydedilmez veya sunucuya gönderilmez
- ✅ **Modüler Yapı**: Kolay genişletilebilir mimari
- ✅ **Performans**: ≥25 FPS, ≤500ms gecikme

---

## 🖥️ Sistem Gereksinimleri

### Minimum Gereksinimler

- **Python**: 3.10 veya üzeri
- **İşlemci**: 4 çekirdekli CPU
- **RAM**: 8 GB
- **Disk**: 5 GB boş alan
- **Webcam**: USB veya yerleşik kamera
- **İşletim Sistemi**: Windows 10/11, Linux, macOS

### Önerilen Gereksinimler

- **GPU**: NVIDIA CUDA destekli (RTX 2060 veya üzeri)
- **RAM**: 16 GB
- **Python**: 3.11

---

## 🚀 Kurulum

### 1. Python Ortamı Oluşturma

#### Windows (PowerShell)

```powershell
# Proje klasörüne gidin
cd C:\Users\leven\OneDrive\Masaüstü\sign_bridge

# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı aktifleştirin
.\venv\Scripts\Activate.ps1

# Not: Script execution policy hatası alırsanız:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Linux / macOS

```bash
# Proje klasörüne gidin
cd ~/sign_bridge

# Sanal ortam oluşturun
python3 -m venv venv

# Sanal ortamı aktifleştirin
source venv/bin/activate
```

### 2. Bağımlılıkları Yükleme

```bash
# Tüm gerekli kütüphaneleri yükleyin
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. PyTorch CUDA Desteği (Opsiyonel - GPU Kullanıcıları İçin)

```bash
# CUDA 11.8 için
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 için
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Kurulum Doğrulama

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import mediapipe; print('MediaPipe: OK')"
python -c "import cv2; print('OpenCV: OK')"
```

---

## 📂 Veri Setlerinin Hazırlanması

### Mevcut Veri Seti (Sign-Language-Digits-Dataset)

Bilgisayarınızda bulunan rakam veri setini şu konuma yerleştirin:

```
sign_bridge/
└── data/
    └── raw/
        └── Sign-Language-Digits-Dataset-master/
            └── Dataset/
                ├── 0/          # Sıfır işareti görüntüleri
                ├── 1/          # Bir işareti görüntüleri
                ├── 2/          # İki işareti görüntüleri
                ...
                └── 9/          # Dokuz işareti görüntüleri
```

### İleride Kullanılacak Veri Setleri

#### AUTSL (Turkish Sign Language Dataset)

1. AUTSL veri setini indirin: [AUTSL Link](https://www.kaggle.com/datasets/signers/autsl)
2. İndirilen dosyayı şuraya yerleştirin:
   ```
   sign_bridge/data/raw/AUTSL/
   ```

**Özellikler:**
- 226 farklı işaret
- 43 işaretçi
- ~38,000 video örneği

#### BosphorusSign22k

1. BosphorusSign22k veri setini indirin: [BosphorusSign Link](http://bosphorus.ee.boun.edu.tr/)
2. İndirilen dosyayı şuraya yerleştirin:
   ```
   sign_bridge/data/raw/BosphorusSign22k/
   ```

**Özellikler:**
- 22,000+ işaret videosu
- Geniş TİD kelime hazinesi

> **NOT**: Bu büyük veri setlerini kullanmak için `preprocess.py` dosyasındaki TODO bölümlerini tamamlamanız gerekecektir.

---

## 📊 Veri Ön İşleme

### MediaPipe ile Landmark Çıkarma

```bash
# Rakam veri setini işleyin
python -m signbridge.data.preprocess
```

Bu komut:
1. Dataset/0, Dataset/1, ..., Dataset/9 klasörlerindeki tüm görüntüleri tarar
2. Her görüntüden MediaPipe Holistic ile landmark'ları çıkarır
3. Sonuçları `data/processed/digits_landmarks.npz` dosyasına kaydeder

**Çıktı:**
```
data/
└── processed/
    └── digits_landmarks.npz    # İşlenmiş landmark verileri
```

---

## 🎓 Model Eğitimi

### 1. CNN Modeli (Rakamlar 0-9)

Basit bir CNN ile işaret dili rakamlarını sınıflandırın:

```bash
python -m signbridge.training.train_digits
```

**Eğitim Parametreleri:**
- Epoch: 50 (erken durdurma ile)
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Data augmentation: Aktif

**Çıktılar:**
```
checkpoints/
├── digits_cnn_best.pt        # En iyi model
├── digits_cnn_final.pt       # Son model
├── training_history.png      # Loss/Accuracy grafikleri
└── confusion_matrix.png      # Karmaşıklık matrisi

logs/
└── train_digits_*.log        # Eğitim logları
```

### 2. Sekans Modeli (LSTM/GRU/Transformer)

Landmark sekanslarından işaret tahmini:

```bash
# LSTM modeli
python -m signbridge.training.train_sequence_model --model lstm

# GRU modeli
python -m signbridge.training.train_sequence_model --model gru

# Transformer modeli
python -m signbridge.training.train_sequence_model --model transformer
```

**Not**: Sekans modelleri için önce landmark verilerinin hazır olması gerekir.

---

## 🎥 Gerçek Zamanlı Çeviri

### Hızlı Başlangıç

```bash
# Varsayılan ayarlarla başlatın
python run_realtime_demo.py
```

### Gelişmiş Kullanım

```bash
# CNN modeli ile
python -m signbridge.inference.realtime_tid_inference \
    --model checkpoints/digits_cnn_best.pt \
    --type cnn \
    --confidence 0.6

# LSTM modeli ile
python -m signbridge.inference.realtime_tid_inference \
    --model checkpoints/lstm_sequence_best.pt \
    --type lstm \
    --confidence 0.7

# TTS olmadan
python -m signbridge.inference.realtime_tid_inference \
    --no-tts
```

### Klavye Kısayolları

| Tuş | İşlev |
|-----|-------|
| `q` | Programı kapat |
| `t` | TTS açık/kapat |

### Ekran Bilgileri

Gerçek zamanlı demo çalışırken ekranda:
- ✅ **Tahmin**: Tanınan işaret (Türkçe)
- ✅ **Güven**: Tahmin güven skoru (%)
- ✅ **FPS**: Anlık kare hızı
- ✅ **Model**: Kullanılan model tipi
- ✅ **TTS**: Sesli okuma durumu

---

## 🌐 API Kullanımı

### FastAPI Servisini Başlatma

```bash
# API'yi başlatın
python -m signbridge.api.service

# Alternatif (doğrudan uvicorn ile)
uvicorn signbridge.api.service:app --host 127.0.0.1 --port 8000 --reload
```

API şu adreste çalışacaktır: **http://127.0.0.1:8000**

### API Dokümantasyonu

Tarayıcınızda açın:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Endpoint'ler

#### 1. Sağlık Kontrolü

```bash
curl http://127.0.0.1:8000/health
```

**Yanıt:**
```json
{
  "status": "healthy",
  "api_version": "0.1.0",
  "model_loaded": true,
  "model_type": "cnn",
  "device": "cuda"
}
```

#### 2. Model Bilgileri

```bash
curl http://127.0.0.1:8000/model/info
```

**Yanıt:**
```json
{
  "model_type": "cnn",
  "num_classes": 10,
  "classes": {
    "0": "sıfır",
    "1": "bir",
    "2": "iki",
    ...
  }
}
```

#### 3. Görüntüden Tahmin (Base64)

```bash
# Base64 encode edilmiş görüntü ile
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "BASE64_ENCODED_IMAGE_HERE"
  }'
```

#### 4. Görüntü Dosyası Upload

```bash
# Dosya upload ederek
curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@path/to/image.jpg"
```

**Yanıt:**
```json
{
  "success": true,
  "prediction_id": 5,
  "prediction_text": "beş",
  "confidence": 0.95,
  "message": "Tahmin başarılı"
}
```

### Python ile API Kullanımı

```python
import requests
import base64

# Görüntü dosyasını oku
with open("test_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# API'ye istek gönder
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"image_base64": image_base64}
)

# Sonucu al
result = response.json()
print(f"Tahmin: {result['prediction_text']}")
print(f"Güven: {result['confidence']:.2%}")
```

---

## 🗺️ Sınıf Haritalama (Model Çıktısı → Türkçe)

Model çıktıları otomatik olarak Türkçe'ye çevrilir. Haritalama `config.py` içinde tanımlanmıştır:

```python
CLASS_TO_TURKISH = {
    0: "sıfır",
    1: "bir",
    2: "iki",
    3: "üç",
    4: "dört",
    5: "beş",
    6: "altı",
    7: "yedi",
    8: "sekiz",
    9: "dokuz",
    
    # İleride AUTSL ve BosphorusSign22k için:
    # 10: "merhaba",
    # 11: "teşekkür ederim",
    # 12: "günaydın",
    # 13: "iyi akşamlar",
    # 14: "evet",
    # 15: "hayır",
    # ... (226 sınıfa kadar genişletilebilir)
}
```

### Yeni İşaretler Ekleme

1. `config.py` dosyasını açın
2. `CLASS_TO_TURKISH` sözlüğüne yeni sınıf ekleyin:
   ```python
   CLASS_TO_TURKISH = {
       ...
       10: "merhaba",
       11: "nasılsın",
   }
   ```
3. Modeli yeni sınıflarla yeniden eğitin

---

## 📁 Proje Yapısı

```
sign_bridge/
│
├── signbridge/                     # Ana Python paketi
│   ├── __init__.py
│   ├── config.py                   # Tüm ayarlar
│   │
│   ├── data/                       # Veri işleme
│   │   ├── __init__.py
│   │   ├── datasets.py             # PyTorch Dataset sınıfları
│   │   └── preprocess.py           # MediaPipe landmark çıkarma
│   │
│   ├── models/                     # Model tanımları
│   │   ├── __init__.py
│   │   ├── cnn_digits.py           # CNN (rakamlar)
│   │   └── islr_sequence_model.py  # LSTM/GRU/Transformer
│   │
│   ├── training/                   # Eğitim scriptleri
│   │   ├── __init__.py
│   │   ├── train_digits.py         # CNN eğitimi
│   │   └── train_sequence_model.py # Sekans eğitimi
│   │
│   ├── inference/                  # Gerçek zamanlı çıkarım
│   │   ├── __init__.py
│   │   └── realtime_tid_inference.py
│   │
│   ├── api/                        # FastAPI servisi
│   │   ├── __init__.py
│   │   └── service.py
│   │
│   └── utils/                      # Yardımcı fonksiyonlar
│       ├── __init__.py
│       ├── logging_utils.py
│       ├── metrics.py
│       └── viz.py
│
├── data/                           # Veri dizini
│   ├── raw/                        # Ham veri setleri
│   │   ├── Sign-Language-Digits-Dataset-master/
│   │   ├── AUTSL/                  # (Manuel indirin)
│   │   └── BosphorusSign22k/      # (Manuel indirin)
│   └── processed/                  # İşlenmiş veriler
│       └── digits_landmarks.npz
│
├── checkpoints/                    # Eğitilmiş modeller
│   ├── digits_cnn_best.pt
│   └── lstm_sequence_best.pt
│
├── logs/                           # Log dosyaları
│
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
└── run_realtime_demo.py           # Hızlı demo script
```

---

## ⚙️ Gelişmiş Özellikler

### Konfigürasyon Ayarları

`signbridge/config.py` dosyasında tüm ayarları özelleştirebilirsiniz:

```python
# MediaPipe ayarları
MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1,  # 0: Lite, 1: Full, 2: Heavy
}

# Eğitim ayarları
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "patience": 10,
}

# Çıkarım ayarları
INFERENCE_CONFIG = {
    "confidence_threshold": 0.6,
    "smoothing_window": 5,
    "enable_tts": True,
    "camera_id": 0,  # Farklı kamera için değiştirin
}
```

### Özel Veri Seti ile Eğitim

1. Veri setinizi `data/raw/custom_dataset/` klasörüne yerleştirin
2. `datasets.py` içinde yeni bir Dataset sınıfı oluşturun
3. `train_*.py` scriptlerini yeni dataset ile çalıştırın

### Model Hyperparameter Tuning

```bash
# Farklı learning rate ile
python -m signbridge.training.train_digits --lr 0.0001

# Farklı batch size ile
python -m signbridge.training.train_digits --batch-size 64
```

---

## 🔧 Sorun Giderme

### Yaygın Sorunlar ve Çözümleri

#### 1. Import Hataları

**Sorun**: `ModuleNotFoundError: No module named 'torch'`

**Çözüm**:
```bash
pip install torch torchvision torchaudio
```

#### 2. Kamera Açılamıyor

**Sorun**: `Kamera açılamadı!`

**Çözüm**:
- Kamera bağlantısını kontrol edin
- `config.py` içinde `camera_id` değerini değiştirin (0, 1, 2...)
- Windows'ta kamera izinlerini kontrol edin

#### 3. CUDA Hatası

**Sorun**: `CUDA out of memory`

**Çözüm**:
- `config.py` içinde `batch_size` değerini düşürün
- Veya CPU modunda çalıştırın: `device = "cpu"`

#### 4. MediaPipe Hatası

**Sorun**: `MediaPipe landmarks bulunamadı`

**Çözüm**:
- İyi aydınlatma koşulları sağlayın
- Kameranın net görüntü aldığından emin olun
- `min_detection_confidence` değerini düşürün (config.py)

#### 5. Model Bulunamadı

**Sorun**: `Model dosyası bulunamadı`

**Çözüm**:
```bash
# Önce modeli eğitin
python -m signbridge.training.train_digits
```

#### 6. TTS Çalışmıyor

**Sorun**: `TTS başlatılamadı`

**Çözüm**:
```bash
# pyttsx3'ü yeniden yükleyin
pip uninstall pyttsx3
pip install pyttsx3

# Windows'ta gerekiyorsa
pip install pywin32
```

### Loglara Bakma

Sorun yaşıyorsanız, log dosyalarını kontrol edin:

```bash
# Windows
type logs\*.log

# Linux/macOS
cat logs/*.log
```

---

## 📈 Performans Metrikleri

### Hedef Metrikler

| Metrik | Hedef | Mevcut (Rakamlar) |
|--------|-------|-------------------|
| Doğruluk (Accuracy) | ≥ %75 | %90+ |
| Top-5 Accuracy | ≥ %90 | %98+ |
| Gecikme | ≤ 500 ms | ~100 ms |
| FPS | ≥ 25 | 30+ |

### Test Sonuçları Görüntüleme

Eğitim sonrası oluşturulan grafikleri kontrol edin:

```
checkpoints/
├── training_history.png      # Loss ve accuracy grafikleri
└── confusion_matrix.png      # Sınıf karışıklık matrisi
```

---

## 🔐 Gizlilik ve Güvenlik

SignBridge, gizliliğinizi korumak için tasarlanmıştır:

✅ **Ham video hiçbir zaman diske kaydedilmez**  
✅ **Veriler internet üzerinden gönderilmez**  
✅ **Tüm işlemler cihazda (on-device) gerçekleşir**  
✅ **Sadece landmark koordinatları işlenir** (görüntü değil)

`config.py` içinde gizlilik ayarları:

```python
PRIVACY_CONFIG = {
    "save_raw_video": False,
    "save_frames": False,
    "send_to_server": False,
    "in_memory_only": True
}
```

---

## 🐳 Docker Deployment

SignBridge Docker ile kolayca deploy edilebilir:

### Quick Start

```bash
# 1. Docker Compose ile başlatın (GPU desteği ile)
docker-compose up -d

# 2. API'nin hazır olup olmadığını kontrol edin
curl http://localhost:5000/health

# 3. Web UI'yi açın
# Tarayıcınızda: http://localhost:5000
```

### Manuel Docker Build

```bash
# Docker image'ı build edin
docker build -t signbridge:latest .

# Çalıştırın (GPU destekli)
docker run --gpus all -p 5000:5000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/yolo_runs:/app/yolo_runs:ro \
  -v $(pwd)/logs:/app/logs \
  signbridge:latest
```

### Sistem Gereksinimleri (Docker)

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Container Toolkit**: GPU kullanımı için
  ```bash
  # Ubuntu/Debian için NVIDIA Container Toolkit kurulumu
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

### Environment Variables

Docker deployment için önemli environment variable'lar:

```bash
# .env dosyası oluşturun
cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
LOG_LEVEL=INFO

# Model Paths
CHECKPOINT_DIR=/app/checkpoints
YOLO_RUNS_DIR=/app/yolo_runs

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
EOF
```

### Docker Services

`docker-compose.yml` 2 servis içerir:

1. **signbridge**: Ana API servisi (port 5000)
   - GPU desteği (NVIDIA)
   - Health check: `/health` endpoint
   - Auto-restart: unless-stopped

2. **redis**: Cache servisi (port 6379, gelecekteki kullanım için)
   - Persistent volume
   - Alpine image (hafif)

### Production Deployment

Production ortamı için öneriler:

```yaml
# docker-compose.prod.yml
services:
  signbridge:
    environment:
      - LOG_LEVEL=WARNING
      - API_WORKERS=4  # Çoklu worker
    deploy:
      replicas: 2  # Load balancing için
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Cloud Deployment

#### AWS EC2 + Docker

```bash
# 1. GPU destekli EC2 instance başlatın (g4dn.xlarge veya üzeri)
# 2. NVIDIA drivers ve Docker yükleyin
# 3. Repository'yi klonlayın
git clone https://github.com/your-username/sign_bridge.git
cd sign_bridge

# 4. Model dosyalarını upload edin
scp -r checkpoints/ ubuntu@ec2-instance:/home/ubuntu/sign_bridge/
scp -r yolo_runs/ ubuntu@ec2-instance:/home/ubuntu/sign_bridge/

# 5. Docker Compose ile başlatın
docker-compose up -d

# 6. Nginx reverse proxy (opsiyonel)
# SSL sertifikası için Let's Encrypt kullanın
```

#### Google Cloud Run (CPU-only)

```bash
# 1. Container Registry'ye push edin
gcloud builds submit --tag gcr.io/PROJECT_ID/signbridge

# 2. Cloud Run'a deploy edin
gcloud run deploy signbridge \
  --image gcr.io/PROJECT_ID/signbridge \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --port 5000
```

### Monitoring

Docker deployment için monitoring:

```bash
# Container loglarını izleyin
docker-compose logs -f signbridge

# Kaynak kullanımını kontrol edin
docker stats signbridge

# Health check
watch -n 5 'curl -s http://localhost:5000/health | jq'
```

### Troubleshooting

**GPU algılanmıyor:**
```bash
# NVIDIA Container Toolkit kurulu mu kontrol edin
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Model yüklenmiyor:**
```bash
# Volume mount'ları kontrol edin
docker-compose exec signbridge ls -la /app/checkpoints
docker-compose exec signbridge ls -la /app/yolo_runs
```

**API yanıt vermiyor:**
```bash
# Container içine girin
docker-compose exec signbridge bash

# Log dosyalarını kontrol edin
cat /app/logs/api_service.log
```

---

## 🏥 Hastane İletişim Modu

SignBridge, işitme engelli bireylerin hastanelerde yaşadığı iletişim sorununu çözmek için özel bir mod içerir.

### Özellikler

- **Metin → TİD Animasyon**: Türkçe metni kelime kelime işaret dili animasyonuna çevirir
- **Hızlı İfadeler**: Doktor ve hasta için hazır cümleler
- **İki Mod**:
  - 👨‍⚕️ **Doktor Modu**: Sağlık personeli için hızlı iletişim
  - 🤟 **Hasta Modu**: İşitme engelli bireyler için kolay ifadeler
- **Emoji Animasyonlar**: Her TİD kelimesi için görsel temsil
- **Canlı Çeviri**: Anlık metin → işaret çevirisi

### Kullanım

#### Web Arayüzü

1. API'yi başlatın:
```bash
python -m signbridge.api.service
```

2. Tarayıcıda açın:
```
http://localhost:5000/hospital_mode.html
```

3. Kullanım:
   - Doktor/Hasta modunu seçin
   - Hızlı ifadelerden birini seçin veya kendi mesajınızı yazın
   - "İşaret Diline Çevir" butonuna tıklayın
   - Mesajınız kelime kelime animasyonlu olarak gösterilecek

#### API Endpoint

```bash
# Metin → TİD kelime listesi
curl -X POST http://localhost:5000/text-to-sign \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba nasılsınız"}'
```

**Yanıt:**
```json
{
  "success": true,
  "original_text": "Merhaba nasılsınız",
  "tid_words": ["Merhaba", "Nasil"],
  "unmapped_words": [],
  "message": "2 kelime başarıyla çevrildi"
}
```

### Desteklenen TİD Kelimeleri

Hastane modu 20 temel TİD kelimesini destekler:

| Kategori | Kelimeler |
|----------|-----------|
| **Selamlaşma** | Merhaba, Teşekkürler |
| **Onay/Red** | Evet, Hayır, Tamam, Dur |
| **Sorular** | Nasıl, Nerede |
| **Aile** | Anne, Baba, Kardeş, Arkadaş |
| **Günlük** | Yemek, İçmek, Tuvalet, Ev, Telefon |
| **Duygular** | İyi, Kötü, Özür Dilemek |

### Hızlı İfadeler

**Doktor Modu:**
- Merhaba
- Nasılsınız
- Nerede ağrı var
- Tamam anlıyorum
- Lütfen bekleyin
- İyi hissediyor musunuz
- Teşekkür ederim
- Geçmiş olsun

**Hasta Modu:**
- Merhaba
- Yardım eder misiniz
- Tuvalet nerede
- Su içmek istiyorum
- Ağrı var
- İyi hissediyorum
- Kötü hissediyorum
- Teşekkür ederim

### Teknik Detaylar

**Kelime Eşleştirme:**
- Türkçe normalizasyon (ğ→g, ü→u, vb.)
- Eş anlamlı kelimeler (selam→Merhaba, sağol→Teşekkürler)
- Otomatik çoğul/tekil dönüşümü (annem→Anne)

**Animasyon:**
- Her kelime ~1.5 saniye
- Kelimeler arası 0.8 saniye bekleme
- İlerleme çubuğu
- Aktif/tamamlanmış kelime vurgusu

**API Entegrasyonu:**
- Endpoint: `POST /text-to-sign`
- OpenAPI dokümantasyonu: `http://localhost:5000/api/docs#Hospital%20Mode`
- Swagger UI ile test edilebilir

### Örnek Senaryolar

**Senaryo 1: Doktor Muayenesi**
```
Doktor yazar: "Merhaba nasılsınız"
→ Sistem gösterir: 👋 Merhaba → ❓ Nasıl
→ Hasta anlayıp tepki verir
```

**Senaryo 2: Hasta Talebi**
```
Hasta seçer: "Tuvalet nerede"
→ Sistem gösterir: 🚻 Tuvalet → 📍 Nerede
→ Hemşire anlar ve yönlendirir
```

**Senaryo 3: Ağrı İletişimi**
```
Doktor yazar: "Nerede ağrı var"
→ Sistem gösterir: 📍 Nerede
→ Hasta işaretle gösterir
```

### Gelecek Geliştirmeler

- [ ] Video tabanlı gerçek TİD animasyonları
- [ ] Daha fazla tıbbi terim (150+ kelime)
- [ ] Anatomik bölge seçici (vücut haritası)
- [ ] Sesli komut desteği
- [ ] Çoklu dil desteği (İngilizce, Almanca)
- [ ] Offline mod (PWA)
- [ ] Tablet/mobil optimizasyon

---

## 🚀 Gelecek Geliştirmeler (Roadmap)

- [ ] AUTSL veri seti entegrasyonu
- [ ] BosphorusSign22k veri seti entegrasyonu
- [ ] 226 işaret için genişletilmiş sözlük
- [ ] Cümle tahmin özelliği
- [ ] Mobil uygulama (iOS/Android)
- [x] Web arayüzü ✅
- [ ] Çoklu dil desteği
- [ ] Gerçek zamanlı video kayıt (isteğe bağlı)
- [ ] Model optimizasyonu (ONNX, TensorRT)
- [x] Docker deployment ✅
- [x] REST API ✅
- [ ] Kubernetes orchestration

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Projeye katkıda bulunmak için:

1. Bu repoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

---

## 📝 Lisans

Bu proje eğitim ve araştırma amaçlıdır.

---

## 📧 İletişim

Sorularınız için:
- **GitHub Issues**: Teknik sorunlar için issue açın
- **Email**: [email@example.com]

---

## 🙏 Teşekkürler

Bu proje şu kaynaklardan faydalanmıştır:

- **MediaPipe**: Google MediaPipe ekibine
- **PyTorch**: PyTorch topluluğuna
- **Sign Language Datasets**: Veri seti oluşturanlara
- **Türk İşaret Dili Topluluğu**: Değerli geri bildirimler için

---

## 📚 Referanslar

1. Sincan, O. M., & Keles, H. Y. (2020). AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset. arXiv preprint arXiv:2008.00932.
2. Camgöz, N. C., et al. (2016). BosphorusSign: A Turkish Sign Language Recognition Dataset. In ECCV Workshop on Assistive Computer Vision and Robotics.
3. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv preprint arXiv:1906.08172.

---

<div align="center">

**🤟 SignBridge ile İletişim Engellerini Aşın! 🤟**

Made with ❤️ for the Turkish Sign Language Community

</div>
