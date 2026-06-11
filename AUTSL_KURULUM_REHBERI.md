# AUTSL Dataset Kurulum Rehberi

## 📥 Adım 1: Dosyaları İndirin

ChaLearn sayfasından şu dosyaları indirin:
https://chalearnlap.cvc.uab.cat/dataset/40/description/

### İndirilecek Dosyalar:
- ✅ Train data (with labels) - ~20-30 GB
- ✅ Validation data (without labels) - ~4-5 GB
- ✅ Validation labels - Küçük dosya
- ✅ Test data (without labels) - ~3-4 GB
- ✅ Test labels - Küçük dosya
- ✅ Class IDs correspondence - CSV dosyası

## 📂 Adım 2: Dosyaları Kopyalayın

İndirdiğiniz TÜM dosyaları şu klasöre kopyalayın:
```
C:\Projects\sign_bridge\autsl_encrypted\
```

### Örnek dosya isimleri:
- `train_data.zip` veya `train.tar.gz`
- `validation_data.zip`
- `validation_labels.csv`
- `test_data.zip`
- `test_labels.csv`
- `SignList_ClassId_TR_EN.csv`

## 🔓 Adım 3: Decrypt Edin

Terminal'de şu komutu çalıştırın:
```powershell
python decrypt_autsl.py
```

**ŞİFRELERİ MANUEL GİRMENİZE GEREK YOK!** Script otomatik kullanacak.

### Gereksinimler:
Script şu araçlardan birini kullanır (en az biri kurulu olmalı):
- **OpenSSL** (önerilen)
- **7-Zip**

#### OpenSSL Kurulumu (Windows):
```powershell
# Chocolatey ile:
choco install openssl

# Veya manuel: https://slproweb.com/products/Win32OpenSSL.html
```

#### 7-Zip Kurulumu:
```powershell
# Chocolatey ile:
choco install 7zip

# Veya manuel: https://www.7-zip.org/download.html
```

## 📊 Adım 4: Veri İşleme

Decrypt sonrası dosyalar `autsl_original/` klasörüne çıkacak.

### A) YOLO Formatına Çevir:
```powershell
python prepare_autsl_yolo_from_videos.py
```

### B) MediaPipe + LSTM için:
```powershell
python prepare_autsl_landmarks.py
```

## 📁 Sonuç Klasör Yapısı

```
sign_bridge/
├── autsl_encrypted/          # İndirdiğiniz şifreli dosyalar
│   ├── train_data.zip
│   ├── validation_data.zip
│   └── ...
├── autsl_original/           # Decrypt edilmiş dosyalar
│   ├── train/
│   │   ├── signer0_sample0_color.mp4
│   │   ├── signer0_sample0_depth.mp4
│   │   └── ...
│   ├── validation/
│   ├── test/
│   └── labels/
└── autsl_yolo/               # YOLO formatı (yeniden oluşturulacak)
    ├── train/
    ├── val/
    └── test/
```

## 🎯 Decryption Keys

Script otomatik kullanacak, ancak referans için:
- **Train:** `MdG3z6Eh1t`
- **Validation:** `bhRY5B9zS2`
- **Validation labels:** `zYX5W7fZ`
- **Test:** `ds6Kvdus3o`

## ❓ Sorun Giderme

### "OpenSSL bulunamadı" hatası:
```powershell
choco install openssl
# Veya 7-Zip kurun
```

### "Decryption failed" hatası:
- Dosya adlarını kontrol edin
- Dosyalar tam inmiş mi kontrol edin (boyutlara bakın)
- İnternet bağlantısı kesilinmişse yeniden indirin

### Boyut sorunları:
- Train data: ~20-30 GB
- Her şey için toplam ~50-60 GB boş alan gerekir

## 📞 Yardım

Sorun yaşarsanız:
1. Hata mesajını kopyalayın
2. Hangi dosyada sorun olduğunu belirtin
3. `python decrypt_autsl.py` çıktısını gönderin
