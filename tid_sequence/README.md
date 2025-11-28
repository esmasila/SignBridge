# TİD Sequence Module

## Genel Bakış

Bu modül Türk İşaret Dili için **işaret sekansı → sınıf etiketi** modelini içerir.

## Amaç

Statik tek karakter tanıma yerine, **zaman içinde gelişen işaret hareketlerini** tanımak:
- ✅ Doğal TİD kullanımına daha yakın
- ✅ Kelime ve kısa cümle düzeyinde tanıma
- ✅ MediaPipe Holistic landmark'larını kullanarak hafif ve hızlı

## Mimari

### Veri Girişi
- **MediaPipe Holistic** landmark çıkarımı:
  - 33 pose landmark (vücut duruşu)
  - 21 + 21 el landmark (sağ ve sol el)
  - 468 yüz landmark (opsiyonel, yüz ifadeleri için)
  - Toplam: ~543 landmark × 3 koordinat (x, y, z) = 1629 özellik

### Model Yapısı
- **LSTM/GRU** tabanlı zaman serisi modeli
- Sekans uzunluğu: 30-60 frame (~1-2 saniye)
- Çıkış: Kelime/cümle sınıfı

### Hedef Kelime Seti (İlk Aşama)

Küçük bir sözlük ile başlanacak:
1. **Temel selamlaşmalar**: MERHABA, GÜNAYDΙN, İYİ AKŞAMLAR, HOŞÇA KAL
2. **Onay/ret**: EVET, HAYIR, TAMAM, OLMAZ
3. **Nezaket**: TEŞEKKÜR EDERİM, LÜTFEN, ÖZÜR DİLERİM
4. **Sorular**: NE, KİM, NEREDE, NASIL, NE ZAMAN, NEDEN
5. **Günlük**: SU, YEMEK, EV, OKUL, ÇALIŞMAK

**Toplam**: ~20-30 kelime/kısa ifade

## Klasör Yapısı

```
tid_sequence/
├── data/
│   ├── raw/              # Ham video kayıtları
│   ├── processed/        # İşlenmiş landmark sekansları (.npy)
│   └── labels.json       # Kelime → sınıf etiket haritası
├── models/
│   ├── lstm_model.py     # LSTM model tanımı
│   ├── gru_model.py      # GRU model tanımı (alternatif)
│   └── __init__.py
├── scripts/
│   ├── collect_data.py   # Webcam'den veri toplama
│   ├── preprocess.py     # Landmark çıkarma ve normalizasyon
│   ├── train.py          # Model eğitimi
│   ├── evaluate.py       # Model değerlendirme
│   └── inference.py      # Gerçek zamanlı tahmin
└── README.md             # Bu dosya
```

## Veri Toplama Stratejisi

1. **Her kelime için**: 50-100 örnek video (farklı kişiler, açılar, hızlar)
2. **Video uzunluğu**: 2-3 saniye
3. **FPS**: 30 (MediaPipe için yeterli)
4. **Preprocessing**: 
   - MediaPipe ile landmark çıkarımı
   - Normalizasyon (koordinatlar 0-1 arasına)
   - Sekans padding/truncation

## Eğitim Planı

### Faz 1: Prototip (Haftalar 1-2)
- 5-10 kelime ile küçük model
- LSTM (2 katman, 128 hidden unit)
- Validation accuracy >85% hedefi

### Faz 2: Genişletme (Haftalar 3-4)
- 20-30 kelime ile tam model
- Hyperparameter optimizasyonu
- Data augmentation (hız değişimi, gürültü ekleme)
- Validation accuracy >90% hedefi

### Faz 3: Production (Hafta 5+)
- Model optimizasyonu (quantization)
- Web API entegrasyonu
- Gerçek zamanlı test ve iyileştirme

## Teknoloji Stack

- **Landmark Extraction**: MediaPipe Holistic
- **Deep Learning**: PyTorch
- **Data Processing**: NumPy, Pandas
- **API**: FastAPI (mevcut altyapı)
- **Deployment**: Docker (mevcut setup)

## Mevcut Sistem ile Entegrasyon

Bu modül mevcut `signbridge/` altyapısını kullanacak:
- `signbridge.data.preprocess.LandmarkExtractor` - MediaPipe wrapper
- `signbridge.api.service` - REST API endpoint'leri
- `signbridge.utils` - Logging, visualization araçları

## Beklenen Sonuçlar

- ✅ Gerçek TİD kullanımına daha yakın
- ✅ Doğal konuşma hızında tanıma
- ✅ Bağlam bilgisi (sekans) sayesinde daha yüksek doğruluk
- ✅ Yeni kelimelere kolayca genişletilebilir

## Sonraki Adımlar

1. **Veri toplama scripti** hazırla (`collect_data.py`)
2. **İlk 5 kelime** için veri topla (50 örnek/kelime)
3. **Basit LSTM modeli** oluştur ve eğit
4. **Baseline accuracy** ölç
5. **Iterasyon**: Veri artırma + model iyileştirme

---

**Not**: Rakam tanıma modeli (`multimodal_digits_best.pt`) hala aktif ve çalışıyor. Bu modül buna ek olarak geliştirilecek.
