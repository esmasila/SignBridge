# TİD Sekans Projesi - Hedef Kelime Listesi

## 20 Kelimelik Sözlük

### Temel İletişim (10 kelime)
1. **MERHABA** - Selamlaşma
2. **EVET** - Onay
3. **HAYIR** - Ret
4. **TEŞEKKÜR** - Nezaket
5. **LÜTFEN** - Rica
6. **TAMAM** - Onay/Anlaşma
7. **BEN** - Kişi zamiri
8. **SEN** - Kişi zamiri
9. **O** - Kişi zamiri
10. **ANLAMADIM** - İletişim sorunu

### Yönlendirme/Komut (3 kelime)
11. **BEKLE** - Durdurma
12. **DUR** - Durdurma
13. **TEKRAR** - Tekrar isteme

### İhtiyaçlar (4 kelime)
14. **SU** - İçecek
15. **YEMEK** - Yiyecek
16. **YARDIM** - Yardım talebi
17. **TELEFON** - İletişim

### Mekan/Durum (3 kelime)
18. **OKUL** - Yer
19. **ACI** - Ağrı/Rahatsızlık (AĞRI kelimesinin eş anlamlısı)
20. **ACIL** - Aciliyet

## Veri Toplama Hedefi

Her kelime için: **50 örnek** × 20 kelime = **1000 toplam örnek**

### Örnek Dağılımı
- Minimum 40 örnek/kelime (zorunlu)
- Hedef 50 örnek/kelime
- Farklı hızlar, açılar, el kullanımları

### Kayıt Formatı
```
tid_sequence/data/
├── MERHABA/     (50 .npy dosyası)
├── EVET/        (50 .npy dosyası)
├── HAYIR/       (50 .npy dosyası)
└── ...
```

## Toplama Sırası (Önerilen)

### Faz 1: Temel İletişim (Gün 1-2)
1. MERHABA → EVET → HAYIR → TEŞEKKÜR → LÜTFEN

### Faz 2: Kişi Zamirleri (Gün 2)
6. BEN → SEN → O

### Faz 3: Komutlar (Gün 3)
7. TAMAM → BEKLE → DUR → TEKRAR → ANLAMADIM

### Faz 4: İhtiyaçlar (Gün 4)
8. SU → YEMEK → YARDIM → TELEFON

### Faz 5: Mekan/Durum (Gün 5)
9. OKUL → ACI → ACIL

## Kullanım

```bash
# Scripti çalıştır
python tid_sequence/scripts/collect_sequences.py

# Kelimeyi gir (büyük harf)
Etiket: MERHABA

# S tuşu ile kaydet (50 kez tekrarla)
# E tuşu ile sonraki kelimeye geç
```

## İpuçları

1. **Her kelime için farklılık yarat**:
   - Hızlı/yavaş yapma
   - Farklı kamera açıları
   - Farklı ışık koşulları
   - Farklı giysi renkleri

2. **Tutarlılık**:
   - Aynı işaretin farklı varyasyonlarını kaydet
   - TİD standartlarına uy

3. **Kalite kontrolü**:
   - Her 10 kayıtta bir ekranı kontrol et
   - Landmark'ların düzgün yakalandığından emin ol

4. **Dinlenme**:
   - Her 50 kayıtta bir 5-10 dakika mola ver
   - El yorgunluğunu önle

## Notlar

- **ACI** kelimesi: "AĞRI" kelimesinin TİD karşılığı ile aynı
- Tüm kelimeler TİD standartlarına uygun olmalı
- Eğer bir kelimenin işaretini bilmiyorsan, TİD kaynakları kontrol et
