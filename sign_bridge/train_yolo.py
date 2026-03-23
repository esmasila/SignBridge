"""
YOLOv11 ile Türk İşaret Dili Eğitimi
20 kelime, 20K görsel
"""

from ultralytics import YOLO
from pathlib import Path


def train_yolo():
    """YOLOv11 modelini eğit"""
    
    print("=" * 70)
    print("YOLOv11 - TÜRK İŞARET DİLİ EĞİTİMİ")
    print("=" * 70)
    
    # Model yükle (YOLOv11 nano - hızlı)
    model = YOLO('yolo11n.pt')  # nano model (en hızlı)
    # Diğer seçenekler: yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large)
    
    print("\n📊 Dataset: turk_isaret_yolo/")
    print("📦 Model: YOLOv11-nano")
    print("🎯 Classes: 20 (Merhaba, Evet, Hayir, vb.)")
    print("\n" + "=" * 70 + "\n")
    
    # Eğitim parametreleri
    results = model.train(
        data='turk_isaret_yolo/data.yaml',
        epochs=50,              # 50 epoch (30-40 dakika)
        imgsz=640,              # 640x640 görsel boyutu
        batch=16,               # Batch size (GPU memory'ye göre)
        device=0,               # GPU (RTX 3050)
        workers=4,              # Veri yükleme thread'leri
        patience=10,            # Early stopping (10 epoch iyileşme yoksa dur)
        save=True,              # Model kaydet
        project='yolo_runs',    # Kayıt klasörü
        name='turk_isaret_v1',  # Deney adı
        exist_ok=True,          # Üzerine yaz
        pretrained=True,        # Pretrained weights kullan
        optimizer='auto',       # Otomatik optimizer seçimi
        verbose=True,           # Detaylı log
        seed=42,                # Reproducibility
        val=True,               # Validation yap
        plots=True,             # Grafik kaydet
        cache=False,            # Cache kullanma (OneDrive'da sorun olabilir)
    )
    
    print("\n" + "=" * 70)
    print("🎉 EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"\n📁 Model kaydedildi: yolo_runs/turk_isaret_v1/weights/best.pt")
    print(f"📊 Sonuçlar: yolo_runs/turk_isaret_v1/")
    
    # Metrikleri göster
    print("\n📈 Son Epoch Sonuçları:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    return model


def test_model(model_path='yolo_runs/turk_isaret_v1/weights/best.pt'):
    """Test modelini değerlendir"""
    
    print("\n" + "=" * 70)
    print("📊 MODEL TEST")
    print("=" * 70)
    
    model = YOLO(model_path)
    
    # Test set'te değerlendir
    metrics = model.val(
        data='turk_isaret_yolo/data.yaml',
        split='test'
    )
    
    print(f"\n✅ Test Sonuçları:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    
    return metrics


if __name__ == "__main__":
    # Eğit
    model = train_yolo()
    
    # Test
    test_model()
    
    print("\n✅ Tamamlandı! Şimdi inference yapabilirsin:")
    print("   python yolo_inference.py")
