"""
AUTSL YOLO Eğitimi - 226 Sınıflık Türk İşaret Dili
===================================================
"""

from ultralytics import YOLO
import os
import torch

def main():
    print("=" * 60)
    print("AUTSL YOLO Egitimi - 226 Sinif Turk Isaret Dili")
    print("=" * 60)
    
    # GPU kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Cihaz: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Model yükle
    model = YOLO('yolo11n.pt')
    
    # Eğitim parametreleri - 226 sınıf için optimize edilmiş
    results = model.train(
        data='autsl_yolo/data.yaml',
        epochs=50,  # 50 epoch yeterli
        imgsz=640,
        batch=8,  # Daha küçük batch - daha stabil
        patience=10,  # 10 epoch iyileşme yoksa dur
        save=True,
        save_period=5,  # Her 5 epoch'ta checkpoint kaydet!
        device=device,
        workers=4,
        project='autsl_runs',
        name='autsl_226_v2',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0005,  # Daha düşük learning rate - gradient patlaması önlenir
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,  # Daha fazla regularization
        warmup_epochs=5,  # Daha uzun warmup
        warmup_momentum=0.8,
        box=7.5,
        cls=1.0,  # Sınıflandırma kaybı artırıldı - 226 sınıf için önemli
        dfl=1.5,
        hsv_h=0.01,  # Daha az augmentation - overfitting önleme
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,  # Daha az rotasyon
        translate=0.05,  # Daha az kaydırma
        scale=0.3,  # Daha az ölçekleme
        fliplr=0.5,
        mosaic=0.5,  # Daha az mosaic - 226 sınıfta çok karışık olabilir
        mixup=0.0,  # Mixup kapatıldı - çok fazla sınıf karışımı yaratır
        copy_paste=0.0,  # Copy-paste kapatıldı
        amp=True,
        verbose=True,
        close_mosaic=10,  # Son 10 epoch'ta mosaic kapat
        resume=False  # Yeni eğitim
    )
    
    print("\n" + "=" * 60)
    print("Egitim Tamamlandi!")
    print("=" * 60)
    
    # En iyi modeli göster
    best_model = os.path.join('autsl_runs', 'autsl_226_v2', 'weights', 'best.pt')
    print(f"En iyi model: {best_model}")
    
    # Validation sonuçları
    print("\nValidation sonuclari:")
    val_results = model.val()
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

if __name__ == "__main__":
    main()
