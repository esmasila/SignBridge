"""
AUTSL YOLO Eğitimi - 226 Sınıflık Türk İşaret Dili
===================================================
Checkpoint destekli, durdurup devam edilebilir eğitim.

Kullanım:
    Yeni eğitim:     python train_autsl_checkpoint.py
    Devam etmek:     python train_autsl_checkpoint.py --resume
"""

from ultralytics import YOLO
import os
import torch
import argparse
import sys

def check_gpu():
    """GPU durumunu kontrol et"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
        
        # VRAM'e göre batch size öner
        if vram >= 8:
            batch = 16
        elif vram >= 6:
            batch = 8
        else:
            batch = 4
        print(f"✅ Önerilen batch size: {batch}")
        return 'cuda', batch
    else:
        print("⚠️ GPU bulunamadı, CPU kullanılacak (çok yavaş olacak!)")
        return 'cpu', 4

def find_last_checkpoint():
    """En son checkpoint'i bul"""
    weights_dir = 'autsl_runs/autsl_226_final/weights'
    
    if not os.path.exists(weights_dir):
        return None
    
    # last.pt var mı?
    last_pt = os.path.join(weights_dir, 'last.pt')
    if os.path.exists(last_pt):
        return last_pt
    
    # epoch*.pt dosyaları var mı?
    epoch_files = [f for f in os.listdir(weights_dir) if f.startswith('epoch') and f.endswith('.pt')]
    if epoch_files:
        # En son epoch'u bul
        epoch_files.sort(key=lambda x: int(x.replace('epoch', '').replace('.pt', '')))
        return os.path.join(weights_dir, epoch_files[-1])
    
    return None

def main():
    parser = argparse.ArgumentParser(description='AUTSL YOLO Eğitimi')
    parser.add_argument('--resume', action='store_true', help='Son checkpoint\'ten devam et')
    parser.add_argument('--epochs', type=int, default=50, help='Toplam epoch sayısı')
    parser.add_argument('--batch', type=int, default=None, help='Batch size (otomatik belirlenir)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  AUTSL YOLO EĞİTİMİ - 226 SINIF TÜRK İŞARET DİLİ")
    print("  Checkpoint destekli - Durdurup devam edilebilir")
    print("=" * 70)
    
    # GPU kontrolü
    device, auto_batch = check_gpu()
    batch_size = args.batch if args.batch else auto_batch
    
    # Resume modu kontrolü
    checkpoint = None
    if args.resume:
        checkpoint = find_last_checkpoint()
        if checkpoint:
            print(f"\n🔄 Devam edilecek checkpoint: {checkpoint}")
        else:
            print("\n⚠️ Checkpoint bulunamadı, yeni eğitim başlatılacak!")
    
    print(f"\n📊 Eğitim Parametreleri:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Resume: {args.resume}")
    print()
    
    # Bellek temizliği
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("🧹 GPU belleği temizlendi")
    
    # Model yükle
    if checkpoint and args.resume:
        print(f"\n📥 Checkpoint yükleniyor: {checkpoint}")
        model = YOLO(checkpoint)
        
        # Batch size değiştiyse uyarı ver
        try:
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
            orig_batch = ckpt.get('train_args', {}).get('batch', 'bilinmiyor')
            if orig_batch != batch_size and orig_batch != 'bilinmiyor':
                print(f"⚠️ UYARI: Orijinal batch={orig_batch}, şimdiki batch={batch_size}")
                print(f"   Batch değişikliği sorun yaratabilir!")
        except:
            pass
    else:
        print("\n📥 Yeni model yükleniyor: yolo11n.pt")
        model = YOLO('yolo11n.pt')
    
    print("\n🚀 Eğitim başlıyor...\n")
    print("-" * 70)
    
    try:
        # Eğitim parametreleri - 226 sınıf için optimize edilmiş
        results = model.train(
            data='autsl_yolo/data.yaml',
            epochs=args.epochs,
            imgsz=640,
            batch=batch_size,
            patience=15,  # 15 epoch iyileşme yoksa dur
            save=True,
            save_period=5,  # ⭐ Her 5 epoch'ta checkpoint kaydet!
            device=device,
            workers=4,
            cache=False,  # RAM yetersiz (241 GB gerekiyor)
            project='autsl_runs',
            name='autsl_226_final',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,  # Başlangıç learning rate
            lrf=0.01,   # Final learning rate = lr0 * lrf
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,  # Rotasyon kapalı - el işaretleri için önemli
            translate=0.1,
            scale=0.5,
            fliplr=0.0,  # Yatay çevirme kapalı - el işaretleri simetrik değil!
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            amp=True,  # Mixed precision - daha hızlı
            verbose=True,
            close_mosaic=10,
            resume=args.resume and checkpoint is not None,
            
            # Gradient clipping - gradient patlamasını önler
            # (YOLO'da varsayılan olarak var)
        )
        
        print("\n" + "=" * 70)
        print("  ✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
        
        # Sonuçları göster
        best_model = 'autsl_runs/autsl_226_final/weights/best.pt'
        last_model = 'autsl_runs/autsl_226_final/weights/last.pt'
        
        print(f"\n📁 Model Dosyaları:")
        print(f"   En iyi model: {best_model}")
        print(f"   Son model: {last_model}")
        
        # Validation sonuçları
        if os.path.exists(best_model):
            print("\n📊 Validation Sonuçları:")
            val_model = YOLO(best_model)
            val_results = val_model.val(data='autsl_yolo/data.yaml')
            print(f"   mAP50: {val_results.box.map50:.4f}")
            print(f"   mAP50-95: {val_results.box.map:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Eğitim kullanıcı tarafından durduruldu!")
        print("💡 Devam etmek için: python train_autsl_checkpoint.py --resume")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        print("💡 Devam etmek için: python train_autsl_checkpoint.py --resume")
        raise

if __name__ == "__main__":
    main()
