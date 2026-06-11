"""
AUTSL YOLO - Ağırlıklardan Devam Etme (Optimizer Reset)
========================================================
Bu script checkpoint'ten sadece ağırlıkları alır, optimizer'ı sıfırlar.
Batch size değişikliği yapıldığında güvenli devam için kullanılır.

Kullanım:
    python resume_from_weights.py --weights epoch5.pt --epochs 50 --batch 16
"""

from ultralytics import YOLO
import os
import torch
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='AUTSL YOLO - Ağırlıklardan Devam')
    parser.add_argument('--weights', type=str, required=True, help='Başlangıç ağırlık dosyası')
    parser.add_argument('--epochs', type=int, default=50, help='Toplam epoch sayısı')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--start-epoch', type=int, default=None, help='Başlangıç epoch (otomatik algılanır)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  AUTSL YOLO - AĞIRLIKLARDAN DEVAM (OPTİMİZER RESET)")
    print("  Batch size değişikliği için güvenli mod")
    print("=" * 70)
    
    # Ağırlık dosyasını bul
    weights_path = args.weights
    if not os.path.exists(weights_path):
        # autsl_runs klasöründe ara
        alt_path = f"autsl_runs/autsl_226_final/weights/{args.weights}"
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"❌ Ağırlık dosyası bulunamadı: {weights_path}")
            sys.exit(1)
    
    print(f"\n📥 Ağırlık dosyası: {weights_path}")
    
    # Epoch bilgisini al
    try:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        saved_epoch = ckpt.get('epoch', 0)
        best_fitness = ckpt.get('best_fitness', 0)
        print(f"   Kaydedilmiş epoch: {saved_epoch + 1}")
        print(f"   Best fitness (mAP50-95): {best_fitness:.4f}")
    except Exception as e:
        print(f"⚠️ Checkpoint bilgisi okunamadı: {e}")
        saved_epoch = 0
    
    # GPU kontrolü
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
        device = 'cuda'
        torch.cuda.empty_cache()
    else:
        print("\n⚠️ GPU bulunamadı!")
        device = 'cpu'
    
    # Kalan epoch hesapla
    start_epoch = args.start_epoch if args.start_epoch else saved_epoch + 1
    remaining_epochs = args.epochs - start_epoch
    
    if remaining_epochs <= 0:
        print(f"\n⚠️ Zaten {args.epochs} epoch tamamlanmış!")
        sys.exit(0)
    
    print(f"\n📊 Eğitim Parametreleri:")
    print(f"   Başlangıç epoch: {start_epoch + 1}")
    print(f"   Hedef epoch: {args.epochs}")
    print(f"   Kalan epoch: {remaining_epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Device: {device}")
    print(f"   ⚠️ Optimizer SIFIRLANACAK (yeni warmup)")
    
    # Model yükle
    print(f"\n📥 Model yükleniyor (sadece ağırlıklar)...")
    model = YOLO(weights_path)
    
    print("\n🚀 Eğitim başlıyor...\n")
    print("-" * 70)
    
    try:
        results = model.train(
            data='autsl_yolo/data.yaml',
            epochs=remaining_epochs,  # Kalan epoch kadar eğit
            imgsz=640,
            batch=args.batch,
            patience=15,
            save=True,
            save_period=5,
            device=device,
            workers=4,
            cache=False,
            project='autsl_runs',
            name='autsl_226_final',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.0005,  # Düşük LR ile başla (ağırlıklar zaten iyi)
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1,  # Kısa warmup
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.0,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            amp=True,
            verbose=True,
            close_mosaic=10,
            resume=False,  # ⭐ RESUME KAPALI - optimizer sıfırlanacak
        )
        
        print("\n" + "=" * 70)
        print("  ✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Eğitim kullanıcı tarafından durduruldu!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        raise

if __name__ == "__main__":
    main()
