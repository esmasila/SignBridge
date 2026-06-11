import torch
import os

print("=" * 70)
print("KAPSAMLI CHECKPOINT VE EĞİTİM ANALİZİ")
print("=" * 70)

# 1. Checkpoint dosyaları
print("\n1. CHECKPOINT DOSYALARI:")
weights_dir = "autsl_runs/autsl_226_final/weights"
files = ['epoch0.pt', 'epoch5.pt', 'epoch10.pt', 'epoch15.pt', 'best.pt', 'last.pt']

for f in files:
    path = os.path.join(weights_dir, f)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', -1) + 1
        bf = ckpt.get('best_fitness', 0)
        model = ckpt.get('model')
        ema = ckpt.get('ema')
        opt = ckpt.get('optimizer')
        
        model_type = type(model).__name__
        ema_type = type(ema).__name__
        opt_status = "VAR" if opt else "YOK"
        
        # EMA ağırlık kontrolü
        ema_ok = "?"
        if ema is not None:
            sd = ema.state_dict()
            has_nan = any(torch.isnan(v).any().item() if torch.is_tensor(v) else False for v in sd.values())
            has_inf = any(torch.isinf(v).any().item() if torch.is_tensor(v) else False for v in sd.values())
            ema_ok = "BOZUK" if (has_nan or has_inf) else "SAĞLAM"
        
        print(f"  {f}: epoch={epoch}, mAP50-95={bf:.4f}, model={model_type}, ema={ema_type}({ema_ok}), opt={opt_status}")

# 2. Results.csv analizi
print("\n2. EĞİTİM SONUÇLARI (results.csv):")
results_path = "autsl_runs/autsl_226_final/results.csv"
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
    
    print("  Epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 | val/cls_loss")
    print("  " + "-" * 75)
    
    problem_epochs = []
    for line in lines[1:]:  # header'ı atla
        parts = line.strip().split(',')
        if len(parts) >= 11:
            epoch = int(parts[0])
            box_loss = float(parts[2])
            cls_loss = float(parts[3])
            dfl_loss = float(parts[4])
            precision = float(parts[5])
            recall = float(parts[6])
            map50 = float(parts[7])
            map50_95 = float(parts[8])
            val_cls_loss = parts[10]
            
            # Sorun var mı?
            problem = ""
            if map50 == 0 or map50_95 == 0:
                problem = " ⚠️ mAP=0!"
                problem_epochs.append(epoch)
            if val_cls_loss == 'inf':
                problem = " ❌ val_cls=inf!"
                if epoch not in problem_epochs:
                    problem_epochs.append(epoch)
            
            print(f"  {epoch:5} | {box_loss:.4f}   | {cls_loss:.4f}   | {dfl_loss:.4f}   | {map50:.4f} | {map50_95:.4f}   | {val_cls_loss}{problem}")
    
    if problem_epochs:
        print(f"\n  ⚠️ SORUNLU EPOCH'LAR: {problem_epochs}")
    else:
        print("\n  ✅ Tüm epoch'lar normal!")

# 3. Son model testi
print("\n3. BEST.PT İLE HIZLI TEST:")
try:
    from ultralytics import YOLO
    model = YOLO("autsl_runs/autsl_226_final/weights/best.pt")
    print(f"  Model yüklendi: {model.model.__class__.__name__}")
    print(f"  Sınıf sayısı: {model.model.nc if hasattr(model.model, 'nc') else '?'}")
except Exception as e:
    print(f"  ❌ Model yükleme hatası: {e}")

print("\n" + "=" * 70)
