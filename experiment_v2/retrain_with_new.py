"""
YENI 22 KELIME ICIN PIPELINE
1. Yeni kelimeleri augment et (eskileri olan augment'leri silmez)
2. precompute_v2.py calistir (122 sinif icin 1755-dim)
3. finetune.py calistir (122 sinif icin GRU egitim)
4. best_model_v2.pt -> best_model.pt kopyala
5. ONNX export

Kullanim: python experiment_v2/retrain_with_new.py
"""
import sys, io, subprocess, shutil
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
AUG_DIR  = BASE / "data_aug"
CKPT_DIR = BASE / "checkpoints"

NEW_WORDS = [
    "NASILSIN", "NE", "KAC", "NEREDE",
    "YAPMAK", "CALISMAK", "YORULMAK", "BITMEK",
    "ICMEK", "BULUSMAK", "GORUSMEK", "BAKMAK",
    "COK", "SAAT", "SONRA", "BOS", "BERABER", "TAMAM",
    "PARK", "YAN", "KAFE", "KENDI",
]

# pipeline.py'den augmentation fonksiyonlarini al
sys.path.insert(0, str(BASE))
from pipeline import (
    aug_noise, aug_scale, aug_translate,
    aug_rotate, aug_timewarp, aug_mirror,
)
AUGS = [aug_noise, aug_scale, aug_translate, aug_rotate, aug_timewarp, aug_mirror]
AUG_NAMES = ["noise", "scale", "translate", "rotate", "timewarp", "mirror"]


# ─── ADIM 1: YENI KELIMELERI AUGMENT ET ───
print("=" * 60)
print("ADIM 1: YENI KELIMELERIN AUGMENTATION'I")
print("=" * 60)
total_orig = 0
total_aug = 0
for word in NEW_WORDS:
    src = DATA_DIR / word
    dst = AUG_DIR / word
    if not src.exists():
        print(f"  [!] {word} klasoru yok, atlandi")
        continue
    files = sorted(src.glob("*.npy"))
    if not files:
        print(f"  [!] {word} bos, atlandi")
        continue
    if dst.exists() and len(list(dst.glob("*.npy"))) >= len(files) * 6:
        print(f"  [SKIP] {word} zaten augment edilmis")
        continue
    dst.mkdir(parents=True, exist_ok=True)

    new_cnt = 0
    for f in files:
        # Orijinali kopyala
        shutil.copy2(f, dst / f.name)
        seq = np.load(f)
        # 6 farkli augmentation uretir
        for fn, name in zip(AUGS, AUG_NAMES):
            aug_seq = fn(seq)
            np.save(dst / f"{f.stem}_aug_{name}.npy", aug_seq)
            new_cnt += 1
    total_orig += len(files)
    total_aug += new_cnt
    print(f"  {word:12s}: {len(files)} orig + {new_cnt} aug = {len(files)+new_cnt}")

print(f"\nTOPLAM yeni: {total_orig} orijinal -> {total_orig + total_aug} augment edildi")


# ─── ADIM 2: PRECOMPUTE V2 (1755-dim) ───
print("\n" + "=" * 60)
print("ADIM 2: PRECOMPUTE V2 (122 sinif -> 1755-dim)")
print("=" * 60)
result = subprocess.run([sys.executable, str(BASE / "precompute_v2.py")],
                        capture_output=False)
if result.returncode != 0:
    print("[HATA] precompute_v2 basarisiz!")
    sys.exit(1)


# ─── ADIM 3: FINETUNE (GRU egitim) ───
print("\n" + "=" * 60)
print("ADIM 3: FINETUNE (GRU egitim 122 sinif)")
print("=" * 60)
result = subprocess.run([sys.executable, str(BASE / "finetune.py")],
                        capture_output=False)
if result.returncode != 0:
    print("[HATA] finetune basarisiz!")
    sys.exit(1)


# ─── ADIM 4: best_model_v2.pt -> best_model.pt ───
print("\n" + "=" * 60)
print("ADIM 4: MODEL KOPYALAMA (web_app icin)")
print("=" * 60)
src = CKPT_DIR / "best_model_v2.pt"
dst = CKPT_DIR / "best_model.pt"
if src.exists():
    # Backup eski model
    backup = CKPT_DIR / "best_model_pre_122.pt"
    if dst.exists() and not backup.exists():
        shutil.copy2(dst, backup)
        print(f"  Backup: {backup.name}")
    shutil.copy2(src, dst)
    print(f"  {src.name} -> {dst.name}")
else:
    print(f"  [!] {src} yok!")


# ─── ADIM 5: ONNX EXPORT (mobil icin) ───
print("\n" + "=" * 60)
print("ADIM 5: ONNX EXPORT")
print("=" * 60)
result = subprocess.run([sys.executable, str(BASE / "export_onnx_v2.py")],
                        capture_output=False)
if result.returncode != 0:
    print("[!] ONNX export basarisiz, devam edilecek (mobil etkilenir)")


print("\n" + "=" * 60)
print("HEPSI TAMAM!")
print("=" * 60)
print("Sonraki adim: web_app.py'i yeniden baslat (server restart)")
