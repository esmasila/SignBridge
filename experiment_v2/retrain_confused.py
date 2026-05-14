"""
9 karışan kelime için yeniden augment + retrain.
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

CONFUSED = ["KENDI"]

sys.path.insert(0, str(BASE))
from pipeline import aug_noise, aug_scale, aug_translate, aug_rotate, aug_timewarp, aug_mirror
AUGS = [aug_noise, aug_scale, aug_translate, aug_rotate, aug_timewarp, aug_mirror]
NAMES = ["noise","scale","translate","rotate","timewarp","mirror"]

print("="*60); print("ADIM 1: 9 KELIMENIN AUG'LARINI YENILE"); print("="*60)
for word in CONFUSED:
    src = DATA_DIR / word
    dst = AUG_DIR / word
    if not src.exists():
        print(f"  [SKIP] {word}: data yok"); continue
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    files = sorted(src.glob("*.npy"))
    n = 0
    for f in files:
        shutil.copy2(f, dst / f.name)
        seq = np.load(f)
        for fn, name in zip(AUGS, NAMES):
            np.save(dst / f"{f.stem}_aug_{name}.npy", fn(seq))
            n += 1
    print(f"  {word:12s}: {len(files)} orig + {n} aug")

print("\n"+"="*60); print("ADIM 2: PRECOMPUTE V2"); print("="*60)
subprocess.run([sys.executable, str(BASE/"precompute_v2.py")], check=True)

print("\n"+"="*60); print("ADIM 3: FINETUNE"); print("="*60)
subprocess.run([sys.executable, str(BASE/"finetune.py")], check=True)

print("\n"+"="*60); print("ADIM 4: MODEL SWAP"); print("="*60)
shutil.copy2(CKPT_DIR/"best_model_v2.pt", CKPT_DIR/"best_model.pt")
print("  best_model_v2.pt -> best_model.pt")

print("\n"+"="*60); print("ADIM 5: ONNX EXPORT"); print("="*60)
subprocess.run([sys.executable, str(BASE/"export_onnx_v2.py")], check=False)

print("\nHEPSI TAMAM!")
