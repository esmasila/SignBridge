"""
Landmark Sequence Augmentation
.npy dosyalarini direkt augment eder - MediaPipe yeniden calistirma gerekmez.

Desteklenen augmentasyonlar:
  1. Gaussian noise  - konum belirsizligini simule eder
  2. Scaling         - zoom in/out simule eder
  3. Translation     - hafif pozisyon kaydirma
  4. Rotation (2D)   - +/- 10 derece donme (landmark koordinatlari uzerinde)
  5. Time warp       - temporal hizi degistir (yavaslat/hizlandir)
  6. Mirror (flip)   - sol/sag el landmarklarini dogru sekilde swap eder

Kullanim:
  python tid_sequence/scripts/augment_sequences.py
  python tid_sequence/scripts/augment_sequences.py --input tid_sequence/data --output tid_sequence/data_aug --multiplier 5
"""

import numpy as np
import argparse
import shutil
from pathlib import Path
from datetime import datetime


# --- MediaPipe Holistic landmark index araliklari ---
# Toplam 1629 deger = 543 landmark x 3 (x, y, z)
FACE_START   = 0
FACE_END     = 468 * 3          # 1404
POSE_START   = FACE_END
POSE_END     = POSE_START + 33 * 3  # 1503
LEFT_START   = POSE_END
LEFT_END     = LEFT_START + 21 * 3  # 1566
RIGHT_START  = LEFT_END
RIGHT_END    = RIGHT_START + 21 * 3  # 1629

TOTAL_FEATURES = 1629


# --------------------------------------------------------------------------- #
#  Temel augmentasyon fonksiyonlari                                            #
# --------------------------------------------------------------------------- #

def aug_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Landmark koordinatlarina kucuk Gaussian gurultu ekle."""
    noise = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    return np.clip(seq + noise, 0.0, 1.0)


def aug_scale(seq: np.ndarray, scale_range=(0.90, 1.10)) -> np.ndarray:
    """Tum koordinatlari merkez etrafinda olcekle (zoom simulasyonu)."""
    factor = np.random.uniform(*scale_range)
    out = seq.copy()
    # x ve y koordinatlarini (her 3. deger) olcekle, merkez 0.5
    for start in [FACE_START, POSE_START, LEFT_START, RIGHT_START]:
        end = start + (468*3 if start == FACE_START else
                       33*3  if start == POSE_START  else 21*3)
        chunk = out[:, start:end].reshape(len(seq), -1, 3)
        chunk[:, :, :2] = (chunk[:, :, :2] - 0.5) * factor + 0.5
        out[:, start:end] = chunk.reshape(len(seq), -1)
    return np.clip(out, 0.0, 1.5)


def aug_translate(seq: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
    """Tum landmarklari x ve y ekseninde rastgele kaydır."""
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    out = seq.copy()
    for start in [FACE_START, POSE_START, LEFT_START, RIGHT_START]:
        end = start + (468*3 if start == FACE_START else
                       33*3  if start == POSE_START  else 21*3)
        chunk = out[:, start:end].reshape(len(seq), -1, 3)
        chunk[:, :, 0] += dx
        chunk[:, :, 1] += dy
        out[:, start:end] = chunk.reshape(len(seq), -1)
    return np.clip(out, 0.0, 1.5)


def aug_rotate(seq: np.ndarray, max_angle_deg: float = 10.0) -> np.ndarray:
    """Tum 2D landmark koordinatlarini merkez etrafinda dondurun."""
    angle = np.random.uniform(-max_angle_deg, max_angle_deg)
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    out = seq.copy()
    for start in [FACE_START, POSE_START, LEFT_START, RIGHT_START]:
        end = start + (468*3 if start == FACE_START else
                       33*3  if start == POSE_START  else 21*3)
        chunk = out[:, start:end].reshape(len(seq), -1, 3)
        x = chunk[:, :, 0] - 0.5
        y = chunk[:, :, 1] - 0.5
        chunk[:, :, 0] = cos_a * x - sin_a * y + 0.5
        chunk[:, :, 1] = sin_a * x + cos_a * y + 0.5
        out[:, start:end] = chunk.reshape(len(seq), -1)
    return np.clip(out, 0.0, 1.5)


def aug_time_warp(seq: np.ndarray, warp_range=(0.85, 1.15)) -> np.ndarray:
    """
    Temporal hizi degistir: sekansı yavaslatir/hizlandirir,
    sonra orijinal uzunluga (60 frame) interpole eder.
    """
    T, F = seq.shape
    factor = np.random.uniform(*warp_range)
    new_T = max(int(T * factor), T // 2)

    # Orijinal frame indeksleri
    old_idx = np.linspace(0, T - 1, new_T)
    new_idx = np.linspace(0, new_T - 1, T)

    warped = np.zeros((new_T, F), dtype=np.float32)
    for f in range(F):
        warped[:, f] = np.interp(old_idx, np.arange(T), seq[:, f])

    # T uzunluguna geri interpole et
    out = np.zeros((T, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(new_idx, np.arange(new_T), warped[:, f])
    return out


def aug_mirror(seq: np.ndarray) -> np.ndarray:
    """
    Yatay aynalama: x = 1 - x yapar VE
    sol el <-> sag el landmarklarini swap eder.
    Pose landmarklarinda da sol/sag swap yapilir.
    """
    out = seq.copy()

    # Yuz: sadece x'i cevir
    face = out[:, FACE_START:FACE_END].reshape(len(seq), 468, 3)
    face[:, :, 0] = 1.0 - face[:, :, 0]
    out[:, FACE_START:FACE_END] = face.reshape(len(seq), -1)

    # Pose: x cevir (sol/sag swap yaklasik - tam deil ama yeterli)
    pose = out[:, POSE_START:POSE_END].reshape(len(seq), 33, 3)
    pose[:, :, 0] = 1.0 - pose[:, :, 0]
    out[:, POSE_START:POSE_END] = pose.reshape(len(seq), -1)

    # Sol el ve sag eli swap et + x cevir
    left  = out[:, LEFT_START:LEFT_END].copy()
    right = out[:, RIGHT_START:RIGHT_END].copy()

    left_r  = left.reshape(len(seq), 21, 3)
    right_r = right.reshape(len(seq), 21, 3)
    left_r[:, :, 0]  = 1.0 - left_r[:, :, 0]
    right_r[:, :, 0] = 1.0 - right_r[:, :, 0]

    # Swap
    out[:, LEFT_START:LEFT_END]   = right_r.reshape(len(seq), -1)
    out[:, RIGHT_START:RIGHT_END] = left_r.reshape(len(seq), -1)

    return out


# --------------------------------------------------------------------------- #
#  Augmentasyon pipeline secici                                                #
# --------------------------------------------------------------------------- #

AUGMENTATIONS = {
    "noise":     lambda s: aug_noise(s, sigma=0.004),
    "scale":     lambda s: aug_scale(s, scale_range=(0.92, 1.08)),
    "translate": lambda s: aug_translate(s, max_shift=0.04),
    "rotate":    lambda s: aug_rotate(s, max_angle_deg=8.0),
    "timewarp":  lambda s: aug_time_warp(s, warp_range=(0.88, 1.12)),
    "mirror":    lambda s: aug_mirror(s),
}


def augment_file(npy_path: Path, output_dir: Path, augmentations: list):
    """Tek bir .npy dosyasini augment et ve kaydet."""
    seq = np.load(npy_path)  # (60, 1629)
    assert seq.shape == (60, TOTAL_FEATURES), \
        f"Beklenmeyen shape: {seq.shape}, dosya: {npy_path}"

    saved = []
    for aug_name in augmentations:
        fn = AUGMENTATIONS[aug_name]
        aug_seq = fn(seq).astype(np.float32)

        stem = npy_path.stem
        out_name = f"{stem}_aug_{aug_name}.npy"
        out_path = output_dir / out_name
        np.save(out_path, aug_seq)
        saved.append(out_name)

    return saved


# --------------------------------------------------------------------------- #
#  Ana fonksiyon                                                               #
# --------------------------------------------------------------------------- #

def run(input_dir: Path, output_dir: Path, augmentations: list,
        copy_originals: bool = True, multiplier: int = None):
    """
    input_dir altindaki her sinif klasorundeki .npy dosyalarini augment et.
    output_dir ayni sinif yapısini korur.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"Sinif klasoru bulunamadi: {input_dir}")
        return

    print(f"\nGirdi : {input_dir}")
    print(f"Cikti : {output_dir}")
    print(f"Siniflar: {[d.name for d in class_dirs]}")
    print(f"Augmentasyonlar: {augmentations}\n")

    total_orig = 0
    total_new  = 0

    for cls_dir in class_dirs:
        npy_files = sorted(cls_dir.glob("*.npy"))
        if not npy_files:
            continue

        out_cls_dir = output_dir / cls_dir.name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        # Orijinalleri kopyala
        if copy_originals:
            for f in npy_files:
                shutil.copy2(f, out_cls_dir / f.name)

        # multiplier varsa augmentasyon listesini kirp/uzat
        aug_list = augmentations
        if multiplier is not None:
            # Her dosya basi kac augmentasyon?
            aug_list = augmentations[:multiplier]

        cls_new = 0
        for npy_path in npy_files:
            saved = augment_file(npy_path, out_cls_dir, aug_list)
            cls_new += len(saved)

        total_orig += len(npy_files)
        total_new  += cls_new
        print(f"  {cls_dir.name:15s}: {len(npy_files)} orijinal + {cls_new} yeni = {len(npy_files)+cls_new} toplam")

    print(f"\nSONUC:")
    print(f"  Orijinal ornekler : {total_orig}")
    print(f"  Yeni ornekler     : {total_new}")
    print(f"  TOPLAM            : {total_orig + total_new}")
    print(f"  Cikti klasoru     : {output_dir}\n")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark sequence augmentation")
    parser.add_argument("--input",  default="tid_sequence/data",
                        help="Orijinal veri klasoru (sinif alt-klasorlerini icerir)")
    parser.add_argument("--output", default="tid_sequence/data_augmented",
                        help="Augmented cikti klasoru")
    parser.add_argument("--augs",   nargs="+",
                        default=["noise", "scale", "translate", "rotate", "timewarp", "mirror"],
                        choices=list(AUGMENTATIONS.keys()),
                        help="Uygulanacak augmentasyonlar")
    parser.add_argument("--no-copy-originals", action="store_true",
                        help="Orijinal dosyalari kopyalama")
    parser.add_argument("--multiplier", type=int, default=None,
                        help="Her dosya basi max augmentasyon sayisi")
    args = parser.parse_args()

    run(
        input_dir       = Path(args.input),
        output_dir      = Path(args.output),
        augmentations   = args.augs,
        copy_originals  = not args.no_copy_originals,
        multiplier      = args.multiplier,
    )
