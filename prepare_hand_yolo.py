"""
AUTSL YOLO veri setini -> Tek sinif "HAND" dedektoru icin hazirla.
Resimlere dokunmadan, etiketleri class_id=0 yap.
"""
import os
from pathlib import Path

SRC = Path("C:/Projects/sign_bridge/autsl_yolo")
DST = Path("C:/Projects/sign_bridge/autsl_hand_yolo")

def convert_labels(split):
    src_lbl = SRC / split / "labels"
    dst_lbl = DST / split / "labels"
    dst_lbl.mkdir(parents=True, exist_ok=True)

    files = list(src_lbl.glob("*.txt"))
    total = len(files)
    print(f"[{split}] {total} etiket donusturuluyor...")

    for i, f in enumerate(files):
        with open(f, "r") as src:
            lines = src.readlines()
        new_lines = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) >= 5:
                # class_id -> 0 (hand), diger degerler aynen kalsin
                new_lines.append("0 " + " ".join(parts[1:]) + "\n")
        with open(dst_lbl / f.name, "w") as dst:
            dst.writelines(new_lines)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{total}")

    print(f"[{split}] TAMAM ({total} dosya)")

# Hepsini donustur
for split in ["train", "val", "test"]:
    convert_labels(split)

# Yeni data.yaml yaz (images orijinal konumundan referans)
yaml_content = f"""path: {SRC.as_posix()}
train: train/images
val: val/images
test: test/images
nc: 1
names:
- hand
"""
(DST / "data.yaml").write_text(yaml_content, encoding="utf-8")
print(f"\ndata.yaml yazildi: {DST / 'data.yaml'}")
print("\nSimdi eger resimleri paylasmak istiyorsak labels klasoruyle ayni yerdeolmali.")
print("ALTERNATIF: Labels dosyalarini mevcut images klasorune kopyalamak yerine")
print("YOLO images klasorunu orijinalden okuyacak sekilde data.yaml'da absolute path kullaniyoruz.")
