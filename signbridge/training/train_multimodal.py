"""
MultimodalDigitsCNN Eğitim Scripti

Kullanım:
    python -m signbridge.training.train_multimodal

Gereksinimler:
    data/raw/Sign-Language-Digits-Dataset-master/Dataset/{0..9}/*.jpg

Adımlar:
    1. Veri setini yükle ve landmark çıkar (MediaPipe Holistic)
    2. MultimodalDigitsCNN modelini eğit
    3. En iyi modeli checkpoints/multimodal_digits_best.pt olarak kaydet
    4. Eğitim geçmişini grafikle kaydet
"""

import sys
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signbridge.config import (
    CHECKPOINT_DIR,
    DATASET_PATHS,
    PROCESSED_DATA_DIR,
    MEDIAPIPE_CONFIG,
    DIGITS_MODEL_CONFIG,
    TRAINING_CONFIG,
    CLASS_TO_TURKISH,
)
from signbridge.models.multimodal_cnn import MultimodalDigitsCNN, save_multimodal_model
from signbridge.data.preprocess import LandmarkExtractor
from signbridge.utils.logging_utils import setup_logger

logger = setup_logger("train_multimodal")

# -----------------------------------------------------------------------
# Sabitler
# -----------------------------------------------------------------------
LANDMARK_DIM = 543 * 3  # 468 yüz + 33 pose + 21+21 el = 543 * 3 = 1629
IMG_SIZE = DIGITS_MODEL_CONFIG["input_size"]  # (64, 64)
NUM_CLASSES = DIGITS_MODEL_CONFIG["num_classes"]  # 10


# -----------------------------------------------------------------------
# Dataset: Görüntü + Landmark
# -----------------------------------------------------------------------
class MultimodalDigitsDataset(Dataset):
    """
    Her örnek için (görüntü tensörü, landmark vektörü, etiket) döndürür.
    İlk çağrıda landmark'lar MediaPipe ile çıkarılır ve önbelleğe alınır.
    """

    def __init__(
        self,
        data_dir: Path,
        transform=None,
        max_per_class: Optional[int] = None,
        cache_path: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.landmarks: list[Optional[np.ndarray]] = []
        self._cache_path = cache_path or (PROCESSED_DATA_DIR / "digits_landmarks.npz")

        # Görüntü listesini oluştur
        for cls_idx in range(NUM_CLASSES):
            cls_dir = self.data_dir / str(cls_idx)
            if not cls_dir.exists():
                logger.warning(f"Sinif klasoru bulunamadi: {cls_dir}")
                continue
            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            if max_per_class:
                imgs = imgs[:max_per_class]
            for p in imgs:
                self.samples.append((p, cls_idx))

        logger.info(f"Toplam ornek: {len(self.samples)}")

        # Landmark'ları yükle veya çıkar
        self._load_or_extract_landmarks()

    # ------------------------------------------------------------------
    def _load_or_extract_landmarks(self):
        """Önbellekten yükle ya da MediaPipe ile çıkar."""
        if self._cache_path.exists():
            logger.info(f"Onbellekten yukleniyor: {self._cache_path}")
            data = np.load(self._cache_path)
            cached_landmarks = data["landmarks"]  # (N, LANDMARK_DIM)
            cached_labels = data["labels"]  # (N,)

            # Önbellek geçerli mi? (aynı örnekler mi?)
            if len(cached_landmarks) == len(self.samples):
                self.landmarks = list(cached_landmarks)
                logger.info(f"Onbellekten {len(self.landmarks)} landmark yuklendi.")
                return
            else:
                logger.warning(
                    f"Onbellekteki ornek sayisi ({len(cached_landmarks)}) "
                    f"guncel veriyle eslesmedi ({len(self.samples)}). Yeniden cikariliyor."
                )

        logger.info("MediaPipe ile landmark cikariliyor (ilk calisma yavas olabilir)...")
        extractor = LandmarkExtractor()
        self.landmarks = []

        for img_path, _ in tqdm(self.samples, desc="Landmark cikarimi"):
            image = cv2.imread(str(img_path))
            if image is None:
                self.landmarks.append(np.zeros(LANDMARK_DIM, dtype=np.float32))
                continue
            lm = extractor.extract_from_image(image)
            if lm is None or len(lm) == 0:
                lm = np.zeros(LANDMARK_DIM, dtype=np.float32)
            # Boyut uyumu
            if len(lm) < LANDMARK_DIM:
                lm = np.pad(lm, (0, LANDMARK_DIM - len(lm)))
            elif len(lm) > LANDMARK_DIM:
                lm = lm[:LANDMARK_DIM]
            self.landmarks.append(lm.astype(np.float32))

        extractor.close()

        # Önbelleğe kaydet
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        labels_arr = np.array([lbl for _, lbl in self.samples])
        np.savez_compressed(
            self._cache_path,
            landmarks=np.array(self.landmarks),
            labels=labels_arr,
        )
        logger.info(f"Landmark'lar kaydedildi: {self._cache_path}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Görüntü
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Landmark
        lm = torch.from_numpy(self.landmarks[idx])

        return image, lm, label


# -----------------------------------------------------------------------
# Eğitim yardımcıları
# -----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, landmarks, labels in loader:
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, landmarks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, landmarks, labels in loader:
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)

        outputs = model(images, landmarks)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def plot_history(history: dict, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Egitim grafigi kaydedildi: {save_path}")


# -----------------------------------------------------------------------
# Ana eğitim fonksiyonu
# -----------------------------------------------------------------------
def train(
    data_dir: Optional[Path] = None,
    epochs: int = TRAINING_CONFIG["num_epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    lr: float = TRAINING_CONFIG["learning_rate"],
    patience: int = TRAINING_CONFIG["patience"],
    val_ratio: float = TRAINING_CONFIG["validation_split"],
    test_ratio: float = TRAINING_CONFIG["test_split"],
    max_per_class: Optional[int] = None,
    num_workers: int = 0,  # Windows'ta 0 güvenli
):
    # Veri dizini
    data_dir = data_dir or DATASET_PATHS["digits"]
    if not data_dir.exists():
        logger.error(
            f"Veri seti bulunamadi: {data_dir}\n"
            "Lutfen Sign-Language-Digits-Dataset-master klasorunu "
            "data/raw/ icine koyun."
        )
        return None

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Transform
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset (landmark çıkarma transform'dan bağımsız, ham görüntü üzerinde yapılır)
    full_dataset = MultimodalDigitsDataset(
        data_dir=data_dir,
        transform=None,  # Sonradan split bazlı atanacak
        max_per_class=max_per_class,
    )

    n = len(full_dataset)
    n_test = int(test_ratio * n)
    n_val = int(val_ratio * n)
    n_train = n - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Her split için transform ata (Subset'in dataset'ine erişim)
    train_set.dataset.transform = train_transform
    # val/test için transform'u geçici olarak val_transform yapmak gerekiyor.
    # Ancak Subset aynı dataset objesini paylaşır. Bunun için ayrı dataset oluşturuyoruz:
    val_dataset = MultimodalDigitsDataset(
        data_dir=data_dir, transform=val_transform, max_per_class=max_per_class
    )
    test_dataset = MultimodalDigitsDataset(
        data_dir=data_dir, transform=val_transform, max_per_class=max_per_class
    )

    val_set_final = torch.utils.data.Subset(val_dataset, val_set.indices)
    test_set_final = torch.utils.data.Subset(test_dataset, test_set.indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_set_final, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set_final, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Model
    model = MultimodalDigitsCNN(
        num_classes=NUM_CLASSES,
        use_landmarks=True,
        landmark_dim=LANDMARK_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parametreleri: {total_params:,}")

    # Optimizasyon
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Eğitim döngüsü
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve = 0
    best_path = CHECKPOINT_DIR / "multimodal_digits_best.pt"

    logger.info("=" * 60)
    logger.info(f"Egitim basliyor: {epochs} epoch, batch={batch_size}, lr={lr}")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch [{epoch}/{epochs}] {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
        )

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            save_multimodal_model(
                model,
                str(best_path),
                metadata={"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss},
            )
            logger.info(f"  -> En iyi model kaydedildi (val_acc={val_acc:.2%})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping: {patience} epoch iyilesme olmadi.")
                break

    # Test
    logger.info("=" * 60)
    logger.info("Test sonuclari hesaplaniyor...")
    # En iyi modeli yükle
    checkpoint = torch.load(str(best_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")
    logger.info(f"En iyi Val Acc: {best_val_acc:.2%}")
    logger.info(f"Model kaydedildi: {best_path}")
    logger.info("=" * 60)

    # Grafik
    plot_history(history, CHECKPOINT_DIR / "training_history.png")

    return model


# -----------------------------------------------------------------------
if __name__ == "__main__":
    train()
