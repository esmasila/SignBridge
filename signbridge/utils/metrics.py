"""
Metrik hesaplama fonksiyonları
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Doğruluk hesaplar
    
    Args:
        predictions: Tahminler
        targets: Gerçek etiketler
        
    Returns:
        accuracy: Doğruluk
    """
    return accuracy_score(targets, predictions)


def calculate_top_k_accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
    k: int = 5
) -> float:
    """
    Top-K doğruluk hesaplar
    
    Args:
        logits: Model çıktıları (batch_size, num_classes)
        targets: Gerçek etiketler (batch_size,)
        k: K değeri
        
    Returns:
        top_k_acc: Top-K doğruluk
    """
    # En yüksek k tahmini al
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    
    # Her örnek için doğru tahmin var mı?
    correct = np.any(top_k_preds == targets[:, np.newaxis], axis=1)
    
    return correct.mean()


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Kapsamlı metrikler hesaplar
    
    Args:
        predictions: Tahminler
        targets: Gerçek etiketler
        average: 'weighted', 'macro', 'micro'
        
    Returns:
        metrics: Metrik sözlüğü
    """
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average=average, zero_division=0),
        "recall": recall_score(targets, predictions, average=average, zero_division=0),
        "f1": f1_score(targets, predictions, average=average, zero_division=0)
    }
    
    return metrics


def calculate_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Confusion matrix hesaplar
    
    Args:
        predictions: Tahminler
        targets: Gerçek etiketler
        class_names: Sınıf isimleri (opsiyonel)
        
    Returns:
        cm: Confusion matrix
    """
    cm = confusion_matrix(targets, predictions)
    
    if class_names is not None:
        logger.info("\nConfusion Matrix:")
        logger.info(f"Classes: {class_names}")
    
    return cm


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str] = None
):
    """
    Sınıflandırma raporunu yazdırır
    
    Args:
        predictions: Tahminler
        targets: Gerçek etiketler
        class_names: Sınıf isimleri
    """
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        zero_division=0
    )
    
    logger.info("\nSınıflandırma Raporu:")
    logger.info(f"\n{report}")


class AverageMeter:
    """
    Ortalama hesaplayıcı (loss, accuracy vb. için)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Değer güncelle
        
        Args:
            val: Yeni değer
            n: Örnek sayısı
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """
    Eğitim sırasında metrikleri takip eder
    """
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
    
    def update(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float = None,
        val_acc: float = None
    ):
        """Metrikleri güncelle"""
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        if val_acc is not None:
            self.history["val_acc"].append(val_acc)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """En iyi metrikleri döndür"""
        best = {
            "best_train_acc": max(self.history["train_acc"]) if self.history["train_acc"] else 0,
            "best_val_acc": max(self.history["val_acc"]) if self.history["val_acc"] else 0,
            "min_train_loss": min(self.history["train_loss"]) if self.history["train_loss"] else float('inf'),
            "min_val_loss": min(self.history["val_loss"]) if self.history["val_loss"] else float('inf')
        }
        return best
    
    def get_history(self) -> Dict[str, List]:
        """Tüm geçmişi döndür"""
        return self.history


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Dummy data
    y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 2, 0])
    
    # Metrik hesapla
    metrics = calculate_metrics(y_pred, y_true)
    logger.info(f"Metrics: {metrics}")
    
    # Classification report
    class_names = ["Sınıf 0", "Sınıf 1", "Sınıf 2"]
    print_classification_report(y_pred, y_true, class_names)
