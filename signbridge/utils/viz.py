"""
Görselleştirme fonksiyonları
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None
):
    """
    Eğitim geçmişini görselleştirir
    
    Args:
        history: Metrik geçmişi
        save_path: Kaydedilecek dosya yolu (opsiyonel)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss grafiği
    if "train_loss" in history and history["train_loss"]:
        axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(history["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Kayıp Değeri (Loss)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy grafiği
    if "train_acc" in history and history["train_acc"]:
        axes[1].plot([acc * 100 for acc in history["train_acc"]], label="Train Acc", marker='o')
    if "val_acc" in history and history["val_acc"]:
        axes[1].plot([acc * 100 for acc in history["val_acc"]], label="Val Acc", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Doğruluk (%)")
    axes[1].set_title("Doğruluk (Accuracy)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grafik kaydedildi: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Confusion matrix görselleştirir
    
    Args:
        cm: Confusion matrix
        class_names: Sınıf isimleri
        save_path: Kaydedilecek dosya yolu
        normalize: Normalize edilsin mi?
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='Gerçek Etiket',
           xlabel='Tahmin Edilen Etiket')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Hücrelere metin ekle
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix kaydedildi: {save_path}")
    
    plt.show()


def draw_landmarks_on_image(
    image: np.ndarray,
    results,
    mp_holistic,
    mp_drawing,
    mp_drawing_styles
) -> np.ndarray:
    """
    MediaPipe landmark'larını görüntü üzerine çizer
    
    Args:
        image: BGR görüntü
        results: MediaPipe Holistic sonuçları
        mp_holistic: MediaPipe Holistic
        mp_drawing: MediaPipe drawing utils
        mp_drawing_styles: MediaPipe drawing styles
        
    Returns:
        annotated_image: Landmark'lı görüntü
    """
    annotated_image = image.copy()
    
    # Yüz landmark'ları
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
    
    # Pose landmark'ları
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # El landmark'ları
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    return annotated_image


def put_turkish_text(
    image: np.ndarray,
    text: str,
    position: tuple,
    font_scale: float = 1.0,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    background: bool = True
) -> np.ndarray:
    """
    Görüntü üzerine Türkçe metin yazar
    
    Args:
        image: Görüntü
        text: Metin (Türkçe)
        position: (x, y) konumu
        font_scale: Font boyutu
        color: Renk (B, G, R)
        thickness: Kalınlık
        background: Arka plan eklensin mi?
        
    Returns:
        image: Metin eklenmiş görüntü
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Metin boyutunu hesapla
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Arka plan ekle
    if background:
        cv2.rectangle(
            image,
            (x, y - text_height - 10),
            (x + text_width + 10, y + baseline),
            (0, 0, 0),
            -1
        )
    
    # Metni yaz
    cv2.putText(
        image,
        text,
        (x + 5, y - 5),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return image


def display_prediction(
    image: np.ndarray,
    prediction: str,
    confidence: float,
    fps: Optional[float] = None
) -> np.ndarray:
    """
    Tahmin sonucunu görüntü üzerine yazar
    
    Args:
        image: Görüntü
        prediction: Tahmin (Türkçe metin)
        confidence: Güven skoru
        fps: FPS değeri (opsiyonel)
        
    Returns:
        image: Annotate edilmiş görüntü
    """
    h, w = image.shape[:2]
    
    # Tahmin metni
    pred_text = f"Tahmin: {prediction}"
    conf_text = f"Guven: {confidence:.2%}"
    
    image = put_turkish_text(image, pred_text, (10, 40), font_scale=1.2, color=(0, 255, 0))
    image = put_turkish_text(image, conf_text, (10, 80), font_scale=0.8, color=(255, 255, 0))
    
    # FPS
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        image = put_turkish_text(image, fps_text, (w - 150, 40), font_scale=0.8, color=(255, 255, 255))
    
    # Çıkış talimatı
    exit_text = "Cikmak icin 'q' tusuna basin"
    image = put_turkish_text(
        image, exit_text, (10, h - 20),
        font_scale=0.6, color=(200, 200, 200), background=False
    )
    
    return image


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Dummy training history
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
        "val_loss": [0.9, 0.7, 0.5, 0.4, 0.35],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_acc": [0.55, 0.65, 0.75, 0.8, 0.82]
    }
    
    plot_training_history(history)
