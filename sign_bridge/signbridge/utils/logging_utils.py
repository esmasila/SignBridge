"""
Logging yardımcı fonksiyonları
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from signbridge.config import LOG_DIR


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Logger yapılandırır
    
    Args:
        name: Logger ismi
        log_file: Log dosyası ismi (None ise otomatik oluşturulur)
        level: Log seviyesi
        console: Console'a da yazsın mı?
        
    Returns:
        logger: Yapılandırılmış logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Mevcut handler'ları temizle
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    
    log_path = LOG_DIR / log_file
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger başlatıldı: {name}")
    logger.info(f"Log dosyası: {log_path}")
    
    return logger


def log_model_summary(logger: logging.Logger, model, input_shape: tuple):
    """
    Model özetini loglar
    
    Args:
        logger: Logger
        model: PyTorch modeli
        input_shape: Örnek input shape
    """
    import torch
    
    logger.info("=" * 60)
    logger.info("MODEL ÖZETİ")
    logger.info("=" * 60)
    
    # Model yapısı
    logger.info(f"\n{model}\n")
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Toplam Parametre: {total_params:,}")
    logger.info(f"Eğitilebilir Parametre: {trainable_params:,}")
    
    # Model boyutu (MB)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    logger.info(f"Model Boyutu: {size_all_mb:.2f} MB")
    
    logger.info("=" * 60)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None
):
    """
    Eğitim ilerlemesini loglar
    
    Args:
        logger: Logger
        epoch: Mevcut epoch
        total_epochs: Toplam epoch sayısı
        train_loss: Eğitim kaybı
        train_acc: Eğitim doğruluğu
        val_loss: Validation kaybı (opsiyonel)
        val_acc: Validation doğruluğu (opsiyonel)
    """
    msg = f"Epoch [{epoch}/{total_epochs}] - "
    msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}"
    
    if val_loss is not None and val_acc is not None:
        msg += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
    
    logger.info(msg)


if __name__ == "__main__":
    # Test
    logger = setup_logger("test_logger")
    logger.info("Bu bir test mesajıdır")
    logger.warning("Bu bir uyarıdır")
    logger.error("Bu bir hatadır")
