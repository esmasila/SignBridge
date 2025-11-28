"""
Sekans bazlı modeller - LSTM/GRU/Transformer
TİD işaretleri için zaman serisi sınıflandırması
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
import logging

from signbridge.config import SEQUENCE_MODEL_CONFIG

logger = logging.getLogger(__name__)


class LSTMSignLanguageModel(nn.Module):
    """
    LSTM tabanlı işaret dili sınıflandırıcı
    Landmark sekanslarından işaret tahmini yapar
    """
    
    def __init__(
        self,
        input_dim: int = SEQUENCE_MODEL_CONFIG["input_dim"],
        hidden_dim: int = SEQUENCE_MODEL_CONFIG["hidden_dim"],
        num_layers: int = SEQUENCE_MODEL_CONFIG["num_layers"],
        num_classes: int = SEQUENCE_MODEL_CONFIG["num_classes"],
        dropout: float = SEQUENCE_MODEL_CONFIG["dropout"],
        bidirectional: bool = SEQUENCE_MODEL_CONFIG["bidirectional"]
    ):
        super(LSTMSignLanguageModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM katmanı
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected katman
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        logger.info(f"LSTM Model oluşturuldu:")
        logger.info(f"  Input: {input_dim}, Hidden: {hidden_dim}, Layers: {num_layers}")
        logger.info(f"  Classes: {num_classes}, Bidirectional: {bidirectional}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, sequence_length, input_dim)
            
        Returns:
            logits: Shape (batch_size, num_classes)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Son zaman adımının çıktısını al
        if self.bidirectional:
            # Son forward ve backward hidden state'leri birleştir
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # FC katmanlar
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        return out


class GRUSignLanguageModel(nn.Module):
    """
    GRU tabanlı işaret dili sınıflandırıcı
    LSTM'e alternatif, daha hızlı
    """
    
    def __init__(
        self,
        input_dim: int = SEQUENCE_MODEL_CONFIG["input_dim"],
        hidden_dim: int = SEQUENCE_MODEL_CONFIG["hidden_dim"],
        num_layers: int = SEQUENCE_MODEL_CONFIG["num_layers"],
        num_classes: int = SEQUENCE_MODEL_CONFIG["num_classes"],
        dropout: float = SEQUENCE_MODEL_CONFIG["dropout"],
        bidirectional: bool = SEQUENCE_MODEL_CONFIG["bidirectional"]
    ):
        super(GRUSignLanguageModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # GRU katmanı
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected katman
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        logger.info(f"GRU Model oluşturuldu:")
        logger.info(f"  Input: {input_dim}, Hidden: {hidden_dim}, Layers: {num_layers}")
        logger.info(f"  Classes: {num_classes}, Bidirectional: {bidirectional}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, sequence_length, input_dim)
            
        Returns:
            logits: Shape (batch_size, num_classes)
        """
        # GRU forward
        gru_out, hidden = self.gru(x)
        
        # Son hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # FC katmanlar
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        return out


class TransformerSignLanguageModel(nn.Module):
    """
    Transformer tabanlı işaret dili sınıflandırıcı
    Uzun menzilli bağımlılıkları yakalamak için
    """
    
    def __init__(
        self,
        input_dim: int = SEQUENCE_MODEL_CONFIG["input_dim"],
        hidden_dim: int = SEQUENCE_MODEL_CONFIG["hidden_dim"],
        num_layers: int = SEQUENCE_MODEL_CONFIG["num_layers"],
        num_classes: int = SEQUENCE_MODEL_CONFIG["num_classes"],
        nhead: int = 8,
        dropout: float = SEQUENCE_MODEL_CONFIG["dropout"]
    ):
        super(TransformerSignLanguageModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Transformer Model oluşturuldu:")
        logger.info(f"  Input: {input_dim}, Hidden: {hidden_dim}, Layers: {num_layers}")
        logger.info(f"  Classes: {num_classes}, Heads: {nhead}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, sequence_length, input_dim)
            
        Returns:
            logits: Shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Transformer için pozisyonel kodlama
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pozisyonel encoding hesapla
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def create_sequence_model(
    model_type: Literal["lstm", "gru", "transformer"] = "lstm",
    **kwargs
) -> nn.Module:
    """
    Sekans modeli fabrika fonksiyonu
    
    Args:
        model_type: "lstm", "gru", veya "transformer"
        **kwargs: Model parametreleri
        
    Returns:
        model: Oluşturulan model
    """
    if model_type == "lstm":
        return LSTMSignLanguageModel(**kwargs)
    elif model_type == "gru":
        return GRUSignLanguageModel(**kwargs)
    elif model_type == "transformer":
        return TransformerSignLanguageModel(**kwargs)
    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")


def load_sequence_model(
    checkpoint_path: str,
    model_type: str = "lstm",
    device: Optional[str] = None
) -> nn.Module:
    """
    Eğitilmiş sekans modelini yükler
    
    Args:
        checkpoint_path: Model checkpoint dosyası
        model_type: "lstm", "gru", veya "transformer"
        device: 'cuda' veya 'cpu'
        
    Returns:
        model: Yüklenmiş model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = create_sequence_model(model_type)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model yüklendi: {checkpoint_path}")
    logger.info(f"Tip: {model_type}, Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test kodu
    logging.basicConfig(level=logging.INFO)
    
    batch_size = 4
    seq_len = 30
    input_dim = SEQUENCE_MODEL_CONFIG["input_dim"]
    
    # Dummy input
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # LSTM test
    logger.info("\n=== LSTM Test ===")
    lstm_model = create_sequence_model("lstm")
    lstm_output = lstm_model(dummy_input)
    logger.info(f"Input: {dummy_input.shape}, Output: {lstm_output.shape}")
    
    # GRU test
    logger.info("\n=== GRU Test ===")
    gru_model = create_sequence_model("gru")
    gru_output = gru_model(dummy_input)
    logger.info(f"Input: {dummy_input.shape}, Output: {gru_output.shape}")
    
    # Transformer test
    logger.info("\n=== Transformer Test ===")
    transformer_model = create_sequence_model("transformer")
    transformer_output = transformer_model(dummy_input)
    logger.info(f"Input: {dummy_input.shape}, Output: {transformer_output.shape}")
