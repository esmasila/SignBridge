"""
LSTM Sequence Classifier for TID Sign Language
5 kelime: MERHABA, EVET, HAYIR, LÜTFEN, TEŞEKKÜR ETMEK
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM tabanlı sekans sınıflandırıcı
    
    Input: (batch, sequence_length, input_size) = (N, 60, 1629)
    Output: (batch, num_classes) = (N, 5)
    """
    
    def __init__(self, input_size=1629, hidden_size=256, num_layers=2, num_classes=5, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM katmanı (bidirectional daha iyi performans verir)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # Bidirectional olduğu için hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size) = (N, 60, 1629)
        Returns:
            logits: (batch, num_classes) = (N, 5)
        """
        # LSTM
        # lstm_out: (batch, seq_len, hidden_size * 2) = (N, 60, 512)
        # h_n: (num_layers * 2, batch, hidden_size) = (4, N, 256)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Son zaman adımını al (veya tüm sequence'in ortalamasını)
        # Option 1: Son hidden state (en yaygın)
        # Bidirectional için ileri ve geri yönlerin son hidden state'lerini birleştir
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # Son layer'ın ileri ve geri yönlerini al
        forward_hidden = h_n[-2, :, :]  # (batch, hidden_size)
        backward_hidden = h_n[-1, :, :] # (batch, hidden_size)
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_size * 2)
        
        # Option 2 (alternatif): Tüm sequence'in ortalaması
        # hidden = torch.mean(lstm_out, dim=1)  # (batch, hidden_size * 2)
        
        # Dropout ve FC layers
        out = self.dropout(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits


class GRUClassifier(nn.Module):
    """
    GRU tabanlı sekans sınıflandırıcı (LSTM'e alternatif, daha hızlı)
    """
    
    def __init__(self, input_size=1629, hidden_size=256, num_layers=2, num_classes=5, dropout=0.3):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        out = self.dropout(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits


def test_model():
    """Model boyutlarını test et"""
    batch_size = 8
    seq_len = 60
    input_size = 1629
    num_classes = 5
    
    # Dummy data
    x = torch.randn(batch_size, seq_len, input_size)
    
    # LSTM
    model_lstm = LSTMClassifier(input_size, hidden_size=256, num_layers=2, num_classes=num_classes)
    out_lstm = model_lstm(x)
    print(f"LSTM Input: {x.shape} -> Output: {out_lstm.shape}")
    print(f"LSTM Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")
    
    # GRU
    model_gru = GRUClassifier(input_size, hidden_size=256, num_layers=2, num_classes=num_classes)
    out_gru = model_gru(x)
    print(f"\nGRU Input: {x.shape} -> Output: {out_gru.shape}")
    print(f"GRU Parameters: {sum(p.numel() for p in model_gru.parameters()):,}")


if __name__ == "__main__":
    test_model()
