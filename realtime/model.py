# model.py
import torch
import torch.nn as nn

class SignRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional
        self.log_softmax = nn.LogSoftmax(dim=-1)  # required for CTC

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        returns: (batch, seq_len, vocab_size)
        """
        out, _ = self.gru(x)
        out = self.fc(out)
        out = self.log_softmax(out)
        return out
