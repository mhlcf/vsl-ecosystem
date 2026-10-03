import torch
import torch.nn as nn
import torch.nn.functional as F


class VSLModel(nn.Module):
    def __init__(self, input_dim=201, hidden_dim=384, num_layers=3, num_classes=3315, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_out_dim = hidden_dim * 2
        self.bn = nn.BatchNorm1d(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout + 0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out.mean(dim=1)
        out = self.bn(out)
        logits = self.classifier(out)
        return logits

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
