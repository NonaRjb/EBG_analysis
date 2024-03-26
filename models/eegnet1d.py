import torch
import torch.nn as nn
from collections import OrderedDict

class EEGNet1D(nn.Module):
    def __init__(self, n_samples, n_classes, n_channels=96, f1=8, d=2, f2=16, kernel_length=64, dropout_rate=0.5):
        super().__init__()

        self.model = torch.nn.Sequential(OrderedDict([
            # Input shape: (batch_size, n_channels, n_time_samples)
            ('conv1', nn.Conv1d(in_channels=n_channels, out_channels=f1, kernel_size=kernel_length, padding='same')),
            ('bn1', nn.BatchNorm1d(f1)),
            ('conv2', nn.Conv1d(in_channels=f1, out_channels=d * f1, kernel_size=1, groups=f1, padding='valid')),
            ('bn2', nn.BatchNorm1d(d * f1)),
            ('elu1', nn.ELU()),
            ('pool1', nn.AvgPool1d(kernel_size=4)),
            ('do1', nn.Dropout(p=dropout_rate)),
            ('conv3', nn.Conv1d(in_channels=d * f1, out_channels=f2, kernel_size=16, groups=d * f1, padding='same')),
            ('conv4', nn.Conv1d(in_channels=f2, out_channels=f2, kernel_size=1, padding='same')),
            ('bn3', nn.BatchNorm1d(f2)),
            ('elu2', nn.ELU()),
            ('pool2', nn.AvgPool1d(kernel_size=8)),
            ('do2', nn.Dropout(p=dropout_rate)),
            ('flat', nn.Flatten()),
            # Adjusted output linear layer size according to your pooling and convolution operations
            ('lnout', nn.Linear(f2 * ((n_samples // 4) // 8), n_classes if n_classes > 2 else 1))
        ]))

    def forward(self, x):
        # No need to unsqueeze since input shape is already in the format (batch_size, n_channels, n_time_samples)
        x = x.double()  # Ensure input is double if required by the model's design
        x = self.model(x)
        return x
