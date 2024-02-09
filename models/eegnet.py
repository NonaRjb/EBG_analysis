import torch
import torch.nn as nn
from collections import OrderedDict


class EEGNet(nn.Module):
    def __init__(self, n_samples, n_classes, n_channels=96, f1=8, d=2, f2=16, kernel_length=64, dropout_rate=0.5):
        super().__init__()

        self.model = torch.nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, kernel_length), padding='same')),
            ('bn1', nn.BatchNorm2d(f1)),
            ('conv2', nn.Conv2d(in_channels=f1, out_channels=d * f1, kernel_size=(n_channels, 1),
                                groups=f1, padding='valid')),
            ('bn2', nn.BatchNorm2d(d * f1)),
            # ('relu1', nn.ReLU()),
            ('elu1', nn.ELU()),
            ('pool1', nn.AvgPool2d(kernel_size=(1, 4))),
            ('do1', nn.Dropout(p=dropout_rate)),
            ('conv3', nn.Conv2d(in_channels=d * f1, out_channels=f2, kernel_size=(1, 16), groups=f2,
                                padding='same')),
            ('conv4', nn.Conv2d(in_channels=f2, out_channels=f2, kernel_size=1, padding='same')),
            ('bn3', nn.BatchNorm2d(f2)),
            # ('relu2', nn.ReLU()),
            ('elu2', nn.ELU()),
            ('pool2', nn.AvgPool2d(kernel_size=(1, 8))),
            ('do2', nn.Dropout(p=dropout_rate)),
            ('flat', nn.Flatten()),
            ('lnout', nn.Linear(f2 * (n_samples // 32), n_classes if n_classes > 2 else 1))
        ]))

    def forward(self, x):
        x = x.unsqueeze(dim=1).double()
        x = self.model(x)
        return x

