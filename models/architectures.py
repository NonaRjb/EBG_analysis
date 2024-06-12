import torch
import torch.nn as nn
from torch.nn import LSTM, RNN
from collections import OrderedDict
import numpy as np


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


class LSTMClassifier(nn.Module):
    def __init__(self, bidirectional=False, **kwargs):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = kwargs['hidden_size']
        self.num_layers = kwargs['num_layers']

        self.lstm = LSTM(
            input_size=kwargs['input_size'],
            hidden_size=kwargs['hidden_size'],
            num_layers=kwargs['num_layers'],
            batch_first=True,
            dropout=kwargs['dropout'],
            bidirectional=bidirectional
        )
        # self.rnn = nn.RNN(input_size=kwargs['input_size'], hidden_size=kwargs['hidden_size'],
        #                   num_layers=kwargs['num_layers'], batch_first=True, dropout=kwargs['dropout'],
        #                   bidirectional=bidirectional)
        # self.fc1 = nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        # self.fc2 = nn.Linear(kwargs['hidden_size'], 64)
        self.fc3 = nn.Linear(kwargs['hidden_size'],
                             kwargs['n_classes'] if kwargs['n_classes'] > 2 else 1)

    def forward(self, x):
        h_0 = torch.randn(self.num_layers, x.shape[0], self.hidden_size).double().to(self.device)
        c_0 = torch.randn(self.num_layers, x.shape[0], self.hidden_size).double().to(self.device)

        x = x.squeeze(dim=1).permute((0, 2, 1))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # output, hn = self.rnn(x, h_0)
        # final_h = torch.cat((hn[-2, ...], hn[-1, ...]), dim=1)
        final_h = nn.functional.relu(hn[-1, ...])
        # out = nn.functional.relu(self.fc1(final_h))
        # out = nn.functional.relu(self.fc2(out))
        out = self.fc3(final_h)
        return out


class RNNClassifier(nn.Module):
    def __init__(self, bidirectional=False, **kwargs):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = kwargs['hidden_size']
        self.num_layers = kwargs['num_layers']

        self.lstm = RNN(
            input_size=kwargs['input_size'],
            hidden_size=kwargs['hidden_size'],
            num_layers=kwargs['num_layers'],
            batch_first=True,
            dropout=kwargs['dropout'],
            bidirectional=bidirectional
        )
        # self.rnn = nn.RNN(input_size=kwargs['input_size'], hidden_size=kwargs['hidden_size'],
        #                   num_layers=kwargs['num_layers'], batch_first=True, dropout=kwargs['dropout'],
        #                   bidirectional=bidirectional)
        # self.fc1 = nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        # self.fc2 = nn.Linear(kwargs['hidden_size'], 64)
        self.fc3 = nn.Linear(kwargs['hidden_size'],
                             kwargs['n_classes'] if kwargs['n_classes'] > 2 else 1)

    def forward(self, x):
        h_0 = torch.randn(self.num_layers, x.shape[0], self.hidden_size).double().to(self.device)

        x = x.squeeze().permute((0, 2, 1))
        output, hn = self.lstm(x, h_0)
        # output, hn = self.rnn(x, h_0)
        # final_h = torch.cat((hn[-2, ...], hn[-1, ...]), dim=1)
        final_h = nn.functional.relu(hn[-1, ...])
        # out = nn.functional.relu(self.fc1(final_h))
        # out = nn.functional.relu(self.fc2(out))
        out = self.fc3(final_h)
        return out


class TFRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        # self.conv2 = nn.Conv2d(8, 8, kernel_size=(5, 5), stride=1, padding=1)
        # self.act2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(8, 4, kernel_size=(5, 5), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(35052, 8154)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(8154, 512)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(512, 1)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        # x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        # x = self.pool2(x)

        x = self.act3(self.conv3(x))
        x = self.pool3(x)

        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)

        x = self.act4(self.fc4(x))
        x = self.drop4(x)
        # input 512, output 10
        x = self.fc5(x)
        return x


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_channels, n_samples, net_filter_size, net_seq_length, n_classes, kernel_size=17,
                 dropout_rate=0.5):
        super(ResNet1d, self).__init__()
        # my modifications!
        input_dim = (n_channels, n_samples)
        blocks_dim = list(zip(net_filter_size, net_seq_length))
        if n_classes == 2:
            n_classes = 1

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x
