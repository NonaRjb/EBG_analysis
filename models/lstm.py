import torch
import torch.nn as nn
from torch.nn import LSTM


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
