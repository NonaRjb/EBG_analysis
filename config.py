import numpy as np
import copy


class DNNConfig:
    def __init__(self):
        self.data_constants = {
            'tmin': -0.1,
            'tmax': 0.25,
            'w': 0.5,
            'fmin': None,
            'fmax': None,
            'fs_new': 256,
            'binary': True,
            'train_size': 0.8,
            'val_size': 0.2,
            'ebg_transform': None,
            'normalize': True,
            'shuffle_labels': False,
            'modality': 'source-ebg',
            'intensity': False,
            'n_classes': 2,
            'tfr_freqs': np.linspace(20, 100, 160),
            'baseline_type': 'zscore'
        }

        self.training_constants = {
            'scheduler_name': 'plateau',
            'patience': 20,
            'weight_decay': 0.1,
            'lr': 0.0001,
            'batch_size': 16,
            'optim_name': 'adamw',
            'warmup_steps': 100
        }

        if self.data_constants['fmax'] is not None and self.data_constants['fmin'] is not None:
            lstm_input_size = np.abs(self.data_constants['tfr_freqs'] - self.data_constants['fmax']).argmin() - \
                              np.abs(self.data_constants['tfr_freqs'] - self.data_constants['fmin']).argmin() + 1
        elif self.data_constants['fmax'] is not None:
            self.lstm_input_size = np.abs(self.data_constants['tfr_freqs'] - self.data_constants['fmax']).argmin()
        elif self.data_constants['fmin'] is not None:
            self.lstm_input_size = len(self.data_constants['tfr_freqs']) - \
                              np.abs(self.data_constants['tfr_freqs'] - self.data_constants['fmin']).argmin()
        else:
            self.lstm_input_size = len(self.data_constants['tfr_freqs'])

        self.model_constants = {
            'eegnet': {'n_channels': 4, 'n_classes': self.data_constants['n_classes']},
            'eegnet1d': {'n_channels': 4, 'n_classes': self.data_constants['n_classes']},
            'eegnet_attention': {'n_channels': 68, 'n_classes': self.data_constants['n_classes']},
            'lstm': {'n_channels': 4, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.0,
                     'n_classes': self.data_constants['n_classes']},
            'mlp':{'n_samples': 256, 'hidden_sizes': [32, 32], 'n_classes':self.data_constants['n_classes']},
            'rnn': {'input_size': 4, 'hidden_size': 16, 'num_layers': 1, 'dropout': 0.2, 'n_classes': self.data_constants['n_classes']},
            'tfrnet': {},
            'resnet1d': {'n_channels': 4, 'n_samples': 256, 'net_filter_size': [16, 16, 32, 32, 64],
                         'net_seq_length': [256, 64, 32, 16, 8], 'n_classes': self.data_constants['n_classes'], 'kernel_size': 31,
                         'dropout_rate': 0.5},
        }

        self.model_constants['multimodal'] = {
            'model1': 'resnet1d', 
            'model2': 'resnet1d', 
            'model1_kwargs':copy.deepcopy(self.model_constants['resnet1d']),
            'model2_kwargs':copy.deepcopy(self.model_constants['resnet1d']),
            'embed_dim1': 64,
            'embed_dim2': 32,
            'n_classes': 2,
            'device': 'cuda'
        }
