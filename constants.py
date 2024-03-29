import numpy as np


data_constants = {
    'tmin': -0.1,
    'tmax': 0.25,
    'fmin': None,
    'fmax': None,
    'binary': True,
    'train_size': 0.8,
    'val_size': 0.2,
    'ebg_transform': None,
    'shuffle_labels': False,
    'modality': 'ebg',
    'tfr_freqs': np.linspace(20, 100, 160),
    'baseline_type': 'zscore'
}

if data_constants['fmax'] is not None and data_constants['fmin'] is not None:
    lstm_input_size = np.abs(data_constants['tfr_freqs'] - data_constants['fmax']).argmin() - \
                      np.abs(data_constants['tfr_freqs'] - data_constants['fmin']).argmin() + 1
elif data_constants['fmax'] is not None:
    lstm_input_size = np.abs(data_constants['tfr_freqs'] - data_constants['fmax']).argmin()
elif data_constants['fmin'] is not None:
    lstm_input_size = len(data_constants['tfr_freqs']) - \
                      np.abs(data_constants['tfr_freqs'] - data_constants['fmin']).argmin()
else:
    lstm_input_size = len(data_constants['tfr_freqs'])

model_constants = {
    'eegnet': {'n_channels': 4, 'n_classes': 2},
    'eegnet1d': {'n_channels': 4, 'n_classes': 2},
    'eegnet_attention': {'n_channels': 68, 'n_classes': 2},
    'lstm': {'input_size': 4, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.5, 'n_classes': 2},
    'rnn': {'input_size': 4, 'hidden_size': 16, 'num_layers': 1, 'dropout': 0.2, 'n_classes': 2},
    'tfrnet': {},
    'resnet1d': {'n_channels': 4, 'n_samples': 256, 'net_filter_size':[16, 16, 32, 32, 64], 'net_seq_length':[256, 128, 64, 32, 16], 'n_classes':2 , 'kernel_size':31, 'dropout_rate':0.5}
}


