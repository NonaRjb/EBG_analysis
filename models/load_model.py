from models.architectures import EEGNet, EEGNet1D, TFRNet, LSTMClassifier, RNNClassifier, ResNet1d


def load(model_name: str, **kwargs):
    if model_name == 'eegnet':
        return EEGNet(**kwargs)
    if model_name == 'eegnet1d':
        return EEGNet1D(**kwargs)
    elif model_name == 'tfrnet':
        return TFRNet()
    elif model_name == 'lstm':
        return LSTMClassifier(**kwargs)
    elif model_name == 'rnn':
        return RNNClassifier(**kwargs)
    # elif model_name == 'eegnet_attention':
    #     return AttentionEEGNet(**kwargs)
    elif model_name == 'resnet1d':
        return ResNet1d(**kwargs)
    else:
        raise NotImplementedError
