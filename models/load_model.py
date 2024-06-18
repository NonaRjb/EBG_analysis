from models.architectures import EEGNet, EEGNet1D, TFRNet, LSTMClassifier, RNNClassifier, ResNet1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import ensemble
# from xgboost import XGBClassifier


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


def load_ml_model(model_name, modality=None, **kwargs):
    if model_name == "logreg":
        if modality == "sniff":
            return Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.99)), ('logreg', LogisticRegression(**kwargs))])
        else:
            return Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(**kwargs))])
    if model_name == "gradboost":
        return ensemble.GradientBoostingClassifier(**kwargs)
    if model_name == "xgboost":
        pass
        # return XGBClassifier(**kwargs)
    if model_name == "svm":
        return Pipeline([('scaler', StandardScaler()), ('svm', SVC(**kwargs))])
    if model_name == "lda":
        return LinearDiscriminantAnalysis(**kwargs)