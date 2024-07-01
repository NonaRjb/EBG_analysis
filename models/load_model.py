from models.architectures import EEGNet, EEGNet1D, TFRNet, LSTMClassifier, RNNClassifier, ResNet1d, MLP, MultiModalNet
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
    elif model_name == "mlp":
        return MLP(**kwargs)
    elif model_name == 'lstm':
        return LSTMClassifier(**kwargs)
    elif model_name == 'rnn':
        return RNNClassifier(**kwargs)
    elif model_name == 'resnet1d':
        return ResNet1d(**kwargs)
    elif model_name == 'multimodal':
        model1 = load(kwargs['model1'], **kwargs['model1_kwargs'])
        model2 = load(kwargs['model2'], **kwargs['model2_kwargs'])
        return MultiModalNet(model1, model2, embed_dim1=kwargs['embed_dim1'], 
                             embed_dim2=kwargs['embed_dim2'], 
                             n_classes=kwargs['n_classes'], 
                             device=kwargs['device'], 
                             model1_name=kwargs['model1'], 
                             model2_name=kwargs['model2'])
    else:
        raise NotImplementedError


def load_ml_model(model_name, modality=None, **kwargs):
    if model_name == "logreg":
        if modality == "sniff":
            return Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.99)), ('logreg', LogisticRegression(**kwargs))])
        else:
            return Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(**kwargs))])
    if model_name == "gradboost":
        return Pipeline([('scaler', StandardScaler()), ('gradboost', ensemble.GradientBoostingClassifier(**kwargs))])
    if model_name == "xgboost":
        pass
        # return XGBClassifier(**kwargs)
    if model_name == "svm":
        return Pipeline([('scaler', StandardScaler()), ('svm', SVC(**kwargs))])
    if model_name == "lda":
        return LinearDiscriminantAnalysis(**kwargs)