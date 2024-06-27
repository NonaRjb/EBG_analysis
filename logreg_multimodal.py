from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_validate
from mne.decoding import CSP, UnsupervisedSpatialFilter
from dataset.data_utils import load_ebg1_mat
from models.load_model import load_ml_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import math
import os

from dataset.data_utils import load_ebg4, apply_tfr, crop_tfr, crop_temporal, apply_baseline

cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Volumes/T5 EVO/Smell/"

model_kwargs = {
    'logreg': {
        'C': 1.0,
        'penalty': 'l1',
        'solver': 'liblinear',
        'max_iter': 2000,
        'random_state': 42
    },
    'gradboost': {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "subsample": 0.5,
        "max_leaf_nodes": 4,
        "max_depth": 5,
        "min_samples_split": 5,
        "alpha": 1.0,
        "random_state": 42
    },
    'xgboost': {
        "objective": 'binary:logistic',
        "n_estimators": 100,
        "learning_rate": 0.1,
        "subsample": 0.5,
        "max_leaf_nodes": 4,
        "max_depth": None,
        "min_loss_split": 2,
        "alpha": 1.0,
        "random_state": 42
    }
}


def confusion_matrix_scorer(clf_, X_, y):
    y_pred = clf_.predict(X_)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


def extract_features(data, times, feature_type, **kwargs):
    if feature_type == "tfr":
        freqs = np.arange(5, 100)
        tfr = apply_tfr(data, kwargs['sfreq'], freqs=freqs, n_cycles=3, method='dpss')

        # apply baseline correction
        tfr = apply_baseline(tfr, bl_lim=(-1.0, -0.6), tvec=times, mode='logratio')

        # crop the time interval of interest
        tfr = crop_tfr(tfr,
                       tmin=kwargs['tmin'], tmax=kwargs['tmax'],
                       fmin=kwargs['fmin'], fmax=kwargs['fmax'],
                       tvec=times, freqs=freqs,
                       w=w)

        n_trials = tfr.shape[0]
        n_time_samples = tfr.shape[-1]

        if kwargs['modality'] == "source":
            collapsed_tfr_mean = tfr.reshape((n_trials, 4, 12, 5, n_time_samples))
            tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
        elif kwargs['modality'] == "eeg":
                collapsed_tfr_mean = tfr.reshape((n_trials, 63, 12, 5, n_time_samples))
                tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
        else:
            # take the mean over channels
            tfr_mean = tfr.mean(axis=1).squeeze()
            collapsed_tfr_mean = tfr_mean.reshape(
                (n_trials, 12, 5, n_time_samples))  # 12, 5 is because I consider fmin=10 and fmax=70
            tfr_mean = np.mean(collapsed_tfr_mean, axis=2)

        return tfr_mean.reshape(n_trials, -1)

    elif feature_type == "raw_temporal":
        data_cropped = crop_temporal(data, kwargs['tmin'], kwargs['tmax'], times)

        if kwargs['modality'] == "sniff":
            percentile_95 = np.percentile(np.abs(data_cropped), 95, axis=-1, keepdims=True)
            data_cropped /= percentile_95

        return data_cropped


def load_data(name, root_path, subject_id, data_type, modality, tmin, tmax, bl_lim, binary):
    if name == "ebg1":
        return load_ebg1_array(
            root_path=root_path,
            subject_id=subject_id,
            modality=modality,
            tmin=tmin,
            tmax=tmax,
            bl_lim=None,
            binary=binary
        )
    elif name == "ebg4":
        return load_ebg4_array(
            root_path=root_path,
            subject_id=subject_id,
            data_type=data_type,
            modality=modality,
            tmin=tmin,
            tmax=tmax,
            bl_lim=bl_lim,
            binary=binary
        )


def load_ebg1_array(root_path, subject_id, modality, tmin, tmax, bl_lim=None, binary=True):
    root_path = root_path
    # if subject_id == 0:
    #     recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat' for subject_id in range(1, 31) if
    #                   subject_id != 4]
    # else:
    recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat']
    with open(os.path.join(root_path, 'kept_indices_dataset1.pkl'), 'rb') as f:
        indices_to_keep = pickle.load(f)

    data = None
    labels = None
    fs = None
    time_vec = None
    for i, recording in enumerate(recordings):
        file = os.path.join(root_path, recording)
        data_subj, label_subj, time_vec_subj, fs_subj = \
            load_ebg1_mat(file, indices_to_keep[recording])
        if fs is None:
            fs = float(fs_subj)
        if time_vec is None:
            time_vec = time_vec_subj

        if modality == 'eeg':
            data_subj = data_subj[:, :64, :]
        elif modality == 'ebg':
            data_subj = data_subj[:, 64:, :]
        else:
            pass

        if data is None:
            data = data_subj
            labels = np.expand_dims(label_subj, axis=1)
        else:
            data = np.vstack((data, data_subj))
            labels = np.vstack((labels, np.expand_dims(label_subj, axis=1)))

    if tmin is None:
        t_min = 0
    else:
        t_min = np.abs(time_vec - tmin).argmin()
    if tmax is None:
        t_max = len(time_vec)
    else:
        t_max = np.abs(time_vec - tmax).argmin()

    if binary:
        new_labels = [1. if label == 40 else 0. for label in labels]
        labels = new_labels
        class_0_count = new_labels.count(0.)
        class_1_count = new_labels.count(1.)
        print(f"N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")
    else:
        new_labels = [y / 10 - 1 for y in labels]
        labels = new_labels
        print(f"new_labels = {set(new_labels)}")

    if bl_lim is not None:
        baseline_min = np.abs(time_vec - bl_lim[0]).argmin()
        baseline_max = np.abs(time_vec - bl_lim[1]).argmin()
        baseline = np.mean(data[..., baseline_min:baseline_max], axis=(0, -1), keepdims=True)
        data = data[..., t_min:t_max] - baseline
    else:
        data = data[..., t_min:t_max]

    return data, labels, time_vec, fs


def load_ebg4_array(root_path, subject_id, data_type, modality, tmin, tmax, bl_lim=None, binary=True):
    # if subject_id == 0:
    #     subjects = [subject_id for subject_id in range(1, 26) if subject_id != 10]
    # else:
    subjects = [subject_id]

    data = None
    labels = None
    time_vec = None
    fs = None
    for subject in subjects:
        data_subj, labels_subj, time_vec_subj, fs_subj = \
            load_ebg4(root=root_path, subject_id=subject, data_type=data_type, fs_new=256)  # TODO
        if fs is None:
            fs = float(fs_subj)

        if time_vec is None:
            time_vec = time_vec_subj

        if modality == "sniff":
            data_subj, _, _, _ = load_ebg4(
                root=root_path,
                subject_id=subject,
                data_type="sniff",
                fs_new=200
            )
        elif data_type == 'sensor' or data_type == 'sensor_ica':
            if modality == 'eeg':
                data_subj = data_subj[:, :63, :]
            elif modality == 'ebg':
                data_subj = data_subj[:, 63:-1, :]
            else:
                pass

        if data is None:
            data = data_subj
            labels = np.expand_dims(labels_subj, axis=1)
        else:
            data = np.vstack((data, data_subj))
            labels = np.vstack((labels, np.expand_dims(labels_subj, axis=1)))

    if tmin is None:
        t_min = 0
    else:
        t_min = np.abs(time_vec - tmin).argmin()

    if tmax is None:
        t_max = len(time_vec)
    else:
        t_max = np.abs(time_vec - tmax).argmin()

    if binary:
        # only consider high intensity labels
        mask = np.logical_not(np.isin(labels.squeeze(), [1, 2, 4]))
        data = data[mask, ...]
        labels = labels[mask]
        # consider both low and high intensity labels
        new_labels = [1. if y == 64 else 0. for y in labels]
        labels = new_labels
        class_0_count = new_labels.count(0.)
        class_1_count = new_labels.count(1.)
        print(f"N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")
    else:
        new_labels = [math.log2(y) for y in labels]
        labels = new_labels
        print(f"new_labels = {set(new_labels)}")

    time_vec = time_vec[t_min:t_max]
    if bl_lim is not None:
        baseline_min = np.abs(time_vec - bl_lim[0]).argmin()
        baseline_max = np.abs(time_vec - bl_lim[1]).argmin()
        baseline = np.mean(data[..., baseline_min:baseline_max], axis=(0, -1), keepdims=True)
        data = data[..., t_min:t_max] - baseline
    else:
        data = data[..., t_min:t_max]

    return data, labels, time_vec, fs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ebg4')
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--tmin', type=float, default=-0.5)
    parser.add_argument('--tmax', type=float, default=0.25)
    parser.add_argument('-w', type=float, default=None)
    parser.add_argument('--fmin', type=float, default=20)
    parser.add_argument('--fmax', type=float, default=100)
    parser.add_argument('--c1', type=float, default=1.0)
    parser.add_argument('--c2', type=float, default=1.0)
    parser.add_argument('--modality1', type=str, default="ebg")
    parser.add_argument('--modality2', type=str, default="sniff")
    parser.add_argument('--model', type=str, default='logreg')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":

    loc = "remote"
    if loc == "local":
        data_path = local_data_path
        save_path = local_save_path
    else:
        data_path = cluster_data_path
        save_path = cluster_save_path

    args = parse_args()

    dataset_name = args.data
    seed = args.seed
    c1 = args.c1
    c2 = args.c2
    w = args.w
    modality1 = args.modality1
    modality2 = args.modality2

    for k in model_kwargs.keys():
        if 'random_state' in model_kwargs[k].keys():
            model_kwargs[k]['random_state'] = seed

    save_path = os.path.join(save_path, "plots")
    save_path = os.path.join(save_path, "grid_search_c" if w is None else "grid_search_c_tmin")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, dataset_name + "_" + args.model + "_" + modality1 + "_" + modality2)
    os.makedirs(save_path, exist_ok=True)
    splits_path = os.path.join(data_path, "splits_"+dataset_name)
    data_path = os.path.join(data_path, dataset_name)

    if dataset_name == "ebg1" and args.subject_id == 0:
        subject_ids = [i for i in range(1, 31) if i != 4]
    elif dataset_name == "ebg4" and args.subject_id == 0:
        subject_ids = [i for i in range(1, 54) if i != 10]
    elif args.subject_id != 0:
        subject_ids = [args.subject_id]
    else:
        raise NotImplementedError

    aucroc_scores = {}
    for subj in subject_ids:

        data_mod1, labels_array, t, sfreq = load_data(
            name=dataset_name,
            root_path=data_path,
            subject_id=subj,
            data_type="source" if modality1 == "source" else "sensor_ica",
            modality=modality1,
            tmin=None,
            tmax=None,
            bl_lim=None,
            binary=True
        )
        data_mod2, _, _, _ = load_data(
            name=dataset_name,
            root_path=data_path,
            subject_id=subj,
            data_type="source" if modality2 == "source" else "sensor_ica",
            modality=modality2,
            tmin=None,
            tmax=None,
            bl_lim=None,
            binary=True
        )

        mod1_features = extract_features(
            data_mod1,
            t,
            feature_type="raw_temporal" if modality1 == "sniff" else "tfr",
            sfreq=sfreq,
            tmin=args.tmin,
            tmax=args.tmax,
            fmin=args.fmin,
            fmax=args.fmax,
            modality=modality1
        )

        mod2_features = extract_features(
            data_mod2,
            t,
            feature_type="raw_temporal" if modality2 == "sniff" else "tfr",
            sfreq=sfreq,
            tmin=args.tmin,
            tmax=args.tmax,
            fmin=args.fmin,
            fmax=args.fmax,
            modality=modality2
        )

        y = np.asarray(labels_array)

        # skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        aucroc_scores[str(subj)] = []
        aucpr_scores = []

        for fold, (train_index, test_index) in enumerate(outer_cv.split(mod1_features, y)):
        # for i, fold in enumerate(os.listdir(os.path.join(splits_path, str(subj)))):

            # with open(os.path.join(splits_path, str(subj), fold), 'rb') as f:
            #     split = pickle.load(f)
            # train_index = split['train']
            # test_index = split['val']

            model_kwargs['logreg']['C'] = c1 # float(best_c_mod1[str(subj)])
            clf1 = load_ml_model(model_name=args.model, modality=modality1, **model_kwargs[args.model])
            model_kwargs['logreg']['C'] = c2 # float(best_c_mod2[str(subj)])
            clf2 = load_ml_model(model_name=args.model, modality=modality2, **model_kwargs[args.model])

            X1_train, X1_test = mod1_features[train_index], mod1_features[test_index]
            X2_train, X2_test = mod2_features[train_index], mod2_features[test_index]
            y_train, y_test = y[train_index], y[test_index]

            inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)

            space = dict()
            # space['logreg__C'] = [0.5, 1, 2, 4, 8, 16, 32, 64]
            space[f'{args.model}__C'] = [math.exp(x) for x in range(-1, 10)]
            search1 = GridSearchCV(clf1, space, scoring='roc_auc', cv=inner_cv, refit=True)
            result1 = search1.fit(X1_train, y_train)
            best_clf1 = result1.best_estimator_

            space = dict()
            # space['logreg__C'] = [0.5, 1, 2, 4, 8, 16, 32, 64]
            space[f'{args.model}__C'] = [math.exp(x) for x in range(-1, 10)]
            search2 = GridSearchCV(clf2, space, scoring='roc_auc', cv=inner_cv, refit=True)
            result2 = search2.fit(X2_train, y_train)
            best_clf2 = result2.best_estimator_

            # # print(f"Fold {fold + 1}:")
            # print(f"Fold {i + 1}:")
            # clf1 = clf1.fit(X1_train, y_train)
            # clf2 = clf2.fit(X2_train, y_train)
            sum_aucs = result1.best_score_ + result2.best_score_
            alpha1 = result1.best_score_ / sum_aucs
            alpha2 = result2.best_score_ / sum_aucs

            print(f"alpha1 = {alpha1}, alpha2 = {alpha2}")
            prob1_scores = best_clf1.predict_proba(X1_test)[:, 1] * alpha1
            prob2_scores = best_clf2.predict_proba(X2_test)[:, 1] * alpha2

            prob_scores = np.concatenate((np.expand_dims(prob1_scores, axis=1), np.expand_dims(prob2_scores, axis=1)),
                                         axis=1)
            prob_scores = prob_scores.mean(axis=-1)
            aucroc_score = roc_auc_score(y_test, prob_scores, average='weighted')

            aucroc_scores[str(subj)].append(aucroc_score)
            print(f"Test Weighted AUCROC = {aucroc_score}")
            print("")

        print("median test auc = ", np.median(np.asarray(aucroc_scores[str(subj)])))        
        if args.save is True:
            print("Saving the AUC Scores")
            os.makedirs(os.path.join(save_path, str(subj)), exist_ok=True)
            np.save(
                os.path.join(save_path, str(subj), f"{args.tmin}_{args.tmax}.npy"),
                np.asarray(aucroc_scores[str(subj)])
            )
