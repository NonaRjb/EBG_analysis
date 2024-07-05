from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from mne.decoding import CSP, UnsupervisedSpatialFilter
from dataset.data_utils import load_ebg1_mat, tfr_feature_extract
from models.load_model import load_ml_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import math
import os

from dataset.data_utils import load_ebg4, apply_tfr, crop_tfr, crop_temporal, apply_baseline

# cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
# cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Volumes/T5 EVO/Smell/"

model_kwargs = {
    'logreg':{
        'C': 1.0, 
        'penalty': 'l1', 
        'solver': 'liblinear',
        'max_iter': 2000,
        'random_state': 42
        },
    'svm':{
        'C': 1.0,
        'kernel': 'linear',
        'gamma': 'auto',
        'probability': True
        }, 
    'rf':{
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "random_state": 42
    },
    'gradboost':{
        "n_estimators": 100,
        "learning_rate": 0.1,
        "subsample": 0.5, 
        "max_leaf_nodes": 4,
        "max_depth": 5,
        "min_samples_split": 5,
        "ccp_alpha": 1.0,
        "random_state": 42
    },
    'xgboost':{
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

time_windows = [(0.00, 0.25), (0.15, 0.40), (0.30, 0.55), (0.45, 0.70), (0.60, 0.85), (0.75, 1.0)]
# time_windows = [(0.00, 0.3), (0.2, 0.50), (0.40, 0.7), (0.6, 0.9)]



def confusion_matrix_scorer(clf_, X_, y):
    y_pred = clf_.predict(X_)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


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
            load_ebg4(root=root_path, subject_id=subject, data_type=data_type, fs_new=256) #TODO
        if fs is None:
            fs = float(fs_subj)

        if time_vec is None:
            time_vec = time_vec_subj

        if data_type == 'sensor' or data_type == 'sensor_ica':
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
    parser.add_argument('--data', type=str, default='ebg4_sensor')
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--tmin', type=float, default=-0.5)
    parser.add_argument('--tmax', type=float, default=0.25)
    parser.add_argument('-w', type=float, default=None)
    parser.add_argument('--fmin', type=float, default=20)
    parser.add_argument('--fmax', type=float, default=100)
    parser.add_argument('-c', type=float, default=1.0)
    parser.add_argument('--data_type', type=str, default="sensor_ica")
    parser.add_argument('--modality', type=str, default="ebg")
    parser.add_argument('--model', type=str, default='logreg')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default="grid_search_c")
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
    c = args.c
    w = args.w
    task = args.task

    if task == "grid_search_c": 
        time_windows = [(args.tmin, args.tmax)]

    for k in model_kwargs.keys():
        if 'random_state' in model_kwargs[k].keys():
            model_kwargs[k]['random_state'] = seed
        if 'C' in model_kwargs[k].keys():
            model_kwargs[k]['C'] = c
        if 'alpha' in model_kwargs[k].keys():
            model_kwargs[k]['alpha'] = c

    save_path = os.path.join(save_path, "plots")
    # save_path = os.path.join(save_path, "grid_search_c" if w is None else "grid_search_c_tmin")
    save_path = os.path.join(save_path, task)
    os.makedirs(save_path, exist_ok=True)
    if args.data_type != "source":
        save_path = os.path.join(save_path, dataset_name+"_"+args.modality+"_"+args.model)
    else:
        save_path = os.path.join(save_path, dataset_name+"_"+args.data_type+"_"+args.model)
    
    os.makedirs(save_path, exist_ok=True)
    splits_path = os.path.join(data_path, "splits_"+dataset_name)
    data_path = os.path.join(data_path, dataset_name)

    if args.subject_id == 0:
        if dataset_name == "ebg1":
            subject_ids = [i for i in range(1, 31) if i != 4]
        elif dataset_name == "ebg4":
            subject_ids = [i for i in range(1, 54) if i != 10]
        else:
            raise NotImplementedError
    else:
        subject_ids = [args.subject_id]

    aucroc_scores = {}
    for subj in subject_ids:
            
        data_array, labels_array, t, sfreq = \
        load_data(
            name=dataset_name,
            root_path=data_path,
            subject_id=subj,
            data_type=args.data_type,
            modality=args.modality,
            tmin=None,
            tmax=None,
            bl_lim=None,
            binary=True
        )

        freqs = np.arange(5, 100)
        tfr = apply_tfr(data_array, sfreq, freqs=freqs, n_cycles=3, method='morlet')
        tfr = apply_baseline(tfr, bl_lim=(-1.0, -0.6), tvec=t, mode='logratio')

        n_trials = tfr.shape[0]
        y = np.asarray(labels_array)
        data = data_array

        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        # outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=seed)

        aucroc_scores[str(subj)] = []
        for fold, (train_index, test_index) in enumerate(outer_cv.split(data, y)):
            best_models_win = []
            best_results_win = []
            best_params_win = []
            for win in time_windows:
                tfr_cropped = crop_tfr(tfr, tmin=win[0], tmax=win[1], fmin=args.fmin, fmax=args.fmax, tvec=t, freqs=freqs, w=w)
                data_array = crop_temporal(data_array, win[0], win[1], t)
                # tfr_cropped1 = crop_tfr(tfr, tmin=win[0], tmax=win[1], fmin=50, fmax=70, tvec=t, freqs=freqs, w=w)
                # tfr_cropped2 = crop_tfr(tfr, tmin=win[0], tmax=win[1], fmin=12, fmax=16, tvec=t, freqs=freqs, w=w)
                # tfr_cropped = np.concatenate((tfr_cropped1, tfr_cropped2), axis=2)

                n_time_samples = tfr_cropped.shape[-1]
                if args.data_type == "source":
                    collapsed_tfr_mean = tfr_cropped.reshape((n_trials, 4, 12, 5, n_time_samples))
                    tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
                elif args.modality == "eeg":
                    collapsed_tfr_mean = tfr_cropped.reshape((n_trials, 63, 12, 5, n_time_samples))
                    tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
                else:
                # take the mean over channels
                    tfr_mean = tfr_cropped.mean(axis=1).squeeze()
                    collapsed_tfr_mean = tfr_mean.reshape((n_trials, 12, 5, n_time_samples))  # 12, 5 is because I consider fmin=10 and fmax=70
                    tfr_mean = np.mean(collapsed_tfr_mean, axis=2)
                # if args.data_type == "source":
                #     tfr_mean = tfr_cropped
                # elif args.modality == "eeg":
                #     # collapsed_tfr_mean = tfr_cropped.reshape((n_trials, 63, 6, 5, n_time_samples))
                #     # tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
                #     tfr_mean = tfr_cropped
                # else:
                #     tfr_mean = tfr_cropped.mean(axis=1).squeeze()
            
                # tfr_mean = tfr_feature_extract(tfr_cropped)
                
                X = tfr_mean.reshape(n_trials, -1)
                X_train, y_train = X[train_index], y[train_index]

                inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)
                clf = load_ml_model(model_name=args.model, **model_kwargs[args.model])
                space = dict()
                # space['logreg__C'] = [0.5, 1, 2, 4, 8, 16, 32, 64]
                if args.model == "gradboost": 
                    space[f'{args.model}__n_estimators'] = [50, 100, 150, 200]
                    space[f'{args.model}__max_depth'] = [3, 5, 7]
                if args.model == "svm" and model_kwargs['svm']['kernel'] == "rbf":
                    space[f'{args.model}__C'] = [math.exp(x) for x in range(-1, 7)]
                    space[f'{args.model}__gamma'] = [pow(10, x) for x in range(-3, 3)]
                if args.model == "svm" and model_kwargs['svm']['kernel'] == "poly":
                    space[f'{args.model}__C'] = [math.exp(x) for x in range(-1, 7)]
                    space[f'{args.model}__gamma'] = [pow(10, x) for x in range(-3, 3)]
                    space[f'{args.model}__degree'] = [x for x in [2, 3, 5, 7]]
                if args.model == "rf": 
                    space[f'{args.model}__n_estimators'] = [50, 100, 150, 200]
                    space[f'{args.model}__max_depth'] = [3, 5]
                    space[f'{args.model}__min_samples_leaf'] = [1, 3, 5]
                else:
                    space[f'{args.model}__C'] = [math.exp(x) for x in range(-1, 10)]
                search = GridSearchCV(clf, space, scoring='roc_auc', cv=inner_cv, refit=True, error_score='raise')
                result = search.fit(X_train, y_train)
                best_model = result.best_estimator_
                # evaluate model on the hold out dataset
                # prob_scores = best_model.predict_proba(X_test)[:, 1]
                # clf = clf.fit(X_train, y_train)
                # prob_scores = clf.predict_proba(X_test)[:, 1]
                # aucroc_score = roc_auc_score(y_test, prob_scores, average='weighted')
                # aucroc_scores.append(aucroc_score)
                # print('>acc=%.3f, est=%.3f, cfg=%s' % (aucroc_score, result.best_score_, result.best_params_))
                # print(f"Model AUCROC = {aucroc_score}")
                best_models_win.append(best_model)
                best_results_win.append(result.best_score_)
                best_params_win.append(result.best_params_)
            
            best_result_final = max(best_results_win)
            best_model_final = best_models_win[best_results_win.index(best_result_final)]
            best_win = time_windows[best_results_win.index(best_result_final)]
            best_c = best_params_win[best_results_win.index(best_result_final)]

            print(f"Best Window is {best_win}, (C = {best_c})")

            tfr_cropped = crop_tfr(tfr, tmin=best_win[0], tmax=best_win[1], fmin=args.fmin, fmax=args.fmax, tvec=t, freqs=freqs, w=w)
            # tfr_cropped1 = crop_tfr(tfr, tmin=best_win[0], tmax=best_win[1], fmin=50, fmax=70, tvec=t, freqs=freqs, w=w)
            # tfr_cropped2 = crop_tfr(tfr, tmin=best_win[0], tmax=best_win[1], fmin=12, fmax=16, tvec=t, freqs=freqs, w=w)
            # tfr_cropped = np.concatenate((tfr_cropped1, tfr_cropped2), axis=2)
            
            n_time_samples = tfr_cropped.shape[-1]
            if args.data_type == "source":
                collapsed_tfr_mean = tfr_cropped.reshape((n_trials, 4, 12, 5, n_time_samples))
                tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
            elif args.modality == "eeg":
                collapsed_tfr_mean = tfr_cropped.reshape((n_trials, 63, 12, 5, n_time_samples))
                tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
            else:
                tfr_mean = tfr_cropped.mean(axis=1).squeeze()
                collapsed_tfr_mean = tfr_mean.reshape((n_trials, 12, 5, n_time_samples))  # 12, 5 is because I consider fmin=10 and fmax=70
                tfr_mean = np.mean(collapsed_tfr_mean, axis=2)
            # if args.data_type == "source":
            #     tfr_mean = tfr_cropped
            # elif args.modality == "eeg":
            #     # collapsed_tfr_mean = tfr_cropped.reshape((n_trials, 63, 6, 5, n_time_samples))
            #     # tfr_mean = np.mean(collapsed_tfr_mean, axis=3)
            #     tfr_mean = tfr_cropped
            # else:
            #     tfr_mean = tfr_cropped.mean(axis=1).squeeze()
            
            # tfr_mean = tfr_feature_extract(tfr_cropped)
            
            X = tfr_mean.reshape(n_trials, -1)

            X_test, y_test = X[test_index], y[test_index]

            prob_scores = best_model_final.predict_proba(X_test)[:, 1]
            aucroc_score = roc_auc_score(y_test, prob_scores, average='weighted')
            aucroc_scores[str(subj)].append(aucroc_score)

            print(f"Best Model's:  Val Score = {best_result_final}, Test Score = {aucroc_score}")

        print(f"Median AUC: {np.median(np.asarray(aucroc_scores[str(subj)]))}")
        if args.save is True:
                print("Saving the AUC Scores")
                os.makedirs(os.path.join(save_path, str(subj)), exist_ok=True)
                np.save(
                    os.path.join(save_path, str(subj), f"{best_win[0]}_{best_win[1]}.npy"),
                    np.asarray(aucroc_scores[str(subj)])
                )