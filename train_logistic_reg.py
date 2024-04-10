from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import os

from dataset.data_utils import load_ebg4, apply_tfr, crop, apply_baseline


cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"


def confusion_matrix_scorer(clf_, X_, y):
    y_pred = clf_.predict(X_)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


def load_eeg_array(root_path, subject_id, data_type, modality, tmin, tmax, bl_lim=None, binary=True):
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
            load_ebg4(root=root_path, subject_id=subject, data_type=data_type)
        if fs is None:
            fs = float(fs_subj)

        if time_vec is None:
            time_vec = time_vec_subj

        if data_type == 'sensor' or data_type == 'sensor_ica':
            if modality == 'eeg':
                data_subj = data_subj[:, :64, :]
            elif modality == 'ebg':
                data_subj = data_subj[:, 64:, :]
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
    parser.add_argument('--data', type=str, default='ebg4_source')
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--tmin', type=float, default=-0.1)
    parser.add_argument('--tmax', type=float, default=1.0)
    parser.add_argument('--fmin', type=float, default=20)
    parser.add_argument('--fmax', type=float, default=100)
    parser.add_argument('--data_type', type=str, default="sensor_ica")
    parser.add_argument('--modality', type=str, default="ebg")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_name = args.data
    fmin = args.fmin
    fmax = args.fmax
    seed = args.seed

    data_path = os.path.join(local_data_path, "ebg4")

    if args.subject_id == 0:
        subject_ids = [i for i in range(1, 26) if i != 10]
        scores = {}
        for subj in subject_ids:
            data_array, labels_array, t, sfreq = \
                load_eeg_array(
                    root_path=data_path,
                    subject_id=subj,
                    data_type=args.data_type,
                    modality=args.modality,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    bl_lim=None
                )

            freqs = np.arange(fmin, fmax)
            tfr = apply_tfr(data_array, sfreq, freqs=freqs, n_cycles=3, method='dpss')

            # apply baseline correction
            tfr = apply_baseline(tfr, bl_lim=(None, None), tvec=t, mode='logratio')

            # crop the time interval of interest
            tfr = crop(tfr, tmin=0.06, tmax=0.1, fmin=45, fmax=70, tvec=t, freqs=freqs)
            # take the mean over channels
            tfr_mean = tfr.mean(axis=1).squeeze()
            n_trials = tfr_mean.shape[0]
            X = tfr_mean.reshape(n_trials, -1)
            y = np.asarray(labels_array)

            skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
            scores[str(subj)] = []
            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

                clf = LogisticRegression(C=.6, penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=2000,
                                         random_state=seed)
                # clf = make_pipeline(StandardScaler(),
                #                     SVC(C=.6,
                #                         kernel='rbf',
                #                         degree=3,
                #                         probability=True,
                #                         class_weight='balanced',
                #                         random_state=seed))

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Count samples from each class in train and test sets
                unique_train, counts_train = np.unique(y_train, return_counts=True)
                unique_test, counts_test = np.unique(y_test, return_counts=True)

                print(f"Fold {fold + 1}:")
                print(f"  Train - Class 0: {counts_train[0]}, Class 1: {counts_train[1]}")
                print(f"  Test  - Class 0: {counts_test[0]}, Class 1: {counts_test[1]}")

                clf = clf.fit(X_train, y_train)
                prob_scores = clf.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, prob_scores, average='weighted')
                scores[str(subj)].append(auc_score)
        score_values = scores.values()
        score_keys = scores.keys()
        plt.figure(figsize=(40, 6))
        plt.boxplot(score_values, labels=score_keys)
        plt.title('Boxplot of AUC Scores for Each Subject')
        plt.xlabel('Subject ID')
        plt.ylabel('AUC Score')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.savefig(os.path.join(local_data_path, "plots", "ebg4_auc_logistic_reg", f"auc_box_plots_svm.png"))
        plt.show()
    else:

        data_array, labels_array, t, sfreq = \
            load_eeg_array(
                root_path=data_path,
                subject_id=args.subject_id,
                data_type=args.data_type,
                modality=args.modality,
                tmin=args.tmin,
                tmax=args.tmax,
                bl_lim=None
            )

        freqs = np.arange(fmin, fmax)
        tfr = apply_tfr(data_array, sfreq, freqs=freqs, n_cycles=3, method='dpss')

        # apply baseline correction
        tfr = apply_baseline(tfr, bl_lim=(None, None), tvec=t, mode='logratio')

        # crop the time interval of interest
        tfr = crop(tfr, tmin=0.06, tmax=0.1, fmin=45, fmax=70, tvec=t, freqs=freqs)
        # take the mean over channels
        tfr_mean = tfr.mean(axis=1).squeeze()
        n_trials = tfr_mean.shape[0]
        X = tfr_mean.reshape(n_trials, -1)
        y = np.asarray(labels_array)

        clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=2000, random_state=seed)

        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

        scores = []
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Count samples from each class in train and test sets
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)

            print(f"Fold {fold + 1}:")
            print(f"  Train - Class 0: {counts_train[0]}, Class 1: {counts_train[1]}")
            print(f"  Test  - Class 0: {counts_test[0]}, Class 1: {counts_test[1]}")

            clf = clf.fit(X_train, y_train)
            prob_scores = clf.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, prob_scores, average='weighted')
            scores.append(auc_score)

        plt.hist(scores, bins=10)
        plt.title(f"AUC Scores Subject {args.subject_id}")
        plt.savefig(os.path.join(local_data_path, "plots", "ebg4_auc_logistic_reg", f"subj_{args.subject_id}.png"))
        plt.show()



