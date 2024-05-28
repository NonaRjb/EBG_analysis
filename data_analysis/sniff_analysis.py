import sys

sys.path.append("/proj/berzelius-2023-338/users/x_nonra/EBG_analysis")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

from dataset.data_utils import load_ebg4, crop_temporal

cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Volumes/T5 EVO/Smell/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--tmin', type=float, default=-0.8)
    parser.add_argument('--tmax', type=float, default=1.7)
    parser.add_argument('-w', type=float, default=None)
    parser.add_argument('-c', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":

    loc = "remote"

    args = parse_args()

    if loc == "local":
        data_path = local_data_path
        save_path = local_save_path
    else:
        data_path = cluster_data_path
        save_path = cluster_save_path
    
    splits_path = os.path.join(data_path, "splits_ebg4")
    data_path = os.path.join(data_path, "ebg4")
    save_path = os.path.join(save_path, "plots")
    save_path = os.path.join(save_path, "grid_search_c" if args.w is None else "grid_search_c_tmin")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "sniff")
    os.makedirs(save_path, exist_ok=True)
    
    seed = args.seed
    c = args.c
    if args.subject_id == -1:
        pass
    else:
        if args.subject_id == 0:
            subjects = [i for i in range(1, 54) if i != 10]
        else:
            subjects = [args.subject_id]

        scores = {}
        for subject in subjects:
            data_sniff, labels_sniff, times_sniff, fs_sniff = load_ebg4(
                data_path,
                subject,
                data_type="sniff",
                fs_new=200
            )
            # only consider high intensity labels
            labels_sniff = np.array(labels_sniff)
            mask = np.logical_not(np.isin(labels_sniff.squeeze(), [1, 2, 4]))
            data_sniff = data_sniff[mask, ...]
            labels_sniff = labels_sniff[mask]
            # consider both low and high intensity labels
            data_sniff_cropped = crop_temporal(data_sniff, tmin=args.tmin, tmax=args.tmax, tvec=times_sniff, w=args.w)
            new_labels = [1. if y == 64 else 0. for y in labels_sniff]
            labels_sniff = np.asarray(new_labels)

            percentile_95 = np.percentile(np.abs(data_sniff_cropped), 95, axis=-1, keepdims=True)
            data_sniff_cropped /= percentile_95

            # skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
            scores[str(subject)] = []
            # for fold, (train_idx, test_idx) in enumerate(skf.split(data_sniff_cropped, labels_sniff)):
            for i, fold in enumerate(os.listdir(os.path.join(splits_path, str(subject)))):

                with open(os.path.join(splits_path, str(subject), fold), 'rb') as f:
                    split = pickle.load(f)
                train_idx = split['train']
                test_idx = split['val']

                X_train, y_train = data_sniff_cropped[train_idx], labels_sniff[train_idx]
                X_test, y_test = data_sniff_cropped[test_idx], labels_sniff[test_idx]

                clf = make_pipeline(# StandardScaler(),
                                    LogisticRegression(C=c, penalty='l1', solver='liblinear',  # l1_ratio=0.5,
                                                       max_iter=2000,
                                                       random_state=seed))

                # Count samples from each class in train and test sets
                unique_train, counts_train = np.unique(y_train, return_counts=True)
                unique_test, counts_test = np.unique(y_test, return_counts=True)

                # print(f"Fold {fold + 1}:")
                print(f"Fold {i + 1}:")

                clf = clf.fit(X_train, y_train)
                prob_scores = clf.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, prob_scores, average='weighted')
                scores[str(subject)].append(auc_score)
                print("auc score = ", auc_score)

            print("Average AUC score = ", np.asarray(scores[str(subject)]).mean)            
            if args.save is True:
                print("Saving the AUC Scores")
                os.makedirs(os.path.join(save_path, str(subject)), exist_ok=True)
                np.save(
                    os.path.join(save_path, str(subject), f"c{c}.npy" if args.w is None else f"c{c}_t{args.tmin}.npy"),
                    np.asarray(scores[str(subject)])
                )
