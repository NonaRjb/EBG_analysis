import sys

sys.path.append("/proj/berzelius-2023-338/users/x_nonra/EBG_analysis")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import math
import os

from dataset.data_utils import load_ebg4, crop_temporal
from models.load_model import load_ml_model

cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
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
        'probability': True
        }
}

time_windows = [(0.00, 0.25), (0.15, 0.40), (0.30, 0.55), (0.45, 0.70), (0.60, 0.85), (0.75, 1.0)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--tmin', type=float, default=-0.8)
    parser.add_argument('--tmax', type=float, default=1.7)
    parser.add_argument('-w', type=float, default=None)
    parser.add_argument('-c', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default="grid_search_c")
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
    save_path = os.path.join(save_path, args.task)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "sniff")
    os.makedirs(save_path, exist_ok=True)

    if args.task == "grid_search_c": 
        time_windows = [(args.tmin, args.tmax)]
    
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
        win = (args.tmin, args.tmax)
        for subject in subjects:
            data_sniff, labels_sniff, times_sniff, fs_sniff = load_ebg4(
                data_path,
                subject,
                data_type="sniff",
                fs_new=200
            )
            print("n_time_samples = ", data_sniff.shape[-1])
            # only consider high intensity labels
            labels_sniff = np.array(labels_sniff)
            mask = np.logical_not(np.isin(labels_sniff.squeeze(), [1, 2, 4]))
            data_sniff = data_sniff[mask, ...]
            labels_sniff = labels_sniff[mask]
            new_labels = [1. if y == 64 else 0. for y in labels_sniff]
            labels_sniff = np.asarray(new_labels)

            outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            scores[str(subject)] = {'val': [], 'test': []}
            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data_sniff, labels_sniff)):

                data_sniff_cropped = crop_temporal(data_sniff, tmin=win[0], tmax=win[1], tvec=times_sniff, w=args.w)
                # print("n_time_samples = ", data_sniff_cropped.shape[-1])                    
                percentile_95 = np.percentile(np.abs(data_sniff_cropped), 95, axis=-1, keepdims=True)
                data_sniff_cropped /= percentile_95
                X_train, y_train = data_sniff_cropped[train_idx], labels_sniff[train_idx]
                X_test, y_test = data_sniff_cropped[test_idx], labels_sniff[test_idx]
                clf = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.99)), 
                    ('logreg', LogisticRegression(C=c, penalty='l1', solver='liblinear',  # l1_ratio=0.5,
                                                    max_iter=2000,
                                                    random_state=seed))
                ])
                inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
                
                space = dict()
                # space['logreg__C'] = [0.5, 1, 2, 4, 8, 16, 32, 64]
                space['logreg__C'] = [math.exp(x) for x in range(-1, 10)]
                search = GridSearchCV(clf, space, scoring='roc_auc', cv=inner_cv, refit=True)
                result = search.fit(X_train, y_train)
                best_model = result.best_estimator_
                prob_scores = best_model.predict_proba(X_test)[:, 1]
                test_auroc = roc_auc_score(y_test, prob_scores, average='weighted')
                
                print(f"Val AUC = {result.best_score_} | Test AUC = {test_auroc}")
                scores[str(subject)]['val'].append(result.best_score_)
                scores[str(subject)]['test'].append(test_auroc)

                
            print(f"Average Val AUC: {np.median(np.asarray(scores[str(subject)]['val']))}")
            print(f"Average Test AUC: {np.median(np.asarray(scores[str(subject)]['test']))}")
            if args.save is True:
                print("Saving the AUC Scores")
                os.makedirs(os.path.join(save_path, str(subject)), exist_ok=True)
                with open(os.path.join(save_path, str(subject), f"{win[0]}_{win[1]}.pkl"), 'wb') as f:
                    pickle.dump(scores[str(subject)], f)
