import pickle

import matplotlib.pyplot as plt
import numpy as np
import os
import re

def compare_logreg_c(root_path, save_path):

    pattern = r's\d+_c([\d.]+)\.npy'

    subjects = os.listdir(root_path)
    for subject in subjects:

        scores = {}
        medians = {}
        for filename in os.listdir(os.path.join(root_path, subject)):
            match = re.match(pattern, filename)
            if match:
                c = match.group(1)
            scores[c] = np.load(os.path.join(root_path, subject, filename))
            medians[c] = np.median(scores[c])

        # sort dict
        c_values = list(scores.keys())
        c_values.sort()
        sorted_scores = {i: scores[i] for i in c_values}
        auc_scores = sorted_scores.values()

        plt.figure(figsize=(15, 6))
        plt.boxplot(auc_scores, labels=c_values)
        plt.title(f'Boxplot of AUC Scores for Subject {subject}')
        plt.xlabel('C Values')
        plt.ylabel('AUC Score')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.savefig(os.path.join(save_path, subject, f"c_auc_box_plots.png"))
        plt.close()
        print(f"\n*********** Subject {subject} ************\n")
        print(medians)
        max_key = max(medians, key=lambda k: medians[k])
        print(f"maximum auc median: {medians[max_key]}, C = {max_key}")

    return


def load_dnn_subj_results(root_path, subject_id):
    filenames = os.listdir(os.path.join(root_path, str(subject_id)))
    aucs = []
    epochs = []
    for fn in filenames:
        with open(os.path.join(root_path, str(subject_id), fn), 'rb') as f:
            res = pickle.load(f)
            aucs.append(res['auroc'][0].detach().numpy())
            epochs.append(res['epoch'][0])
    return aucs, epochs


def plot_dnn_res(root_path, save_path):
    subjects = [i for i in range(1, 26) if i != 10]
    aucs = {}
    epochs = {}
    for subject in subjects:
        aucs[str(subject)], epochs[str(subject)] = load_dnn_subj_results(root_path, subject)

    auc_values = aucs.values()
    auc_keys = aucs.keys()
    plt.figure(figsize=(40, 6))
    plt.boxplot(auc_values, labels=auc_keys)
    plt.grid(axis='y', color='0.97')
    plt.title('Boxplot of AUC Scores for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC Score')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.savefig(
        os.path.join(save_path, "plots", "ebg4_dnn", f"auc_box_plot_eegnet1d.png"))

    epoch_vals = epochs.values()
    epoch_keys = epochs.keys()
    plt.figure(figsize=(40, 6))
    plt.boxplot(epoch_vals, labels=epoch_keys)
    plt.grid(axis='y', color='0.97')
    plt.title('Boxplot of Epochs with Best AUC Score for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('Epoch')
    plt.savefig(
        os.path.join(save_path, "plots", "ebg4_dnn", f"epoch_box_plot_eegnet1d.png"))

    plt.show()


if __name__ == "__main__":
    task = "plot_dnn_res"

    if task == "compare_logreg_c":
        path_to_data = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/grid_search_c"
        path_to_save = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/ebg4_auc_box_plot"
        compare_logreg_c(path_to_data, path_to_save)
    elif task == "plot_dnn_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_ebg"
        path_to_save = "/Volumes/T5 EVO/Smell"
        plot_dnn_res(path_to_data, path_to_save)
