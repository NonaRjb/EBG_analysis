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


def load_dnn_subj_results(root_path):
    filenames = os.listdir(root_path)
    aucs = []
    epochs = []
    for fn in filenames:
        with open(os.path.join(root_path, fn), 'rb') as f:
            res = pickle.load(f)
            aucs.append(res['auroc'][0].detach().numpy())
            epochs.append(res['epoch'][0])
    return aucs, epochs


def find_best_param(data_dict):
    performance_dict = {i: np.median(data_dict[i]) for i in data_dict.keys()}
    best_param = max(performance_dict, key=lambda x: performance_dict[x])
    return best_param, performance_dict[best_param]


def plot_dnn_res(root_path, save_path):
    subjects = [i for i in range(1, 26) if i != 10]
    aucs = {}
    epochs = {}
    for subject in subjects:
        path = os.path.join(root_path, str(subject))
        aucs[str(subject)], epochs[str(subject)] = load_dnn_subj_results(path)

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


def plot_dnn_win_res(root_path, w_size, save_path):
    root_path = os.path.join(root_path, "ebg4_sensor_ica_eegnet1d_ebg_w" + str(w_size))
    subjects = [i for i in range(1, 26) if i != 10]
    auc_dict = {}
    epoch_dict = {}
    best_medians_subj = {}
    best_medians_all = []
    for subject in subjects:
        auc_dict[str(subject)] = {}
        epoch_dict[str(subject)] = {}
        tmin = os.listdir(os.path.join(root_path, str(subject)))
        for t in tmin:
            path = os.path.join(root_path, str(subject), t)
            t = "{:.1f}".format(float(t))
            auc_dict[str(subject)][t], epoch_dict[str(subject)][t] = load_dnn_subj_results(path)

        tmin_keys = list(auc_dict[str(subject)].keys())
        tmin_keys = [float(tmin_key) for tmin_key in tmin_keys]
        tmin_keys.sort()

        auc_sorted = {str(i): auc_dict[str(subject)][str(i)] for i in tmin_keys}
        epoch_sorted = {str(i): epoch_dict[str(subject)][str(i)] for i in tmin_keys}
        auc_subj = auc_sorted.values()
        epoch_subj = epoch_sorted.values()
        t_keys = auc_sorted.keys()

        fig, axs = plt.subplots(2, 1, figsize=(40, 15))
        axs[0].boxplot(auc_subj, labels=t_keys)
        axs[0].axhline(y=0.5, color='r', linestyle='--')
        axs[0].set_title(f'Boxplot of AUC Scores for Subject {subject} per Each Tmin')
        axs[0].set_ylabel('AUC Scores')
        axs[0].set_xlabel('Subject ID')
        axs[1].boxplot(epoch_subj, labels=t_keys)
        axs[1].set_title(f'Boxplot of Epochs for Best AUC Scores for Subject {subject} per Each Tmin')
        axs[1].set_ylabel('Epochs')
        axs[1].set_xlabel('Subject ID')
        plt.savefig(os.path.join(save_path, str(w_size), f"box_plot_eegnet1d_s{subject}.png"))
        plt.close(fig)

        best_param, best_performance = find_best_param(auc_sorted)
        print(f"Best Tmin for Subject {subject} with Window Size {w_size} s is {best_param}")
        best_medians_subj[str(subject)] = auc_sorted[best_param]
        best_medians_all.append(best_performance)

    auc_scores = best_medians_subj.values()
    auc_keys = best_medians_subj.keys()
    plt.boxplot(auc_scores, labels=auc_keys)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Boxplot of Best AUC Scores for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, str(w_size), f"box_plot_eegnet1d_all_subjects.png"))
    plt.close()

    plt.boxplot(best_medians_all, labels=["EEGNet1D"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects Window Size {w_size}')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, str(w_size), f"box_plot_eegnet1d_single_plot.png"))
    plt.close()


if __name__ == "__main__":
    task = "plot_dnn_win_res"

    if task == "compare_logreg_c":
        path_to_data = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/grid_search_c"
        path_to_save = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/ebg4_auc_box_plot"
        compare_logreg_c(path_to_data, path_to_save)
    elif task == "plot_dnn_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_ebg"
        path_to_save = "/Volumes/T5 EVO/Smell"
        plot_dnn_res(path_to_data, path_to_save)
    elif task == "plot_dnn_win_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/w_results/"
        plot_dnn_win_res(path_to_data, 0.5, path_to_save)
