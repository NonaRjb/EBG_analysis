import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import re


def compare_logreg_c_tmin(root_path, w_size, save_path):

    os.makedirs(os.path.join(save_path, "w" + str(w_size)), exist_ok=True)

    pattern = r"c(\d+(\.\d+)?)_t(-?\d+(\.\d+)?)\.npy"

    best_c = {}
    best_tmin = {}
    best_scores = {}
    medians = []
    subjects = os.listdir(root_path)
    for subject in subjects:

        scores = {}
        for filename in os.listdir(os.path.join(root_path, subject)):

            scores[filename] = np.load(os.path.join(root_path, subject, filename))

        best_param_subj, best_median_subj = find_best_param(scores)
        match = re.match(pattern, best_param_subj)
        if match:
            # Extract X and Y values from the matched groups
            c = float(match.group(1))
            tmin = float(match.group(3))
            # Append to the lists
            best_c[subject] = c
            best_tmin[subject] = tmin
        best_scores[subject] = scores[best_param_subj]
        medians.append(best_median_subj)

    # sort dict
    auc_keys = list(best_scores.keys())
    auc_keys = [int(k) for k in auc_keys]
    auc_keys.sort()
    sorted_best_scores = {str(k): best_scores[str(k)] for k in auc_keys}
    auc_scores = sorted_best_scores.values()

    plt.figure(figsize=(15, 6))
    plt.boxplot(auc_scores, labels=auc_keys)
    plt.title(f'Boxplot of Best AUC Scores for All Subjects')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC Score')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"auc_box_plots_logreg_c_tmin.png"))
    plt.close()

    plt.boxplot(medians, labels=["Logistic Regression"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects Window Size {w_size}')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_logreg_single_plot.png"))
    plt.close()

    print(f"Median of Best Medians is: {np.median(medians)}")

    tmins = best_tmin.values()
    plt.figure()
    plt.boxplot(tmins, labels=['EBG4'])
    plt.title("Boxplot of Best Tmin Values for Logistic Regression")
    plt.xlabel("Dataset")
    plt.ylabel("Tmin Values")
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_logreg_tmins.png"))
    plt.close()

    print(f"Median of Best Tmins is: {np.median(list(tmins))}")

    np.save(os.path.join(save_path, "w" + str(w_size), "logreg_w" + str(w_size) + ".npy"), np.asarray(medians))
    np.save(os.path.join(save_path, "w" + str(w_size), "logreg_t_w" + str(w_size) + ".npy"), np.asarray(list(tmins)))

    return


def compare_logreg_c(root_path, save_path):
    # pattern = r's\d+_c([\d.]+)\.npy'
    pattern = r'c([\d.]+)\.npy'

    subjects = os.listdir(root_path)
    best_scores = {}
    medians = []
    for subject in subjects:

        scores = {}
        for filename in os.listdir(os.path.join(root_path, subject)):
            match = re.match(pattern, filename)
            if match:
                c = match.group(1)
                scores[c] = np.load(os.path.join(root_path, subject, filename))
        best_param_subj, best_median_subj = find_best_param(scores)
        best_scores[subject] = scores[best_param_subj]
        print(f"Subject {subject}: maximum auc median = {best_median_subj}, C = {best_param_subj}")
        medians.append(best_median_subj)

        # sort dict
        # c_values = list(scores.keys())
        # c_values.sort()
        # sorted_scores = {i: scores[i] for i in c_values}
        # auc_scores = sorted_scores.values()
        #
        # plt.figure(figsize=(15, 6))
        # plt.boxplot(auc_scores, labels=c_values)
        # plt.title(f'Boxplot of AUC Scores for Subject {subject}')
        # plt.xlabel('C Values')
        # plt.ylabel('AUC Score')
        # plt.axhline(y=0.5, color='r', linestyle='--')
        # plt.savefig(os.path.join(save_path, f"s{subject}_c_auc_box_plots.png"))
        # plt.close()
        # sort dict
    auc_keys = list(best_scores.keys())
    auc_keys = [int(k) for k in auc_keys]
    auc_keys.sort()
    sorted_best_scores = {str(k): best_scores[str(k)] for k in auc_keys}
    auc_scores = sorted_best_scores.values()

    plt.figure(figsize=(20, 6))
    plt.boxplot(auc_scores, labels=auc_keys)
    plt.title(f'Boxplot of Best AUC Scores for All Subjects')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC Score')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.savefig(os.path.join(save_path, f"auc_box_plots_logreg_c.png"))
    plt.close()

    plt.boxplot(medians, labels=["Logistic Regression"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, f"box_plot_logreg_single_plot.png"))
    plt.close()

    print(f"Median of Best Medians is: {np.median(medians)}")

    return


def plot_logreg_win_res(root_path, w_size, save_path):
    subjects = [i for i in range(1, 26) if i != 10]
    aucs_subj = {}
    best_medians_subj = {}
    best_medians_all = []
    best_tmins = []
    for subject in subjects:
        path = os.path.join(root_path, str(subject))
        tmin_files = os.listdir(path)
        aucs_subj[str(subject)] = {}
        for f in tmin_files:
            t = f[:-4]
            aucs_subj[str(subject)][t] = np.load(os.path.join(path, f))

        tmin_keys = list(aucs_subj[str(subject)].keys())
        tmin_keys = [float(tmin_key) for tmin_key in tmin_keys]
        tmin_keys.sort()

        auc_sorted = {str(i): aucs_subj[str(subject)][str(i)] for i in tmin_keys}
        auc_subj = auc_sorted.values()
        t_keys = auc_sorted.keys()

        plt.figure(figsize=(40, 6))
        plt.boxplot(auc_subj, labels=t_keys)
        plt.grid(axis='y', color='0.97')
        plt.title('Boxplot of AUC Scores for Each Subject')
        plt.xlabel('Subject ID')
        plt.ylabel('AUC Score')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.savefig(
            os.path.join(save_path, "w" + str(w_size), f"auc_box_plot_logreg_w{w_size}_s{subject}.png"))
        plt.close()

        best_param, best_performance = find_best_param(auc_sorted)
        print(f"Best Tmin for Subject {subject} with Window Size {w_size} s is {best_param}")
        best_medians_subj[str(subject)] = auc_sorted[best_param]
        best_medians_all.append(best_performance)
        best_tmins.append(float(best_param))

    auc_scores = best_medians_subj.values()
    auc_keys = best_medians_subj.keys()
    plt.boxplot(auc_scores, labels=auc_keys)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Boxplot of Best AUC Scores for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_logreg_all_subjects.png"))
    plt.close()

    plt.boxplot(best_medians_all, labels=["Logistic Regression"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects Window Size {w_size}')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_logreg_single_plot.png"))
    plt.close()

    np.save(os.path.join(save_path, "w" + str(w_size), "logreg_w" + str(w_size) + ".npy"), np.asarray(best_medians_all))
    np.save(os.path.join(save_path, "w" + str(w_size), "logreg_t_w" + str(w_size) + ".npy"),
            np.asarray(best_tmins))


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
    # root_path = os.path.join(root_path, "ebg4_sensor_ica_eegnet1d_ebg_w" + str(w_size))
    root_path = os.path.join(root_path, "ebg4_source_eegnet1d_ebg_w" + str(w_size))
    # root_path = os.path.join(root_path, "ebg1_eegnet1d_ebg_w" + str(w_size))

    os.makedirs(os.path.join(save_path, "w" + str(w_size)), exist_ok=True)

    subjects = [i for i in range(1, 26) if i != 10]
    # subjects = [0]
    # subjects = [i for i in range(1, 31) if i != 4]
    auc_dict = {}
    epoch_dict = {}
    best_medians_subj = {}
    best_medians_all = []
    best_tmins = []
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
        plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_eegnet1d_s{subject}.png"))
        plt.close(fig)

        best_param, best_performance = find_best_param(auc_sorted)
        print(f"Best Tmin for Subject {subject} with Window Size {w_size} s is {best_param}")
        best_medians_subj[str(subject)] = auc_sorted[best_param]
        best_medians_all.append(best_performance)
        best_tmins.append(float(best_param))

        # plt.boxplot(auc_sorted[best_param], labels=["EEGNet1D"])
        # plt.title('Boxplot of Best AUC Scores for All Subjects')
        # plt.xlabel("Model")
        # plt.ylabel("AUC Score")
        # plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_eegnet1d_combined_subjects.png"))
        # plt.close()
        #
        # print(f"Best AUC Score: {np.median(auc_sorted[best_param])}")

    auc_scores = best_medians_subj.values()
    auc_keys = best_medians_subj.keys()
    plt.figure(figsize=(15, 6))
    plt.boxplot(auc_scores, labels=auc_keys)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Boxplot of Best AUC Scores for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_eegnet1d_all_subjects.png"))
    plt.close()

    plt.boxplot(best_medians_all, labels=["EEGNet1D"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects Window Size {w_size}')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_eegnet1d_single_plot.png"))
    plt.close()

    print(f"Median of Best Medians is: {np.median(best_medians_all)}")

    plt.boxplot(best_tmins, labels=["EBG1"])
    plt.title("Boxplot of Best Tmin Values for EEGNet1D")
    plt.xlabel("Dataset")
    plt.ylabel("Tmin Values")
    plt.savefig(os.path.join(save_path, "w" + str(w_size), f"box_plot_eegnet1d_tmins.png"))
    plt.close()

    print(f"Median of Best Tmins is: {np.median(best_tmins)}")

    np.save(os.path.join(save_path, "w" + str(w_size), "eegnet1d_w" + str(w_size) + ".npy"),
            np.asarray(best_medians_all))
    np.save(os.path.join(save_path, "w" + str(w_size), "eegnet1d_t_w" + str(w_size) + ".npy"),
            np.asarray(best_tmins))


def compare_models(save_path):
    ebg4_eegnet1d_path = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/w_results/w0.5/"
    ebg1_eegnet1d_path = "/Volumes/T5 EVO/Smell/plots/ebg1_dnn/w_results/w0.5/"
    ebg4_100_logreg_path = "/Volumes/T5 EVO/Smell/plots/grid_search_c_tmin/plots_c_tmin_ebg4/w0.1/"
    ebg1_100_logreg_path = "/Volumes/T5 EVO/Smell/plots/grid_search_c_tmin/plots_c_tmin_ebg1/w0.1/"
    ebg4_500_logreg_path = "/Volumes/T5 EVO/Smell/plots/grid_search_c_tmin/plots_c_tmin_ebg4/w0.5/"
    ebg1_500_logreg_path = "/Volumes/T5 EVO/Smell/plots/grid_search_c_tmin/plots_c_tmin_ebg1/w0.5/"

    # aucs = {
    #     'EEGNet1D': np.load(os.path.join(eegnet1d_path, "eegnet1d_w0.5.npy")),
    #     'LogReg': np.load(os.path.join(logreg_path, "logreg_w0.1.npy"))
    # }
    #
    # tmins = {
    #     'EEGNet1D': np.load(os.path.join(eegnet1d_path, "eegnet1d_t_w0.5.npy")),
    #     'LogReg': np.load(os.path.join(logreg_path, "logreg_t_w0.1.npy"))
    # }
    #
    # plt.boxplot(aucs.values(), labels=aucs.keys())
    # plt.axhline(y=0.5, color='r', linestyle='--')
    # plt.title(f'Compare Best AUC Scores for Each Model')
    # plt.xlabel('Model')
    # plt.ylabel('AUC Score')
    # plt.savefig(os.path.join(save_path, f"box_plot_compare_models.png"))
    # plt.close()
    #
    # plt.boxplot(tmins.values(), labels=tmins.keys())
    # plt.title(f'Compare Best tmins for Each Model')
    # plt.xlabel('Model')
    # plt.ylabel('tmin')
    # plt.savefig(os.path.join(save_path, f"box_plot_compare_tmins.png"))
    # plt.close()

    # Generate some random data for demonstration
    data_model1_dataset1 = np.load(os.path.join(ebg1_eegnet1d_path, "eegnet1d_w0.5.npy"))
    data_model1_dataset2 = np.load(os.path.join(ebg4_eegnet1d_path, "eegnet1d_w0.5.npy"))
    data_model2_dataset1 = np.load(os.path.join(ebg1_100_logreg_path, "logreg_w0.1.npy"))
    data_model2_dataset2 = np.load(os.path.join(ebg4_100_logreg_path, "logreg_w0.1.npy"))
    data_model2_dataset1_2 = np.load(os.path.join(ebg1_500_logreg_path, "logreg_w0.5.npy"))
    data_model2_dataset2_2 = np.load(os.path.join(ebg4_500_logreg_path, "logreg_w0.5.npy"))

    # Create positions for the boxplots
    positions_dataset1 = np.array([1, 2, 3])
    positions_dataset2 = np.array([5, 6, 7])

    # Plot boxplots with different colors
    plt.boxplot([data_model1_dataset1, data_model1_dataset2],
                positions=[positions_dataset1[0], positions_dataset2[0]],
                patch_artist=True,  # fill with color
                notch=False,  # notch shape
                boxprops=dict(facecolor="#F0746E"),  # color of the box
                whiskerprops=dict(color="black"),  # color of whiskers
                capprops=dict(color="black"),  # color of caps
                medianprops=dict(color="black"),  # color of median
                flierprops=dict(marker="o", markersize=5, markerfacecolor="#7C1D6F"))  # color of outliers

    plt.boxplot([data_model2_dataset1, data_model2_dataset2],
                positions=[positions_dataset1[1], positions_dataset2[1]],
                patch_artist=True,  # fill with color
                notch=False,  # notch shape
                boxprops=dict(facecolor="#7CCBA2"),  # color of the box
                whiskerprops=dict(color="black"),  # color of whiskers
                capprops=dict(color="black"),  # color of caps
                medianprops=dict(color="black"),  # color of median
                flierprops=dict(marker="o", markersize=5, markerfacecolor="#7C1D6F"))  # color of outliers

    plt.boxplot([data_model2_dataset1_2, data_model2_dataset2_2],
                positions=[positions_dataset1[2], positions_dataset2[2]],
                patch_artist=True,  # fill with color
                notch=False,  # notch shape
                boxprops=dict(facecolor="#089099"),  # color of the box
                whiskerprops=dict(color="black"),  # color of whiskers
                capprops=dict(color="black"),  # color of caps
                medianprops=dict(color="black"),  # color of median
                flierprops=dict(marker="o", markersize=5, markerfacecolor="#7C1D6F"))  # color of outliers

    model1_patch = mpatches.Patch(color='#F0746E', label='EEGNet-1D')
    model2_patch = mpatches.Patch(color='#7CCBA2', label='LogReg-100ms')
    model2_2_patch = mpatches.Patch(color='#089099', label='LogReg-500ms')

    # Set labels for x-axis
    plt.xticks([positions_dataset1.mean(), positions_dataset2.mean()], ['Dataset 1', 'Dataset 2'])

    # Set title and labels
    plt.title('Boxplots of Model Performance on Different Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Performance')

    # Add legend
    plt.legend(handles=[model1_patch, model2_patch, model2_2_patch])
    plt.grid(axis='y', color='0.92')
    plt.savefig(os.path.join(save_path, f"box_plot_compare_models_all.png"))
    plt.close()
    # Show the plot
    # plt.show()

    data_model1_dataset1 = np.load(os.path.join(ebg1_eegnet1d_path, "eegnet1d_t_w0.5.npy"))
    data_model1_dataset2 = np.load(os.path.join(ebg4_eegnet1d_path, "eegnet1d_t_w0.5.npy"))
    data_model2_dataset1 = np.load(os.path.join(ebg1_100_logreg_path, "logreg_t_w0.1.npy"))
    data_model2_dataset2 = np.load(os.path.join(ebg4_100_logreg_path, "logreg_t_w0.1.npy"))
    data_model2_dataset1_2 = np.load(os.path.join(ebg1_500_logreg_path, "logreg_t_w0.5.npy"))
    data_model2_dataset2_2 = np.load(os.path.join(ebg4_500_logreg_path, "logreg_t_w0.5.npy"))

    # Plot boxplots with different colors
    plt.boxplot([data_model1_dataset1, data_model1_dataset2],
                positions=[positions_dataset1[0], positions_dataset2[0]],
                patch_artist=True,  # fill with color
                notch=False,  # notch shape
                boxprops=dict(facecolor="#F0746E"),  # color of the box
                whiskerprops=dict(color="black"),  # color of whiskers
                capprops=dict(color="black"),  # color of caps
                medianprops=dict(color="black"),  # color of median
                flierprops=dict(marker="o", markersize=5, markerfacecolor="#7C1D6F"))  # color of outliers

    plt.boxplot([data_model2_dataset1, data_model2_dataset2],
                positions=[positions_dataset1[1], positions_dataset2[1]],
                patch_artist=True,  # fill with color
                notch=False,  # notch shape
                boxprops=dict(facecolor="#7CCBA2"),  # color of the box
                whiskerprops=dict(color="black"),  # color of whiskers
                capprops=dict(color="black"),  # color of caps
                medianprops=dict(color="black"),  # color of median
                flierprops=dict(marker="o", markersize=5, markerfacecolor="#7C1D6F"))  # color of outliers

    plt.boxplot([data_model2_dataset1_2, data_model2_dataset2_2],
                positions=[positions_dataset1[2], positions_dataset2[2]],
                patch_artist=True,  # fill with color
                notch=False,  # notch shape
                boxprops=dict(facecolor="#089099"),  # color of the box
                whiskerprops=dict(color="black"),  # color of whiskers
                capprops=dict(color="black"),  # color of caps
                medianprops=dict(color="black"),  # color of median
                flierprops=dict(marker="o", markersize=5, markerfacecolor="#7C1D6F"))  # color of outliers

    model1_patch = mpatches.Patch(color='#F0746E', label='EEGNet-1D')
    model2_patch = mpatches.Patch(color='#7CCBA2', label='LogReg-100ms')
    model2_2_patch = mpatches.Patch(color='#089099', label='LogReg-500ms')

    # Set labels for x-axis
    plt.xticks([positions_dataset1.mean(), positions_dataset2.mean()], ['Dataset 1', 'Dataset 2'])

    # Set title and labels
    plt.title('Boxplots of Best T-Starts for Different Models on Different Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('T-Start')

    # Add legend
    plt.legend(handles=[model1_patch, model2_patch, model2_2_patch])
    plt.grid(axis='y', color='0.92')
    plt.savefig(os.path.join(save_path, f"box_plot_compare_models_tmin_all.png"))
    plt.close()


if __name__ == "__main__":
    task = "compare_logreg_c_sniff"

    if task == "compare_logreg_c":
        path_to_data = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/grid_search_c"
        path_to_save = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/ebg4_auc_box_plot"
        compare_logreg_c(path_to_data, path_to_save)
    elif task == "plot_logreg_win_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_tmin/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/w_results/"
        plot_logreg_win_res(path_to_data, 0.1, path_to_save)
    elif task == "plot_dnn_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_ebg"
        path_to_save = "/Volumes/T5 EVO/Smell"
        plot_dnn_res(path_to_data, path_to_save)
    elif task == "plot_dnn_win_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/w_results/source_data"
        plot_dnn_win_res(path_to_data, 0.5, path_to_save)
    elif task == "compare_models":
        path_to_save = "/Volumes/T5 EVO/Smell/plots/compare_models"
        compare_models(path_to_save)
    elif task == "compare_logreg_c_tmin":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/grid_search_c_tmin/grid_search_c_tmin/ebg1"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/grid_search_c_tmin/plots_c_tmin_ebg1"
        compare_logreg_c_tmin(path_to_data, 0.5, path_to_save)
    elif task == "compare_logreg_c_sniff":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/sniff/grid_search_c"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/sniff/grid_search_c_plots"
        compare_logreg_c(path_to_data, path_to_save)

