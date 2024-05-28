import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.ticker as ticker
import os
import re

from collections import Counter

colors = {
    "eeg": '#bf812d',
    "ebg": '#35978f',
    "sniff": '#dfc27d',
    "source": '#f6e8c3',
    "eeg-ebg": "#8c510a",
    "ebg-sniff": "#59a89c",
    "eeg-sniff": "#01665e",
    "source-sniff": "#c7eae5",
    "source-eeg": "#543005",
    "source-ebg": "#003c30",
}

np.random.seed(29)


def scatterplot_multimodal(
        ebg_file, eeg_file, sniff_file, source_file,
        ebg_sniff_file, eeg_sniff_file, source_sniff_file,
        eeg_ebg_file, source_ebg_file, eeg_source_file,
        save_path, model):
    # Load data from pickle files
    def load_data(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    EBG_data = load_data(ebg_file)
    EEG_data = load_data(eeg_file)
    sniff_data = load_data(sniff_file)
    source_data = load_data(source_file)

    EBG_sniff_data = load_data(ebg_sniff_file)
    EEG_sniff_data = load_data(eeg_sniff_file)
    source_sniff_data = load_data(source_sniff_file)
    EEG_EBG_data = load_data(eeg_ebg_file)
    source_EBG_data = load_data(source_ebg_file)
    EEG_source_data = load_data(eeg_source_file)

    subjects = [int(k) for k in EBG_data.keys()]
    sorted_subjects = sorted(subjects)

    subjects_source = [int(k) for k in source_sniff_data.keys()]
    sorted_subjects_source = sorted(subjects_source)

    def sort_data(data, subjects_sorted):
        return {str(subject): data[str(subject)] for subject in subjects_sorted}

    EBG_data = sort_data(EBG_data, sorted_subjects)
    EEG_data = sort_data(EEG_data, sorted_subjects)
    sniff_data = sort_data(sniff_data, sorted_subjects)
    source_data = sort_data(source_data, sorted_subjects)
    EBG_sniff_data = sort_data(EBG_sniff_data, sorted_subjects)
    EEG_sniff_data = sort_data(EEG_sniff_data, sorted_subjects)
    source_sniff_data = sort_data(source_sniff_data, sorted_subjects_source)
    EEG_EBG_data = sort_data(EEG_EBG_data, sorted_subjects)
    source_EBG_data = sort_data(source_EBG_data, sorted_subjects_source)
    EEG_source_data = sort_data(EEG_source_data, sorted_subjects_source)

    # Calculate means and standard errors
    def calc_mean_and_se(data):
        means = np.array([np.mean(data[subject]) for subject in data.keys()])
        mean = means.mean()
        # mean = np.median(means)
        se = means.std() / np.sqrt(means.shape[0])
        return means, mean, se

    EBG_means, EBG_mean, EBG_se = calc_mean_and_se(EBG_data)
    EEG_means, EEG_mean, EEG_se = calc_mean_and_se(EEG_data)
    sniff_means, sniff_mean, sniff_se = calc_mean_and_se(sniff_data)
    source_means, source_mean, source_se = calc_mean_and_se(source_data)
    EBG_sniff_means, EBG_sniff_mean, EBG_sniff_se = calc_mean_and_se(EBG_sniff_data)
    EEG_sniff_means, EEG_sniff_mean, EEG_sniff_se = calc_mean_and_se(EEG_sniff_data)
    source_sniff_means, source_sniff_mean, source_sniff_se = calc_mean_and_se(source_sniff_data)
    EEG_EBG_means, EEG_EBG_mean, EEG_EBG_se = calc_mean_and_se(EEG_EBG_data)
    source_EBG_means, source_EBG_mean, source_EBG_se = calc_mean_and_se(source_EBG_data)
    EEG_source_means, EEG_source_mean, EEG_source_se = calc_mean_and_se(EEG_source_data)

    # Data for plotting
    means = np.array([
        EBG_sniff_mean, EBG_mean, sniff_mean,
        EEG_EBG_mean, EEG_mean, EBG_mean,
        source_EBG_mean, source_mean, EBG_mean,
        EEG_source_mean, EEG_mean, source_mean,
        EEG_sniff_mean, EEG_mean, sniff_mean,
        source_sniff_mean, source_mean, sniff_mean
    ])
    errors = np.array([
        EBG_sniff_se, EBG_se, sniff_se,
        EEG_EBG_se, EEG_se, EBG_se,
        source_EBG_se, source_se, EBG_se,
        EEG_source_se, EEG_se, source_se,
        EEG_sniff_se, EEG_se, sniff_se,
        source_sniff_se, source_se, sniff_se
    ])

    # Define the categories and their positions
    combined_categories = [
        'EBG-Sniff', 'EBG', 'Sniff',
        'EEG-EBG', 'EEG', 'EBG',
        'Source-EBG', 'Source', 'EBG',
        'EEG-Source', 'EEG', 'Source',
        'EEG-Sniff', 'EEG', 'Sniff',
        'Source-Sniff', 'Source', 'Sniff'
    ]

    plt.figure(figsize=(16, 8))

    # Add bar plot with error bars centered at 0.5
    dist_scale = 1.5
    bar_positions = np.arange(0, len(combined_categories), 3) * dist_scale
    bar_positions_all = [[dist_scale * i, dist_scale * i + 0.8, dist_scale * i + 1.6] for i in
                         np.arange(0, len(combined_categories), 3)]
    bar_positions_all = np.array([i for x in bar_positions_all for i in x])

    plt.figure(figsize=(10, 6))

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0)

    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    plt.bar(bar_positions_all[0], means[0] - 0.5, yerr=errors[0], color=colors['ebg-sniff'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[3], means[3] - 0.5, yerr=errors[3], color=colors['eeg-ebg'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[6], means[6] - 0.5, yerr=errors[6], color=colors['source-ebg'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[9], means[9] - 0.5, yerr=errors[9], color=colors['source-eeg'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[12], means[12] - 0.5, yerr=errors[12], color=colors['eeg-sniff'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[15], means[15] - 0.5, yerr=errors[15], color=colors['source-sniff'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[[1, 5, 8]], means[[1, 5, 8]] - 0.5, yerr=errors[[1, 5, 8]],
            color=colors['ebg'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[[2, 14, 17]], means[[2, 14, 17]] - 0.5, yerr=errors[[2, 14, 17]],
            color=colors['sniff'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[[4, 10, 13]], means[[4, 10, 13]] - 0.5, yerr=errors[[4, 10, 13]],
            color=colors['eeg'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=1, edgecolor='black')
    plt.bar(bar_positions_all[[7, 11, 16]], means[[7, 11, 16]] - 0.5, yerr=errors[[7, 11, 16]],
            color=colors['source'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=1, edgecolor='black')

    # Customize the plot
    plt.xticks(bar_positions_all, combined_categories, rotation=45, ha='right')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('Data Modalities', fontsize=15)
    plt.ylabel('Mean Performance', fontsize=15)
    plt.ylim((0.5, 1.0))
    plt.title(f'Mean Performance of {model.capitalize()} Across Data Modalities', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"compare_subjects_multimodal_{model}.svg"))
    plt.show()


def compare_whole_crop(whole_bars, crop_bars, whole_scatter, crop_scatter, model, save_path):
    channels = list(whole_bars.keys())

    dist_scale = 1.7
    label_positions = np.arange(len(channels)) * dist_scale
    bar_positions = [[i - 0.25, i + 0.25] for i in label_positions]
    bar_positions = [i for x in bar_positions for i in x]

    plt.figure(figsize=(3, 4))

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')

    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    plt.bar(bar_positions[0], whole_bars['EBG']['mu'] - 0.5, yerr=whole_bars['EBG']['sigma'], color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[1], crop_bars['EBG']['mu'] - 0.5, yerr=crop_bars['EBG']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[2], whole_bars['EEG']['mu'] - 0.5, yerr=whole_bars['EEG']['sigma'], color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[3], crop_bars['EEG']['mu'] - 0.5, yerr=crop_bars['EEG']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[4], whole_bars['Sniff']['mu'] - 0.5, yerr=whole_bars['Sniff']['sigma'], color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[5], crop_bars['Sniff']['mu'] - 0.5, yerr=crop_bars['Sniff']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[6], whole_bars['Source']['mu'] - 0.5, yerr=whole_bars['Source']['sigma'], color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)
    plt.bar(bar_positions[7], crop_bars['Source']['mu'] - 0.5, yerr=crop_bars['Source']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=1)

    # jitter_strength = 0.2
    #
    # x_positions_whole = [bar_positions[i] for i in range(len(bar_positions)) if i % 2 == 0]
    # whole_scatter['Jittered Channel'] = \
    #     whole_scatter['Channel'].apply(lambda x: x_positions_whole[channels.index(x)] +
    #                                   np.random.uniform(-jitter_strength, jitter_strength))
    #
    # x_positions_crop = [bar_positions[i] for i in range(len(bar_positions)) if i % 2 != 0]
    # crop_scatter['Jittered Channel'] = \
    #     crop_scatter['Channel'].apply(lambda x: x_positions_crop[channels.index(x)] +
    #                                              np.random.uniform(-jitter_strength, jitter_strength))
    #
    # # Plot scatter plot with jitter
    # sns.scatterplot(data=whole_scatter, x='Jittered Channel', y='Mean Performance', edgecolor='black',
    #                 facecolor='none', s=50, alpha=0.7, linewidth=1.5, zorder=2)
    # sns.scatterplot(data=crop_scatter, x='Jittered Channel', y='Mean Performance', edgecolor='black',
    #                 facecolor='none', s=50, alpha=0.7, linewidth=1.5, zorder=2)

    # Color circles for selected subjects
    # c = sns.color_palette("magma", len(selected_subjects))  # Generate unique c for selected subjects
    # for i, subject_id in enumerate(selected_subjects):
    #     mask = whole_scatter['Subject'] == subject_id
    #     selected_data = whole_scatter[mask]
    #     plt.scatter(selected_data['Jittered Channel'], selected_data['Mean Performance'], color=c[i], edgecolor='black',
    #                 linewidth=1.5, s=50, zorder=3)
    #
    #     mask = crop_scatter['Subject'] == subject_id
    #     selected_data = crop_scatter[mask]
    #     plt.scatter(selected_data['Jittered Channel'], selected_data['Mean Performance'], color=c[i], edgecolor='black',
    #                 linewidth=1.5, s=50, zorder=3)

    whole_patch = mpatches.Patch(color='#bf812d', label='whole')
    crop_patch = mpatches.Patch(color='#35978f', label='cropped')

    plt.xticks(label_positions, channels)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.xlabel('Data Channels', fontsize=11)
    plt.ylabel('Mean Performance', fontsize=11)
    plt.ylim((0.5, 1.0))
    plt.legend(handles=[whole_patch, crop_patch])
    plt.title(f"Performance Across Modalities", pad=10)

    plt.subplots_adjust(left=0.25, right=0.87, top=0.92, bottom=0.12)

    plt.savefig(os.path.join(save_path, f"compare_whole_cropped_{model}.pdf"))

    plt.show()


def scatterplot(ebg_file, eeg_file, sniff_file, source_file, save_path, model, selected_subjects):
    # Load data from pickle file
    with open(ebg_file, 'rb') as file:
        EBG_data = pickle.load(file)
    with open(eeg_file, 'rb') as file:
        EEG_data = pickle.load(file)
    with open(sniff_file, 'rb') as file:
        sniff_data = pickle.load(file)
    with open(source_file, 'rb') as file:
        source_data = pickle.load(file)
    subjects = [int(k) for k in EBG_data.keys()]
    sorted_subjects = sorted(subjects)
    EBG_data = {str(subject): EBG_data[str(subject)] for subject in sorted_subjects}
    EEG_data = {str(subject): EEG_data[str(subject)] for subject in sorted_subjects}
    sniff_data = {str(subject): sniff_data[str(subject)] for subject in sorted_subjects}
    source_data = {str(subject): source_data[str(subject)] for subject in sorted_subjects}

    n_subjects = len(EBG_data.keys())

    # Calculate mean and standard deviation for each data channel across subjects
    EBG_means = np.array([np.mean(EBG_data[subject]) for subject in EBG_data.keys()])
    EBG_mean = EBG_means.mean()
    EBG_std = EBG_means.std() / np.sqrt(EBG_means.shape[0])

    EEG_means = np.array([np.mean(EEG_data[subject]) for subject in EEG_data.keys()])
    EEG_mean = EEG_means.mean()
    EEG_std = EEG_means.std() / np.sqrt(EEG_means.shape[0])

    sniff_means = np.array([np.mean(sniff_data[subject]) for subject in sniff_data.keys()])
    sniff_mean = sniff_means.mean()
    sniff_std = sniff_means.std() / np.sqrt(sniff_means.shape[0])

    source_means = np.array([np.mean(source_data[subject]) for subject in source_data.keys()])
    source_mean = source_means.mean()
    source_std = source_means.std() / np.sqrt(source_means.shape[0])

    # Calculate overall mean and standard error for each channel
    mean_performance = np.concatenate((
        np.expand_dims(EBG_means, axis=1),
        np.expand_dims(EEG_means, axis=1),
        np.expand_dims(sniff_means, axis=1),
        np.expand_dims(source_means, axis=1)
    ), axis=1)
    overall_mean = np.array([EBG_mean, EEG_mean, sniff_mean, source_mean])
    standard_error = np.array([EBG_std, EEG_std, sniff_std, source_std])

    # Calculate the height of the bars (difference from 0.5)
    bar_heights = overall_mean - 0.5

    # Create a DataFrame for Seaborn
    channels = ['EBG', 'EEG', 'Sniff', 'Source']
    x_positions = np.arange(len(channels))  # [0, 1, 2, 3]

    data = {
        'Subject': np.repeat(np.arange(n_subjects), len(channels)),
        'Channel': np.tile(channels, n_subjects),
        'Mean Performance': mean_performance.flatten()
    }
    df = pd.DataFrame(data)

    # Add jitter to avoid overlap
    jitter_strength = 0.2
    df['Jittered Channel'] = df['Channel'].apply(lambda x: x_positions[channels.index(x)] +
                                                           np.random.uniform(-jitter_strength, jitter_strength))

    plt.figure(figsize=(8, 6))

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Add bar plot with error bars centered at 0.5
    plt.bar(x_positions, bar_heights, yerr=standard_error, color='grey', alpha=0.5, capsize=5, bottom=0.5, width=0.5,
            zorder=1)
    # Plot scatter plot with jitter
    sns.scatterplot(data=df, x='Jittered Channel', y='Mean Performance', edgecolor='black', facecolor='none', s=50,
                    alpha=0.7, linewidth=1.5, zorder=2)

    # Color circles for selected subjects
    c = sns.color_palette("magma", len(selected_subjects))  # Generate unique c for selected subjects
    for i, subject_id in enumerate(selected_subjects):
        mask = df['Subject'] == subject_id
        selected_data = df[mask]
        plt.scatter(selected_data['Jittered Channel'], selected_data['Mean Performance'], color=c[i], edgecolor='black',
                    linewidth=1.5, s=50, zorder=3)

    # Add bar plot with error bars centered at 0.5
    # plt.bar(x_positions, bar_heights, yerr=standard_error, color='grey', alpha=0.5, capsize=5, bottom=0.5, width=0.5)

    # Customize the plot
    plt.xticks(x_positions, channels)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Data Channels', fontsize=16)
    plt.ylabel('Mean Performance', fontsize=16)
    plt.ylim((0.5, 1.0))
    plt.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')  # Adding a reference line at y=0.5
    plt.title(f'Mean Performance of {model.capitalize()} Across Data Channels', fontsize=17)

    plt.subplots_adjust(left=0.09, right=0.95, top=0.95, bottom=0.1)

    plt.savefig(os.path.join(save_path, f"compare_subjects_{model}.pdf"))
    plt.show()

    bars = {ch: {'mu': mu, 'sigma': sigma} for ch, mu, sigma in zip(channels, overall_mean, standard_error)}
    df = pd.DataFrame(data)
    return bars, df


def horizontal_bar_subject(scores, modality, model, save_path):
    # Calculate mean performance and standard deviation for each subject
    mean_performances = {subject: np.mean(performances) for subject, performances in scores.items()}
    std_performances = {subject: np.std(performances) / np.sqrt(len(performances)) for subject, performances in
                        scores.items()}

    # Sort subjects based on mean performance
    sorted_subjects = sorted(mean_performances, key=lambda s: (mean_performances[s], int(s)))
    sorted_mean_performances = [mean_performances[subject] for subject in sorted_subjects]
    sorted_std_performances = [std_performances[subject] for subject in sorted_subjects]

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 10))  # Adjust size as needed
    y = np.arange(len(scores))
    y_spaced = np.linspace(0, len(scores) * 1.2, len(scores))
    bars = ax.barh(y_spaced, sorted_mean_performances, xerr=sorted_std_performances, capsize=2, color=colors[modality],
                   zorder=3)
    ax.axvline(x=0.5, color='red', linestyle='--', zorder=4)
    ax.set_yticks(y_spaced)
    ax.set_yticklabels(sorted_subjects)
    ax.set_xlabel('AUC-ROC')
    ax.set_ylabel('Subject')
    # Add gridlines every 0.1 unit
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(which='major', axis='x', linestyle='--', color='0.92', linewidth=0.5, zorder=0)
    plt.legend([bars[0], plt.Line2D([0], [0], color='red', linestyle='--')], ['Performance', 'Chance'],
               loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_title(f'Average AUC by Subject ({modality.upper()})')
    plt.tight_layout()  # Adjust layout to prevent clippirng of labels
    plt.savefig(os.path.join(save_path, f"auc_bar_plots_{model}_{modality}.pdf"))
    plt.close()

    np.save(os.path.join(save_path, f"{model}_ebg4_whole_{modality}.npy"), sorted_mean_performances)

    return


def compare_logreg_c_tmin(root_path, save_path, modality):
    # os.makedirs(os.path.join(save_path, "w" + str(w_size)), exist_ok=True)

    pattern = r"c(\d+(\.\d+)?)_t(-?\d+(\.\d+)?)\.npy"

    best_c = {}
    best_tmin = {}
    best_scores = {}
    metrics = []
    subjects = os.listdir(root_path)
    subjects = [int(s) for s in subjects]
    subjects.sort()
    subjects = [str(s) for s in subjects]
    for subject in subjects:

        scores = {}
        for filename in os.listdir(os.path.join(root_path, subject)):
            select_t = re.match(pattern, filename)
            if float(select_t.group(3)) == -0.6:
                continue
            scores[filename] = np.load(os.path.join(root_path, subject, filename))
            scores[filename] = [i for i in scores[filename] if i != 0]

        best_param_subj, best_metric_subj = find_best_param(scores, metric="mean")
        match = re.match(pattern, best_param_subj)
        if match:
            # Extract X and Y values from the matched groups
            c = float(match.group(1))
            tmin = float(match.group(3))
            # Append to the lists
            best_c[subject] = c
            best_tmin[subject] = tmin
        best_scores[subject] = scores[best_param_subj]
        metrics.append(best_metric_subj)
        print(f"Subject {subject}: tmin={best_tmin[subject]}, auc={best_metric_subj}")

    with open(os.path.join(save_path, f"scores_subject_{modality}.pkl"), 'wb') as f:
        pickle.dump(best_scores, f)

    horizontal_bar_subject(best_scores, modality, "logreg", save_path)

    plt.boxplot(metrics, labels=["Logistic Regression"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, f"box_plot_logreg_single_plot_{modality}.png"))
    plt.close()

    print(f"Median of Best Medians is: {np.median(metrics)}")

    tmins = best_tmin.values()
    plt.figure()
    plt.boxplot(tmins, labels=['EBG4'])
    plt.title("Boxplot of Best Tmin Values for Logistic Regression")
    plt.xlabel("Dataset")
    plt.ylabel("Tmin Values")
    plt.savefig(os.path.join(save_path, f"box_plot_logreg_tmins_{modality}.png"))
    plt.close()

    print(f"Median of Best Tmins is: {np.median(list(tmins))}")

    # Use Counter to count the occurrences of each unique value
    counter = Counter(list(tmins))

    # Convert to dictionary if you need
    count_dict = dict(counter)

    print(count_dict)
    # np.save(os.path.join(save_path, "logreg_" + ".npy"), np.asarray(metrics))
    # np.save(os.path.join(save_path, "logreg_t_" + ".npy"), np.asarray(list(tmins)))

    return


def compare_logreg_c(root_path, save_path, modality):
    if modality == "sniff":
        pattern = r'c([\d.]+)\.npy'
    elif "-" in modality:
        pattern = r'([\d.]+)\_([\d.]+)\.npy'
    else:
        pattern = r'([\d.]+)\.npy'

    subjects = os.listdir(root_path)
    best_scores = {}
    best_params = {}
    best_metric = []
    for subject in subjects:

        scores = {}
        for filename in os.listdir(os.path.join(root_path, subject)):
            match = re.match(pattern, filename)
            if match:
                # c = match.group(1)
                c = filename.split(".n")[0]
                scores[c] = np.load(os.path.join(root_path, subject, filename))
                scores[c] = [i for i in scores[c] if i != 0]
        best_param_subj, best_metric_subj = find_best_param(scores, metric="mean")
        best_scores[subject] = scores[best_param_subj]
        best_params[subject] = best_param_subj
        print(f"Subject {subject}: maximum auc median = {best_metric_subj}, C = {best_param_subj}")
        best_metric.append(best_metric_subj)
    with open(os.path.join(save_path, f"scores_subject_{modality}.pkl"), 'wb') as f:
        pickle.dump(best_scores, f)

    horizontal_bar_subject(best_scores, modality, "logreg", save_path)

    plt.boxplot(best_metric, labels=["Logistic Regression"])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Boxplot of Best AUC Scores for All Subjects')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.savefig(os.path.join(save_path, f"box_plot_logreg_single_plot_{modality}.png"))
    plt.close()

    print(f"Median of Best Average is: {np.median(best_metric)}")

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


def find_best_param(data_dict, metric="median"):
    if metric == "median":
        performance_dict = {i: np.median(data_dict[i]) for i in data_dict.keys()}
    elif metric == "mean":
        performance_dict = {i: np.mean(data_dict[i]) for i in data_dict.keys()}
    else:
        raise NotImplementedError
    best_param = max(performance_dict, key=lambda x: performance_dict[x])
    return best_param, performance_dict[best_param]


def plot_dnn_res(root_path, save_path, modality):

    subjects = os.listdir(root_path)
    subjects = [int(s) for s in subjects]
    subjects.sort()
    aucs = {}
    epochs = {}
    metrics = []
    scores = {}
    for subject in subjects:
        path = os.path.join(root_path, str(subject))
        aucs[str(subject)], epochs[str(subject)] = load_dnn_subj_results(path)
        zero_idx = [i for i in range(len(aucs[str(subject)])) if aucs[str(subject)][i] == 0]
        aucs[str(subject)] = [auc for auc in aucs[str(subject)] if auc != 0]
        epochs[str(subject)] = [e for i, e in enumerate(epochs[str(subject)]) if i not in zero_idx]
        metrics.append(np.mean(aucs[str(subject)]))
        scores[str(subject)] = aucs[str(subject)]

    with open(os.path.join(save_path, f"scores_subjects_eegnet1d_{modality}.pkl"), 'wb') as f:
        pickle.dump(scores, f)

    horizontal_bar_subject(scores, modality, "eegnet1d", save_path)

    epoch_vals = epochs.values()
    epoch_keys = epochs.keys()
    plt.figure(figsize=(40, 6))
    plt.boxplot(epoch_vals, labels=epoch_keys)
    plt.grid(axis='y', color='0.95')
    plt.title('Boxplot of Epochs with Best AUC Score for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('Epoch')
    plt.savefig(
        os.path.join(save_path, f"epoch_box_plot_eegnet1d_{modality}.png"))

    plt.figure()
    plt.boxplot(metrics, labels=['EEGNet-1D'])
    plt.grid(axis='y', color='0.95')
    plt.title('Boxplot of AUC Scores for EEGNet-1D with Normalized EBG')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.savefig(
        os.path.join(save_path, f"auc_box_plot_eegnet1d_{modality}_all.png"))
    print(f"median of best performances: {np.median(metrics)}")
    np.save(os.path.join(save_path, f"eegnet1d_ebg4_whole_{modality}.npy"), metrics)


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


def compare_modalities(ebg_path, eeg_path, sniff_path, source_path, save_path, model):
    EBG_data = np.load(ebg_path)
    EEG_data = np.load(eeg_path)
    sniff_data = np.load(sniff_path)
    source_data = np.load(source_path)
    # Calculate mean and standard deviation for each data modality
    EBG_mean = np.mean(EBG_data)
    EEG_mean = np.mean(EEG_data)
    sniff_mean = np.mean(sniff_data)
    source_mean = np.mean(source_data)

    EBG_std = np.std(EBG_data) / np.sqrt(len(EBG_data))
    EEG_std = np.std(EEG_data) / np.sqrt(len(EEG_data))
    sniff_std = np.std(sniff_data) / np.sqrt(len(sniff_data))
    source_std = np.std(source_data) / np.sqrt(len(source_data))

    # Plotting
    labels = ['EBG', 'EEG', 'Sniff', 'Source']
    means = [EBG_mean, EEG_mean, sniff_mean, source_mean]
    stds = [EBG_std, EEG_std, sniff_std, source_std]
    colors_ = [colors['ebg'], colors['eeg'], colors['sniff'], colors['source']]

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors_, width=0.5)

    ax.set_ylabel('Average AUC Score')
    ax.set_title('Average AUC Scores Over Subjects')
    ax.set_ylim(0, max(means) * 1.2)  # Adjust y-axis limit for better visualization

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(12, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig(os.path.join(save_path, f"compare_{model}_whole.pdf"))
    plt.show()


def compare_subjects(ebg_file, eeg_file, sniff_file, save_path, model):
    # Load data from pickle file
    with open(ebg_file, 'rb') as file:
        EBG_data = pickle.load(file)
    with open(eeg_file, 'rb') as file:
        EEG_data = pickle.load(file)
    with open(sniff_file, 'rb') as file:
        sniff_data = pickle.load(file)

    # Sort subjects and extract performances for each data channel
    subjects = [int(k) for k in EBG_data.keys()]
    sorted_subjects = sorted(subjects)
    EBG_data = {str(subject): EBG_data[str(subject)] for subject in sorted_subjects}
    EEG_data = {str(subject): EEG_data[str(subject)] for subject in sorted_subjects}
    sniff_data = {str(subject): sniff_data[str(subject)] for subject in sorted_subjects}

    # Calculate mean and standard deviation for each data channel across subjects
    EBG_mean = np.array([np.mean(EBG_data[subject]) for subject in EBG_data.keys()])
    EBG_std = np.array([np.std(EBG_data[subject]) / np.sqrt(len(EBG_data[subject])) for subject in EBG_data.keys()])

    EEG_mean = np.array([np.mean(EEG_data[subject]) for subject in EEG_data.keys()])
    EEG_std = np.array([np.std(EEG_data[subject]) / np.sqrt(len(EEG_data[subject])) for subject in EEG_data.keys()])

    sniff_mean = np.array([np.mean(sniff_data[subject]) for subject in sniff_data.keys()])
    sniff_std = np.array(
        [np.std(sniff_data[subject]) / np.sqrt(len(sniff_data[subject])) for subject in sniff_data.keys()])

    # Plotting
    x = np.arange(len(sorted_subjects)) * 1.5
    width = 0.3  # Width of the bars

    fig, ax = plt.subplots(figsize=(25, 8))
    bars1 = ax.bar(x - width, EBG_mean, yerr=EBG_std, capsize=5, width=width, label='EBG', color=colors['ebg'])
    bars2 = ax.bar(x, EEG_mean, yerr=EEG_std, capsize=5, width=width, label='EEG', color=colors['eeg'])
    bars3 = ax.bar(x + width, sniff_mean, yerr=sniff_std, capsize=5, width=width, label='Sniff', color=colors['sniff'])
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')

    ax.set_ylabel('Mean Performance')
    ax.set_title('Mean Performance Comparison by Data Channel')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_subjects, rotation=90)
    ax.legend()
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(save_path, f"compare_subjects_whole_{model}.pdf"))
    plt.show()


if __name__ == "__main__":
    task = "compare_logreg_c"

    if task == "compare_logreg_c":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_plots/"
        compare_logreg_c(path_to_data, path_to_save, "sniff")
    elif task == "plot_logreg_win_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_tmin/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/w_results/"
        plot_logreg_win_res(path_to_data, 0.1, path_to_save)
    elif task == "plot_dnn_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_source_eegnet1d_source-sniff/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_source_eegnet1d_source-sniff_plots/"
        plot_dnn_res(path_to_data, path_to_save, "source-sniff")
    elif task == "plot_dnn_win_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/w_results/source_data"
        plot_dnn_win_res(path_to_data, 0.5, path_to_save)
    elif task == "compare_models":
        path_to_save = "/Volumes/T5 EVO/Smell/plots/compare_models"
        compare_models(path_to_save)
    elif task == "compare_logreg_c_tmin":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_eeg_logratio_-1.0_-0.6/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_eeg_logratio_-1.0_-0.6_plots/"
        compare_logreg_c_tmin(path_to_data, path_to_save, "eeg")
    elif task == "compare_logreg_c_sniff":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/sniff_-0.5_1.5"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/sniff_-0.5_1.5_plots"
        compare_logreg_c(path_to_data, path_to_save)
    elif task == "compare_logreg_models":
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_ebg_bl_-1.0_-0.6_plots/" \
                      "logreg_ebg4_whole_ebg.npy"
        # path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/ebg4_ebg_resampled_-0.5_1.5_plots/logreg_ebg4_whole_ebg.npy"
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_eeg_bl_-1.0_-0.6_plots/" \
                      "logreg_ebg4_whole_eeg.npy"
        # path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/ebg4_eeg_resampled_-0.8_1.7_plots/logreg_ebg4_whole_eeg.npy"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_-0.5_1.5_plots/logreg_ebg4_whole_sniff.npy"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_source_plots/" \
                         "logreg_ebg4_whole_source.npy"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/"
        compare_modalities(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save, "logreg")
    elif task == "compare_eegnet1d_models":
        path_to_ebg = "//Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_eegnet1d_ebg_plots/eegnet1d_ebg4_whole_ebg.npy"
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_eegnet1d_eeg_plots/eegnet1d_ebg4_whole_eeg.npy"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_eegnet1d_sniff_plots/eegnet1d_ebg4_whole_sniff.npy"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/"
        compare_modalities(path_to_ebg, path_to_eeg, path_to_sniff, path_to_save, "eegnet1d")
    elif task == "compare_subjects_logreg":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/ebg4_eeg_resampled_-0.5_1.5_12features_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/ebg4_ebg_resampled_-0.5_1.5_12features_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/sniff_-0.5_1.5_plots/scores_subject_sniff.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg"
        compare_subjects(path_to_ebg, path_to_eeg, path_to_sniff, path_to_save, model="logreg")
    elif task == "compare_subjects_eegnet1d":
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_eegnet1d_ebg_plots/scores_subjects_eegnet1d_ebg.pkl"
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_eegnet1d_eeg_plots/scores_subjects_eegnet1d_eeg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_eegnet1d_sniff_plots/scores_subjects_eegnet1d_sniff.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/"
        compare_subjects(path_to_ebg, path_to_eeg, path_to_sniff, path_to_save, model="eegnet1d")
    elif task == "subject_modality_scatter_logreg":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_eeg_bl_-1.0_-0.6_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_ebg_bl_-1.0_-0.6_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_-0.5_1.5_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_source_plots/" \
                         "scores_subject_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c"
        bars_whole, df_whole = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                           "logreg", [4, 10, 13, 35])
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_eeg_logratio_-1.0_-0.6_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_ebg_logratio_-1.0_-0.6_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/sniff_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/" \
                         "ebg4_source_logratio_-1.0_-0.6_plots/scores_subject_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin"
        bars_crop, df_crop = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                         "logreg", [4, 10, 13, 35])
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/"
        compare_whole_crop(bars_whole, bars_crop, df_whole, df_crop, "logreg", path_to_save)
    elif task == "subject_modality_scatter_eegnet1d":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_eeg_bl_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_eeg_bl_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_sniff_95p_bl_plots/" \
                        "scores_subjects_eegnet1d_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_source_eegnet1d_plots/" \
                         "scores_subjects_eegnet1d_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn"
        scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save, "eegnet1d", [14, 37, 9, 45])
    elif task == "subject_multimodal_scatter_logreg":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_eeg_bl_-1.0_-0.6_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_ebg_bl_-1.0_-0.6_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_-0.5_1.5_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_source_plots/" \
                         "scores_subject_source.pkl"
        path_to_ebg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_ebg_sniff_plots/" \
                            "scores_subject_ebg-sniff.pkl"
        path_to_eeg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_eeg_sniff_plots/" \
                            "scores_subject_eeg-sniff.pkl"
        path_to_source_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_source_sniff_plots/" \
                               "scores_subject_source-sniff.pkl"
        path_to_eeg_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_eeg_ebg_plots/" \
                          "scores_subject_eeg-ebg.pkl"
        path_to_source_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_source_ebg_plots/" \
                             "scores_subject_source-ebg.pkl"
        path_to_eeg_source = path_to_ebg_sniff
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg"
        scatterplot_multimodal(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_ebg_sniff,
                               path_to_eeg_sniff, path_to_source_sniff, path_to_eeg_ebg, path_to_source_ebg,
                               path_to_eeg_source, path_to_save, "logreg")

    elif task == "subject_multimodal_scatter_eegnet1d":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_eeg_bl_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_ebg_bl_plots/" \
                      "scores_subjects_eegnet1d_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_sniff_95p_bl_plots/" \
                        "scores_subjects_eegnet1d_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_source_eegnet1d_plots/" \
                         "scores_subjects_eegnet1d_source.pkl"
        path_to_ebg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/" \
                            "ebg4_sensor_ica_eegnet1d_ebg-sniff_sniff_95p_bl_plots/" \
                            "scores_subjects_eegnet1d_ebg-sniff.pkl"
        path_to_eeg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_eeg-sniff_plots/" \
                            "scores_subjects_eegnet1d_eeg-sniff.pkl"
        path_to_source_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_source_eegnet1d_source-sniff_plots/" \
                               "scores_subjects_eegnet1d_source-sniff.pkl"
        path_to_eeg_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/ebg4_sensor_ica_eegnet1d_eeg-ebg_plots/" \
                          "scores_subjects_eegnet1d_eeg-ebg.pkl"
        path_to_source_ebg = path_to_ebg_sniff
        path_to_eeg_source = path_to_ebg_sniff
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn"
        scatterplot_multimodal(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_ebg_sniff,
                               path_to_eeg_sniff, path_to_source_sniff, path_to_eeg_ebg, path_to_source_ebg,
                               path_to_eeg_source, path_to_save, "eegnet1d")
