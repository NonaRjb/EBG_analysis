import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon, ranksums, normaltest, mannwhitneyu
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


def compare_models(model1, model2, model1_df, model2_df, save_path):
    channels = list(model1.keys())

    # Merge the dataframes for comparison
    merged_df = pd.merge(model1_df, model2_df, on=['Subject', 'Channel'], suffixes=('_Model1', '_Model2'))

    # Perform paired t-test (Wilcoxon signed-rank test) for each channel
    results = {}
    for channel, group in merged_df.groupby('Channel'):
        res = wilcoxon(group['Mean Performance_Model1'], group['Mean Performance_Model2'])
        results[channel] = {'stat': res.statistic, 'p_value': res.pvalue}

    # Print results
    for channel, result in results.items():
        print(f"Channel: {channel}")
        print(f"  T-statistic: {result['stat']}")
        print(f"  P-value: {result['p_value']}")

    # Prepare data for violin plot
    melted_df = pd.melt(merged_df, id_vars=['Subject', 'Channel'],
                        value_vars=['Mean Performance_Model1', 'Mean Performance_Model2'],
                        var_name='Model', value_name='Mean Performance')

    # Plotting
    plt.figure(figsize=(10, 6))

    # Customize the plot
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.3, color='0.93', zorder=-1)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax = sns.violinplot(x='Channel', y='Mean Performance', hue='Model', data=melted_df, split=False, inner='quart',
                        palette={'Mean Performance_Model1': '#bf812d', 'Mean Performance_Model2': '#35978f'},
                        edgecolor='black', saturation=1, alpha=1, zorder=4)

    ax.set_zorder(50)

    # Add a red line at 0.5
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1, zorder=2)

    model1_patch = mpatches.Patch(color='#bf812d', label='ResNet-1D')
    model2_patch = mpatches.Patch(color='#35978f', label='LogReg')

    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Data Modalities', fontsize=17)
    plt.ylabel('Mean Performance', fontsize=17)
    plt.legend(handles=[model1_patch, model2_patch], loc='upper left', fontsize=14)
    plt.title(f"Performance Across Modalities", pad=10, fontsize=18)

    plt.tight_layout()

    # Save the plot
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "compare_models.svg"))

    plt.show()


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

    def sort_data(data, subjects_sorted):
        return {str(subject): data[str(subject)] for subject in subjects_sorted}

    EBG_data = sort_data(EBG_data, sorted_subjects)
    EEG_data = sort_data(EEG_data, sorted_subjects)
    sniff_data = sort_data(sniff_data, sorted_subjects)
    source_data = sort_data(source_data, sorted_subjects)
    EBG_sniff_data = sort_data(EBG_sniff_data, sorted_subjects)
    EEG_sniff_data = sort_data(EEG_sniff_data, sorted_subjects)
    source_sniff_data = sort_data(source_sniff_data, sorted_subjects)
    EEG_EBG_data = sort_data(EEG_EBG_data, sorted_subjects)
    source_EBG_data = sort_data(source_EBG_data, sorted_subjects)
    EEG_source_data = sort_data(EEG_source_data, sorted_subjects)

    # Calculate means and standard errors
    def calc_mean_and_se(data):
        means = np.array([np.mean(data[subject]) for subject in data.keys()])
        res = ttest_1samp(means, 0.5, alternative='greater')
        print(f"t_stats = {res.statistic}, p_value = {res.pvalue}")
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

    res = ttest_rel(EBG_sniff_means, EBG_means)
    print(f"EBG-Sniff compared to EBG: t = {res.statistic}, p_val = {res.pvalue}")

    res = ttest_rel(EBG_sniff_means, sniff_means)
    print(f"EBG-Sniff compared to Sniff: t = {res.statistic}, p_val = {res.pvalue}\n")

    res = ttest_rel(EEG_sniff_means, EEG_means)
    print(f"EEG-Sniff compared to EEG: t = {res.statistic}, p_val = {res.pvalue}")

    res = ttest_rel(EEG_sniff_means, sniff_means)
    print(f"EEG-Sniff compared to Sniff: t = {res.statistic}, p_val = {res.pvalue}\n")

    res = ttest_rel(source_sniff_means, source_means)
    print(f"Source-Sniff compared to Source: t = {res.statistic}, p_val = {res.pvalue}")

    res = ttest_rel(source_sniff_means, sniff_means)
    print(f"Source-Sniff compared to Sniff: t = {res.statistic}, p_val = {res.pvalue}\n")

    res = ttest_rel(EEG_EBG_means, EBG_means)
    print(f"EEG-EBG compared to EBG: t = {res.statistic}, p_val = {res.pvalue}")

    res = ttest_rel(EEG_EBG_means, EEG_means)
    print(f"EEG-EBG compared to EEG: t = {res.statistic}, p_val = {res.pvalue}\n")

    res = ttest_rel(EEG_source_means, source_means)
    print(f"EEG-Source compared to Source: t = {res.statistic}, p_val = {res.pvalue}")

    res = ttest_rel(EEG_source_means, EEG_means)
    print(f"EEG-Source compared to EEG: t = {res.statistic}, p_val = {res.pvalue}\n")

    res = ttest_rel(source_EBG_means, source_means)
    print(f"EBG-Source compared to Source: t = {res.statistic}, p_val = {res.pvalue}")

    res = ttest_rel(source_EBG_means, EBG_means)
    print(f"EBG-Source compared to EBG: t = {res.statistic}, p_val = {res.pvalue}")

    print("\n\n")
    print(f"EBG: {EBG_mean} (stderr = {EBG_se})")
    print(f"EEG: {EEG_mean} (stderr = {EEG_se})")
    print(f"Sniff: {sniff_mean} (stderr = {sniff_se})")
    print(f"Source: {source_mean} (stderr = {source_se})")
    print(f"EBG-Sniff: {EBG_sniff_mean} (stderr = {EBG_sniff_se})")
    print(f"EEG-Sniff: {EEG_sniff_mean} (stderr = {EEG_sniff_se})")
    print(f"Source-Sniff: {source_sniff_mean} (stderr = {source_sniff_se})")
    print(f"EEG-EBG: {EEG_EBG_mean} (stderr = {EEG_EBG_se})")
    print(f"Source-EBG: {source_EBG_mean} (stderr = {source_EBG_se})")
    print(f"Source-EEG: {EEG_source_mean} (stderr = {EEG_source_se})")

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

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')

    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    bar_zorder = 2
    plt.bar(bar_positions_all[0], means[0] - 0.5, yerr=errors[0], color=colors['ebg-sniff'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[3], means[3] - 0.5, yerr=errors[3], color=colors['eeg-ebg'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[6], means[6] - 0.5, yerr=errors[6], color=colors['source-ebg'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[9], means[9] - 0.5, yerr=errors[9], color=colors['source-eeg'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[12], means[12] - 0.5, yerr=errors[12], color=colors['eeg-sniff'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[15], means[15] - 0.5, yerr=errors[15], color=colors['source-sniff'], alpha=0.7, capsize=3,
            bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[[1, 5, 8]], means[[1, 5, 8]] - 0.5, yerr=errors[[1, 5, 8]],
            color=colors['ebg'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[[2, 14, 17]], means[[2, 14, 17]] - 0.5, yerr=errors[[2, 14, 17]],
            color=colors['sniff'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[[4, 10, 13]], means[[4, 10, 13]] - 0.5, yerr=errors[[4, 10, 13]],
            color=colors['eeg'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')
    plt.bar(bar_positions_all[[7, 11, 16]], means[[7, 11, 16]] - 0.5, yerr=errors[[7, 11, 16]],
            color=colors['source'], alpha=0.7, capsize=3, bottom=0.5, width=0.8, zorder=bar_zorder, edgecolor='black')

    max_height = np.max(means)

    # Customize the plot
    plt.xticks(bar_positions_all, combined_categories, rotation=45, ha='right')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('Data Modalities', fontsize=15)
    plt.ylabel('Mean Performance', fontsize=15)
    plt.ylim((0.47, min(1.0, 1.2 * max_height)))
    plt.title(f'Mean Performance of {model.capitalize()} Across Data Modalities', fontsize=16)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"compare_subjects_multimodal_{model}.svg"))
    plt.show()


def compare_whole_crop(whole_bars, crop_bars, model, save_path):
    channels = list(whole_bars.keys())

    dist_scale = 1.7
    label_positions = np.arange(len(channels)) * dist_scale
    bar_positions = [[i - 0.25, i + 0.25] for i in label_positions]
    bar_positions = [i for x in bar_positions for i in x]

    plt.figure(figsize=(4, 5))

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')

    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    bar_zorder = 2
    plt.bar(bar_positions[0], whole_bars['Scalp-EBG']['mu'] - 0.5, yerr=whole_bars['Scalp-EBG']['sigma'],
            color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[1], crop_bars['Scalp-EBG']['mu'] - 0.5, yerr=crop_bars['Scalp-EBG']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[2], whole_bars['EEG']['mu'] - 0.5, yerr=whole_bars['EEG']['sigma'], color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[3], crop_bars['EEG']['mu'] - 0.5, yerr=crop_bars['EEG']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[4], whole_bars['Sniff']['mu'] - 0.5, yerr=whole_bars['Sniff']['sigma'], color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[5], crop_bars['Sniff']['mu'] - 0.5, yerr=crop_bars['Sniff']['sigma'], color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[6], whole_bars['Source-EBG']['mu'] - 0.5, yerr=whole_bars['Source-EBG']['sigma'],
            color='#bf812d',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)
    plt.bar(bar_positions[7], crop_bars['Source-EBG']['mu'] - 0.5, yerr=crop_bars['Source-EBG']['sigma'],
            color='#35978f',
            alpha=0.7, capsize=3, bottom=0.5, width=0.5, zorder=bar_zorder)

    # find maximum bar height
    max_height = None
    min_height = None
    for key in crop_bars.keys():
        if max_height is None or crop_bars[key]['mu'] > max_height:
            max_height = crop_bars[key]['mu']
        if whole_bars[key]['mu'] > max_height:
            max_height = whole_bars[key]['mu']
        if min_height is None or crop_bars[key]['mu'] - 0.5 < min_height:
            min_height = crop_bars[key]['mu'] - 0.5
        if whole_bars[key]['mu'] - 0.5 < min_height:
            min_height = whole_bars[key]['mu'] - 0.5

    whole_patch = mpatches.Patch(color='#bf812d', label='whole')
    crop_patch = mpatches.Patch(color='#35978f', label='cropped')

    plt.xticks(label_positions, channels, rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Data Channels', fontsize=13)
    plt.ylabel('Mean Performance', fontsize=13)
    plt.ylim(0.45 if model == "logreg" else 0.47, min(1.1 * max_height, 1.0))
    plt.legend(handles=[whole_patch, crop_patch])
    plt.title(f"Performance Across Modalities", pad=10)

    plt.subplots_adjust(left=0.23, right=0.92, top=0.92, bottom=0.1)

    if path_to_save is not None:
        plt.savefig(os.path.join(save_path, f"compare_whole_cropped_{model}.svg"))

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

    deg_f = 51
    # Calculate mean and standard deviation for each data channel across subjects
    EBG_means = np.array([np.mean(EBG_data[subject]) for subject in EBG_data.keys()])
    print("normaltest EBG = ", normaltest(EBG_means))
    res = ttest_1samp(EBG_means, 0.5, alternative='greater')
    t_statistic, p_value = res.statistic, res.pvalue
    print(f"EBG: t({deg_f}) = {t_statistic}, p_value = {p_value}")
    EBG_mean = EBG_means.mean()
    EBG_std = EBG_means.std() / np.sqrt(EBG_means.shape[0])

    EEG_means = np.array([np.mean(EEG_data[subject]) for subject in EEG_data.keys()])
    print("normaltest EEG = ", normaltest(EEG_means))
    res = ttest_1samp(EEG_means, 0.5, alternative='greater')
    t_statistic, p_value = res.statistic, res.pvalue
    print(f"EEG: t({deg_f}) = {t_statistic}, p_value = {p_value}")
    EEG_mean = EEG_means.mean()
    EEG_std = EEG_means.std() / np.sqrt(EEG_means.shape[0])

    sniff_means = np.array([np.mean(sniff_data[subject]) for subject in sniff_data.keys()])
    print("normaltest Sniff = ", normaltest(sniff_means))
    res = ttest_1samp(sniff_means, 0.5, alternative='greater')
    t_statistic, p_value = res.statistic, res.pvalue
    print(f"Sniff: t({deg_f}) = {t_statistic}, p_value = {p_value}")
    sniff_mean = sniff_means.mean()
    sniff_std = sniff_means.std() / np.sqrt(sniff_means.shape[0])

    source_means = np.array([np.mean(source_data[subject]) for subject in source_data.keys()])
    print("normaltest Source = ", normaltest(source_means))
    res = ttest_1samp(source_means, 0.5, alternative='greater')
    t_statistic, p_value = res.statistic, res.pvalue
    print(f"Source: t({deg_f}) = {t_statistic}, p_value = {p_value}")
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
    channels = ['Scalp-EBG', 'EEG', 'Sniff', 'Source-EBG']
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

    plt.figure(figsize=(8, 7))

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Add bar plot with error bars centered at 0.5
    plt.bar(x_positions, bar_heights, yerr=standard_error, color='grey', alpha=0.5, capsize=5, bottom=0.5, width=0.5,
            zorder=2)
    # Plot scatter plot with jitter
    sns.scatterplot(data=df, x='Jittered Channel', y='Mean Performance', edgecolor='black', facecolor='none', s=50,
                    alpha=0.7, linewidth=1.5, zorder=3)

    # Color circles for selected subjects
    c = sns.color_palette("tab10", len(selected_subjects))  # Generate unique c for selected subjects
    for i, subject_id in enumerate(selected_subjects):
        mask = df['Subject'] == subject_id
        selected_data = df[mask]
        plt.scatter(selected_data['Jittered Channel'], selected_data['Mean Performance'], color=c[i], edgecolor='black',
                    linewidth=1.5, s=100, zorder=3)

    # Add bar plot with error bars centered at 0.5
    # plt.bar(x_positions, bar_heights, yerr=standard_error, color='grey', alpha=0.5, capsize=5, bottom=0.5, width=0.5)

    # Customize the plot
    plt.xticks(x_positions, channels)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('Data Channels', fontsize=18)
    plt.ylabel('Mean Performance', fontsize=18)
    plt.ylim(df['Mean Performance'].min(), df['Mean Performance'].max())
    # plt.ylim(0.5, 1)
    plt.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')  # Adding a reference line at y=0.5
    plt.title(f'Mean Performance of {model.capitalize()} Across Data Channels', fontsize=18, pad=10)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.1)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"compare_subjects_{model}.svg"))
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
            best_scores[subject] = np.load(os.path.join(root_path, subject, filename))

    with open(os.path.join(save_path, f"scores_subject_{modality}.pkl"), 'wb') as f:
        pickle.dump(best_scores, f)

    horizontal_bar_subject(best_scores, modality, "logreg", save_path)
    mean_score = np.mean(np.asarray([best_scores[str(subject)] for subject in subjects]))
    print("Median test AUC = ", mean_score)

    return


def compare_logreg_c(root_path, save_path, modality, file_type='npy'):
    subjects = os.listdir(root_path)
    best_scores = {}

    for subject in subjects:

        for filename in os.listdir(os.path.join(root_path, subject)):
            if file_type == "npy":
                best_scores[subject] = np.load(os.path.join(root_path, subject, filename))
            else:
                with open(os.path.join(root_path, subject, filename), 'rb') as f:
                    best_scores[subject] = pickle.load(f)

    with open(os.path.join(save_path, f"scores_subject_{modality}.pkl"), 'wb') as f:
        pickle.dump(best_scores, f)

    if file_type == "npy":
        horizontal_bar_subject(best_scores, modality, "logreg", save_path)
        mean_score = np.mean(np.asarray([np.mean(best_scores[str(subject)]) for subject in subjects]))
        print("Average test AUC = ", mean_score)
        print("Std Error = ",
              np.std(np.asarray([np.mean(best_scores[str(subject)]) for subject in subjects]))/np.sqrt(len(subjects)))
    else:
        val_mean_score = np.mean(np.asarray([np.mean(best_scores[str(subject)]['val']) for subject in subjects]))
        val_std_err = np.std(np.asarray([np.mean(best_scores[str(subject)]['val']) for subject in subjects])) \
                      / np.sqrt(len(best_scores.keys()))
        test_mean_score = np.mean(np.asarray([np.mean(best_scores[str(subject)]['test']) for subject in subjects]))
        test_std_err = np.std(np.asarray([np.mean(best_scores[str(subject)]['test']) for subject in subjects])) \
                       / np.sqrt(len(best_scores.keys()))

        print(f"Val Average AUC = {val_mean_score} (stderr = {val_std_err})")
        print(f"Test Average AUC = {test_mean_score} (stderr = {test_std_err})")

    return


def load_dnn_subj_results(root_path):
    filenames = os.listdir(root_path)

    with open(os.path.join(root_path, filenames[0]), 'rb') as f:
        res = pickle.load(f)
        aucs = res['auroc']
        # aucs = [auc.detach().numpy() for auc in aucs]
        epochs = res['epoch']
        if 'test_auroc' in res.keys():
            test_aucs = res['test_auroc']
        else:
            test_aucs = res['auroc']
    return aucs, epochs, test_aucs


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
    test_aucs = {}
    epochs = {}
    metrics = []
    test_metrics = []
    scores = {}
    for subject in subjects:
        path = os.path.join(root_path, str(subject))
        aucs[str(subject)], epochs[str(subject)], test_aucs[str(subject)] = load_dnn_subj_results(path)
        # zero_idx = [i for i in range(len(aucs[str(subject)])) if aucs[str(subject)][i] == 0]
        # print(zero_idx)
        # aucs[str(subject)] = [auc for auc in aucs[str(subject)] if auc != 0]
        # epochs[str(subject)] = [e for i, e in enumerate(epochs[str(subject)]) if i not in zero_idx]
        metrics.append(np.mean(aucs[str(subject)]))
        test_metrics.append(np.mean(test_aucs[str(subject)]))
        scores[str(subject)] = aucs[str(subject)]

    with open(os.path.join(save_path, f"scores_subjects_eegnet1d_{modality}.pkl"), 'wb') as f:
        pickle.dump(scores, f)

    horizontal_bar_subject(scores, modality, "eegnet1d", save_path)

    score_vals = scores.values()
    score_keys = scores.keys()
    plt.figure(figsize=(40, 6))
    plt.boxplot(score_vals, labels=score_keys)
    plt.grid(axis='y', color='0.95')
    plt.title('Boxplot of AUC Scores for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('AUC')
    plt.savefig(
        os.path.join(save_path, f"box_plot_resnet1d_{modality}.svg"))

    epoch_vals = epochs.values()
    epoch_keys = epochs.keys()
    plt.figure(figsize=(40, 6))
    plt.boxplot(epoch_vals, labels=epoch_keys)
    plt.grid(axis='y', color='0.95')
    plt.title('Boxplot of Epochs with Best AUC Score for Each Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('Epoch')
    plt.savefig(
        os.path.join(save_path, f"epoch_box_plot_resnet1d_{modality}.svg"))

    plt.figure()
    plt.boxplot(metrics, labels=['EEGNet-1D'])
    plt.grid(axis='y', color='0.95')
    plt.title('Boxplot of AUC Scores for EEGNet-1D with Normalized EBG')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.savefig(
        os.path.join(save_path, f"auc_box_plot_resnet1d_{modality}_all.svg"))
    print(f"Average of best performances: {np.mean(metrics)} (stderr = {np.std(metrics) / np.sqrt(len(metrics))})")
    print(f"Average of best performances: {np.mean(test_metrics)} "
          f"(stderr = {np.std(test_metrics) / np.sqrt(len(test_metrics))})")
    print(f"Median AUCs = {np.mean(np.asarray([np.median(scores[str(subject)]) for subject in subjects]))}")
    np.save(os.path.join(save_path, f"eegnet1d_ebg4_whole_{modality}.npy"), metrics)


def compare_within_subject(df, selected_subjects, save_path, model_name):
    # Filter the DataFrame based on selected subjects
    df_filtered = df[df['Subject'].isin(selected_subjects)]

    channel_colors = {
        "EEG": '#bf812d',
        "Scalp-EBG": '#35978f',
        "Sniff": '#dfc27d',
        "Source-EBG": '#f6e8c3'
    }

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0, color='0.93')

    # Set major grid locator
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    # Create the bar plot
    sns.barplot(x='Subject', y='Mean Performance', hue='Channel',
                data=df_filtered, palette=channel_colors, edgecolor='black', width=0.5, zorder=2)

    # Add title and labels
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.title(f'Performances for Selected Subjects ({model_name})', fontsize=18, pad=10)
    plt.xlabel('Subject', fontsize=18)
    plt.ylabel('Mean Performance', fontsize=18)

    plt.ylim(0.0, 1.0)

    # Adjust the legend
    plt.legend(loc='upper left', fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.12)
    # Save the plot
    if path_to_save is not None:
        plt.savefig(os.path.join(save_path, f"compare_within_subj_{model_name}.svg"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    task = "plot_dnn_res"

    if task == "compare_logreg_c":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/architectures/ebg4_ebg_rf_pca/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/architectures/ebg4_ebg_rf_pca_plots/"
        compare_logreg_c(path_to_data, path_to_save, "ebg")
    elif task == "compare_logreg_c_tmin":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_eeg_logreg_freq_lim/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_eeg_logreg_freq_lim_plots/"
        compare_logreg_c_tmin(path_to_data, path_to_save, "eeg")
    elif task == "plot_dnn_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_eegnet1d_ebg/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_eegnet1d_ebg_plots/"
        plot_dnn_res(path_to_data, path_to_save, "ebg")
    elif task == "plot_dnn_win_res":
        path_to_data = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win/ebg4_resnet1d_ebg/"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win/ebg4_resnet1d_ebg_plots/"
        plot_dnn_res(path_to_data, path_to_save, "ebg")
    elif task == "subject_modality_scatter_logreg":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_eeg_logreg_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_ebg_logreg_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_source_logreg_plots/" \
                         "scores_subject_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c"
        # path_to_save = None
        bars_whole, df_whole = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                           "logreg", [4, 10, 13, 35])
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_eeg_logreg_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/ebg4_ebg_logreg_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/sniff_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin/" \
                         "ebg4_source_logreg_plots/scores_subject_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c_tmin"
        # path_to_save = None
        bars_crop, df_crop = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                         "logreg", [4, 10, 13, 35])
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/"
        # path_to_save = None
        compare_whole_crop(bars_whole, bars_crop, "logreg", path_to_save)
    elif task == "subject_modality_scatter_eegnet1d":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_eeg_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_ebg_plots/" \
                      "scores_subjects_eegnet1d_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_sniff_plots/" \
                        "scores_subjects_eegnet1d_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_source_plots/" \
                         "scores_subjects_eegnet1d_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/"
        # path_to_save = None
        bars_whole, df_whole = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                           "resnet1d", [14, 37, 9, 45])
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win/ebg4_resnet1d_eeg_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win/ebg4_resnet1d_ebg_plots/" \
                      "scores_subjects_eegnet1d_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win/ebg4_resnet1d_sniff_plots/" \
                        "scores_subjects_eegnet1d_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win/ebg4_resnet1d_source_plots/" \
                         "scores_subjects_eegnet1d_source.pkl"
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/cropped_win"
        # path_to_save = None
        bars_crop, df_crop = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                         "resnet1d", [14, 37, 9, 45])
        path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/"
        # path_to_save = None
        compare_whole_crop(bars_whole, bars_crop, "resnet1d", path_to_save)
    elif task == "subject_multimodal_scatter_logreg":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_eeg_logreg_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_ebg_logreg_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_source_logreg_plots/" \
                         "scores_subject_source.pkl"
        path_to_ebg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_ebg_sniff_plots/" \
                            "scores_subject_ebg-sniff.pkl"
        path_to_eeg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_eeg_sniff_plots/" \
                            "scores_subject_eeg-sniff.pkl"
        path_to_source_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_source_sniff_plots/" \
                               "scores_subject_source-sniff.pkl"
        path_to_eeg_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_ebg_eeg_plots/" \
                          "scores_subject_eeg-ebg.pkl"
        path_to_source_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_ebg_source_plots/" \
                             "scores_subject_source-ebg.pkl"
        path_to_eeg_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_logreg_eeg_source_plots/" \
                             "scores_subject_source-eeg.pkl"
        # path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg"
        path_to_save = None
        scatterplot_multimodal(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_ebg_sniff,
                               path_to_eeg_sniff, path_to_source_sniff, path_to_eeg_ebg, path_to_source_ebg,
                               path_to_eeg_source, path_to_save, "logreg")

    elif task == "subject_multimodal_scatter_eegnet1d":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_eeg_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_ebg_plots/" \
                      "scores_subjects_eegnet1d_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_sniff_plots/" \
                        "scores_subjects_eegnet1d_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_source_plots/" \
                         "scores_subjects_eegnet1d_source.pkl"
        # path_to_ebg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win_multimodal/" \
        #                     "ebg4_multimodal_ebg-sniff_plots/scores_subjects_eegnet1d_ebg-sniff.pkl"
        # path_to_eeg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win_multimodal/" \
        #                     "ebg4_multimodal_eeg-sniff_plots/" \
        #                     "scores_subjects_eegnet1d_eeg-sniff.pkl"
        # path_to_source_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win_multimodal/" \
        #                        "ebg4_multimodal_source-sniff_plots/" \
        #                        "scores_subjects_eegnet1d_source-sniff.pkl"
        # path_to_eeg_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win_multimodal/ebg4_multimodal_eeg-ebg_plots/" \
        #                   "scores_subjects_eegnet1d_eeg-ebg.pkl"
        # path_to_source_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win_multimodal" \
        #                      "/ebg4_multimodal_source-ebg_plots/" \
        #                      "scores_subjects_eegnet1d_source-ebg.pkl"
        # path_to_eeg_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win_multimodal/" \
        #                      "ebg4_multimodal_source-eeg_plots/" \
        #                      "scores_subjects_eegnet1d_source-eeg.pkl"
        path_to_ebg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/" \
                            "ebg4_resnet1d_ebg-sniff_plots/scores_subjects_eegnet1d_ebg-sniff.pkl"
        path_to_eeg_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_eeg-sniff_plots/" \
                            "scores_subjects_eegnet1d_eeg-sniff.pkl"
        path_to_source_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_source-sniff_plots/" \
                               "scores_subjects_eegnet1d_source-sniff.pkl"
        path_to_eeg_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_eeg-ebg_plots/" \
                          "scores_subjects_eegnet1d_eeg-ebg.pkl"
        path_to_source_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_source-ebg_plots/" \
                             "scores_subjects_eegnet1d_source-ebg.pkl"
        path_to_eeg_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_source-eeg_plots/" \
                             "scores_subjects_eegnet1d_source-eeg.pkl"
        # path_to_save = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn"
        path_to_save = None
        scatterplot_multimodal(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_ebg_sniff,
                               path_to_eeg_sniff, path_to_source_sniff, path_to_eeg_ebg, path_to_source_ebg,
                               path_to_eeg_source, path_to_save, "resnet1d")

    elif task == "compare_models":
        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_eeg_plots/" \
                      "scores_subjects_eegnet1d_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_ebg_plots/" \
                      "scores_subjects_eegnet1d_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_sniff_plots/" \
                        "scores_subjects_eegnet1d_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_dnn/whole_win/ebg4_resnet1d_source_plots/" \
                         "scores_subjects_eegnet1d_source.pkl"
        path_to_save = None
        bars_dnn, df_dnn = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                       "resnet1d", [14, 37, 9, 45])

        path_to_eeg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_eeg_logreg_plots/" \
                      "scores_subject_eeg.pkl"
        path_to_ebg = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_ebg_logreg_plots/" \
                      "scores_subject_ebg.pkl"
        path_to_sniff = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/sniff_plots/" \
                        "scores_subject_sniff.pkl"
        path_to_source = "/Volumes/T5 EVO/Smell/plots/ebg4_logreg/grid_search_c/ebg4_source_logreg_plots/" \
                         "scores_subject_source.pkl"
        path_to_save = None
        bars_logreg, df_logreg = scatterplot(path_to_ebg, path_to_eeg, path_to_sniff, path_to_source, path_to_save,
                                             "logreg", [4, 10, 13, 35])

        path_to_save = "/Volumes/T5 EVO/Smell/plots/"
        # path_to_save = None
        compare_models(bars_dnn, bars_logreg, df_dnn, df_logreg, save_path=path_to_save)

        compare_within_subject(df_dnn, [22, 14, 38, 45, 49], path_to_save, "ResNet-1D")
        compare_within_subject(df_logreg, [22, 14, 38, 45, 49], path_to_save, "LogReg")
