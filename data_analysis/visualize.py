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






if __name__ == "__main__":
    task = "compare_logreg_c"

    if task == "compare_logreg_c":
        path_to_data = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/grid_search_c"
        path_to_save = "/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/ebg4_auc_box_plot"
        compare_logreg_c(path_to_data, path_to_save)
