import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import os
import random
import dataset.data_utils as data_utils
from dataset.data_utils import load_source_data_recent


class SourceData(Dataset):
    def __init__(
            self, root_path: str,
            tmin: float = None,
            tmax: float = None,
            binary: bool = True,
            shuffle_labels: bool = False,
            seed: int = 42
    ):

        self.root_path = root_path
        subjects = [subject_id for subject_id in range(1, 26) if subject_id != 10]

        self.baseline_min = -0.5
        self.baseline_max = -0.2
        self.source_data = None
        self.labels = None
        self.subject_id = None
        self.fs = None
        self.time_vec = None
        self.class_weight = None

        for i, subject in enumerate(subjects):
            file = os.path.join(root_path, str(subject), "source_data.mat")
            source_data, label, time_vec, fs = load_source_data_recent(file)

            if self.fs is None:
                self.fs = float(fs)

            if self.time_vec is None:
                self.time_vec = time_vec

            if self.source_data is None:
                self.source_data = source_data
                self.labels = np.expand_dims(label, axis=1)
                self.subject_id = i * np.ones((len(label), 1))
            else:
                self.source_data = np.vstack((self.source_data, source_data))
                self.labels = np.vstack((self.labels, np.expand_dims(label, axis=1)))
                self.subject_id = np.vstack((self.subject_id, i * np.ones((len(label), 1))))

        if tmin is None:
            self.t_min = 0
        else:
            self.t_min = np.abs(self.time_vec - tmin).argmin()

        if tmax is None:
            self.t_max = len(self.time_vec)
        else:
            self.t_max = np.abs(self.time_vec - tmax).argmin()

        self.baseline_min = np.abs(self.time_vec - self.baseline_min).argmin()
        self.baseline_max = np.abs(self.time_vec - self.baseline_max).argmin()
        self.time_vec = self.time_vec[self.t_min:self.t_max]

        if binary:
            new_labels = [1. if y == 64 else 0. for y in self.labels]
            self.labels = new_labels
            # self.class_weight = torch.tensor([
            #     len(new_labels) / (new_labels.count(0.) * 2),
            #     len(new_labels) / (new_labels.count(1.) * 2)
            # ])
            class_0_count = new_labels.count(0.)
            class_1_count = new_labels.count(1.)
            self.class_weight = torch.tensor(class_0_count/class_1_count)

        self.data = self.source_data
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max], axis=(0, -1), keepdims=True)
        self.data = self.data[..., self.t_min:self.t_max] - self.baseline

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = self.data[item, ...]
        return sample, self.labels[item]


if __name__ == "__main__":
    data_args = {'tmin': None, 'tmax': None}
    ebg_dataset = SourceData(root_path='/Volumes/T5 EVO/Odor_Intensity/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]