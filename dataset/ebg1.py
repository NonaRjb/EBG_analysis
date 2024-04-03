import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import os
import random
import dataset.data_utils as data_utils
from dataset.data_utils import load_ebg1_mat


class EBG1(Dataset):
    def __init__(
            self, root_path: str,
            tmin: float = None,
            tmax: float = None,
            fmin: float = None,
            fmax: float = None,
            binary: bool = True,
            freqs: np.ndarray = None,
            modality: str = 'ebg'
    ):

        self.root_path = root_path
        recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat' for subject_id in range(1, 31) if
                      subject_id != 4]
        indices_to_keep = scio.loadmat(os.path.join(root_path, 'kept_indices_dataset1.mat'))
        indices_to_keep = indices_to_keep['kept_trials']

        self.baseline_min = -0.5
        self.baseline_max = -0.2
        self.init_data = None
        self.labels = None
        self.subject_id = None
        self.fs = None
        self.time_vec = None
        self.class_weight = None
        self.freqs = freqs
        self.modality = modality

        for i, recording in enumerate(recordings):
            file = os.path.join(root_path, recording)
            init_data, label, time_vec, fs = load_ebg1_mat(file, indices_to_keep[0][i])

            if self.fs is None:
                self.fs = fs.astype(float)

            if self.time_vec is None:
                self.time_vec = time_vec

            if modality == 'eeg':
                init_data = init_data[:, :64, :]
            elif modality == 'ebg':
                init_data = init_data[:, 64:, :]
            else:
                pass

            if self.init_data is None:
                self.init_data = init_data
                self.labels = np.expand_dims(label, axis=1)
                self.subject_id = i * np.ones((len(label), 1))
            else:
                self.init_data = np.vstack((self.init_data, init_data))
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

        if fmin is None:
            self.f_min = 0
        else:
            self.f_min = np.abs(self.freqs - fmin).argmin()

        if fmax is None:
            self.f_max = len(self.freqs)
        else:
            self.f_max = np.abs(self.freqs - fmax).argmin()

        self.baseline_min = np.abs(self.time_vec - self.baseline_min).argmin()
        self.baseline_max = np.abs(self.time_vec - self.baseline_max).argmin()
        self.time_vec = self.time_vec[self.t_min:self.t_max]

        if binary:
            new_labels = [1. if label == 40 else 0. for label in self.labels]
            # new_labels = self.labels != 40
            self.labels = new_labels
            # self.class_weight = torch.tensor([
            #     len(new_labels) / (new_labels.count(0.) * 2),
            #     len(new_labels) / (new_labels.count(1.) * 2)
            # ])
            class_0_count = new_labels.count(0.)
            class_1_count = new_labels.count(1.)
            self.class_weight = torch.tensor(class_0_count/class_1_count)

        self.data = self.init_data
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max], axis=(0, -1), keepdims=True)
        self.data = self.data[..., self.t_min:self.t_max] - self.baseline

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = torch.from_numpy(self.data[item, ...])
        label = self.labels[item]
        return sample, label


if __name__ == "__main__":
    data_args = {'tmin': None, 'tmax': None, 'transform': None, 'freqs': np.linspace(20, 100, 160)}
    ebg_dataset = EBG1(root_path='/Users/nonarajabi/Desktop/KTH/Smell/Novel_Bulb_measure/data/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]
