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
            transform: str = None,
            freqs: np.ndarray = None,
            include_eeg: bool = False,
            shuffle_labels: bool = False,
            seed: int = 42
    ):

        self.root_path = root_path
        recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat' for subject_id in range(1, 31) if
                      subject_id != 4]
        indices_to_keep = scio.loadmat(os.path.join(root_path, 'kept_indices_dataset1.mat'))
        indices_to_keep = indices_to_keep['kept_trials']

        self.baseline_min = -0.5
        self.baseline_max = -0.2
        self.eeg = None
        self.ebg = None
        self.labels = None
        self.subject_id = None
        self.fs = None
        self.time_vec = None
        self.class_weight = None
        self.transform = transform
        self.freqs = freqs
        self.include_eeg = include_eeg

        for i, recording in enumerate(recordings):
            file = os.path.join(root_path, recording)
            eeg, ebg, label, time_vec, fs = load_ebg1_mat(file, indices_to_keep[0][i])

            if self.fs is None:
                self.fs = fs.astype(float)

            if self.time_vec is None:
                self.time_vec = time_vec

            if self.eeg is None:
                self.eeg = eeg
                self.ebg = ebg
                self.labels = np.expand_dims(label, axis=1)
                self.subject_id = i * np.ones((len(label), 1))
            else:
                self.eeg = np.vstack((self.eeg, eeg))
                self.ebg = np.vstack((self.ebg, ebg))
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
            new_labels = [0. if label == 40 else 1. for label in self.labels]
            self.labels = new_labels
            # self.class_weight = torch.tensor([
            #     len(new_labels) / (new_labels.count(0.) * 2),
            #     len(new_labels) / (new_labels.count(1.) * 2)
            # ])
            class_0_count = new_labels.count(0.)
            class_1_count = new_labels.count(1.)
            self.class_weight = torch.tensor(class_0_count/class_1_count)

        if self.include_eeg:
            self.data = np.concatenate((self.eeg, self.ebg), axis=1)
        else:
            self.data = self.ebg


        # if self.transform == 'tfr_morlet':
        #     self.data = data_utils.apply_tfr(self.data, self.fs, self.freqs, method='dpss')
        #     # self.baseline = np.mean(self.data[:, :, self.f_min:self.f_max, self.baseline_min:self.baseline_max], axis=(0, -1),
        #     #                         keepdims=True)
        #     self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max],
        #                             axis=(0, -1),
        #                             keepdims=True)
        #     self.data = 10 * np.log10(self.data / self.baseline)
        #     np.save('/Midgard/home/nonar/EBG_analysis/tfr1_baseline_corrected.npy')
        #     self.data = self.data[:, :, self.f_min:self.f_max, self.t_min:self.t_max]
        # else:
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max], axis=(0, -1), keepdims=True)
        self.data = self.data[..., self.t_min:self.t_max] - self.baseline
        #
        # if shuffle_labels:
        #     self.labels = random.Random(seed).sample(self.labels, len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = self.data[item, ...]
        # if self.transform == 'tfr_morlet':
        #     # sample = ebg_transforms.apply_tfr_morlet(np.expand_dims(self.sample[item, :, :], axis=0), self.fs, self.freqs)
        #     # sample = 10 * np.log10(np.expand_dims(self.data[item, ...], axis=0) / self.baseline)
        #     # sample = np.mean(sample, axis=1)  # average over channels
        #     sample = np.mean(np.expand_dims(self.data[item, ...], axis=0), axis=1)
        #     # mean = sample.mean(axis=(1, 2), keepdims=True)
        #     # std = sample.std(axis=(1, 2), keepdims=True)
        #     # sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))  # min-max normalization
        #     sample = torch.from_numpy(sample).double()
        # else:
        #     # sample = np.expand_dims(self.data[item, :, self.t_min:self.t_max], axis=0) - self.baseline
        #     sample = np.expand_dims(self.data[item, ...], axis=0)
        #
        #     # sample /= np.max(np.absolute(sample))
        #     # sample = (np.expand_dims(self.sample[item, :, self.t_min:self.t_max], axis=0)
        #     #        - np.mean(self.sample[item, ...], axis=-1, keepdims=True)) / \
        #     #       np.std(self.sample[item, ...], axis=-1, keepdims=True)
        #     sample = torch.from_numpy(sample).double()
        return sample, self.labels[item]


if __name__ == "__main__":
    data_args = {'tmin': None, 'tmax': None, 'transform': None, 'freqs': np.linspace(20, 100, 160)}
    ebg_dataset = EBG1(root_path='/Users/nonarajabi/Desktop/KTH/Smell/Novel_Bulb_measure/data/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]
