import numpy as np
import scipy.io as scio
import pickle
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
            w: float = None,
            binary: bool = True,
            modality: str = 'ebg',
            pick_subjects: int = 0
    ):

        self.root_path = root_path
        if pick_subjects == 0:
            recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat' for subject_id in range(1, 31) if
                          subject_id != 4]
        else:
            recordings = ['SL06_' + str("{:02d}".format(pick_subjects)) + '.mat']

        with open(os.path.join(root_path, 'kept_indices_dataset1.pkl'), 'rb') as f:
            indices_to_keep = pickle.load(f)
        # indices_to_keep = scio.loadmat(os.path.join(root_path, 'kept_indices_dataset1.mat'))
        # indices_to_keep = indices_to_keep['kept_trials']

        self.baseline_min = -0.5
        self.baseline_max = -0.2
        self.init_data = None
        self.labels = None
        self.subject_id = None
        self.fs = None
        self.time_vec = None
        self.class_weight = None
        self.modality = modality

        for i, recording in enumerate(recordings):
            file = os.path.join(root_path, recording)
            init_data, label, time_vec, fs = load_ebg1_mat(file, indices_to_keep[recording])

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

        if w is None:
            if tmax is None:
                self.t_max = len(self.time_vec)
            else:
                self.t_max = np.abs(self.time_vec - tmax).argmin()
        else:
            if tmin is None:
                self.t_max = int(w * self.fs)
            else:
                tmax = tmin + w
                self.t_max = np.abs(self.time_vec - tmax).argmin()
        print(f"first time sample: {self.t_min}, last time sample: {self.t_max}")

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
            print(f"N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")
            self.class_weight = torch.tensor(class_0_count/class_1_count)
        else:
            new_labels = [y/10-1 for y in self.labels]
            self.labels = new_labels
            print(f"new_labels = {set(new_labels)}")

        self.data = self.init_data
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max], axis=(0, -1), keepdims=True)
        self.data = self.data[..., self.t_min:self.t_max] - self.baseline

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = torch.from_numpy(self.data[item, ...])
        return sample, self.labels[item]


if __name__ == "__main__":
    data_args = {'tmin': 0.1, 'tmax': 0.3, 'pick_subjects': 9}
    ebg_dataset = EBG1(root_path='/Volumes/T5 EVO/Smell/ebg1/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]
