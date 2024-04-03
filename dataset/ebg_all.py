import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import Dataset
import os
from dataset.data_utils import load_ebg4, load_ebg1_mat


def add_ebg4(root_path, modality, tmin, tmax, baseline_min, baseline_max, binary):
    root_path = os.path.join(root_path, "ebg4")
    Fs = None
    T_vec = None
    init_data = None
    labels = None
    subject_id = None
    subjects = [subject_id for subject_id in range(1, 26) if subject_id != 10]
    for i, subject in enumerate(subjects):
        source_data, label, time_vec, fs = load_ebg4(root_path, subject, data_type="sensor_ica")

        if Fs is None:
            Fs = float(fs)

        if T_vec is None:
            T_vec = time_vec

        if modality == 'eeg':
            source_data = source_data[:, :64, :]
        elif modality == 'ebg':
            source_data = source_data[:, 64:, :]
        else:
            pass

        if init_data is None:
            init_data = source_data
            labels = np.expand_dims(label, axis=1)
            subject_id = i * np.ones((len(label), 1))
        else:
            init_data = np.vstack((init_data, source_data))
            labels = np.vstack((labels, np.expand_dims(label, axis=1)))
            subject_id = np.vstack((subject_id, i * np.ones((len(label), 1))))

    t_min = np.abs(T_vec - tmin).argmin()
    t_max = np.abs(T_vec - tmax).argmin()

    baseline_min = np.abs(T_vec - baseline_min).argmin()
    baseline_max = np.abs(T_vec - baseline_max).argmin()
    T_vec = T_vec[t_min:t_max]

    if binary:
        new_labels = [1. if y == 64 else 0. for y in labels]
        labels = new_labels
        class_0_count = new_labels.count(0.)
        class_1_count = new_labels.count(1.)
        print(f"EBG 4: N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")

    data = init_data
    baseline = np.mean(data[..., baseline_min:baseline_max], axis=(0, -1), keepdims=True)
    data = data[..., t_min:t_max] - baseline

    return data, np.array(labels), T_vec


def add_ebg1(root_path, modality, tmin, tmax, baseline_min, baseline_max, binary):
    root_path = os.path.join(root_path, "ebg1")
    recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat' for subject_id in range(1, 31) if
                  subject_id != 4]
    indices_to_keep = scio.loadmat(os.path.join(root_path, 'kept_indices_dataset1.mat'))
    indices_to_keep = indices_to_keep['kept_trials']

    Fs = None
    T_vec = None
    init_data = None
    labels = None
    subject_id = None
    for i, recording in enumerate(recordings):
        file = os.path.join(root_path, recording)
        source_data, label, time_vec, fs = load_ebg1_mat(file, indices_to_keep[0][i])

        if Fs is None:
            Fs = fs.astype(float)

        if T_vec is None:
            T_vec = time_vec

        if modality == 'eeg':
            source_data = source_data[:, :64, :]
        elif modality == 'ebg':
            source_data = source_data[:, 64:, :]
        else:
            pass

        if init_data is None:
            init_data = source_data
            labels = np.expand_dims(label, axis=1)
            subject_id = i * np.ones((len(label), 1))
        else:
            init_data = np.vstack((init_data, source_data))
            labels = np.vstack((labels, np.expand_dims(label, axis=1)))
            subject_id = np.vstack((subject_id, i * np.ones((len(label), 1))))

    t_min = np.abs(T_vec - tmin).argmin()
    t_max = np.abs(T_vec - tmax).argmin()

    baseline_min = np.abs(T_vec - baseline_min).argmin()
    baseline_max = np.abs(T_vec - baseline_max).argmin()
    T_vec = T_vec[t_min:t_max]

    if binary:
        new_labels = [1. if y == 40 else 0. for y in labels]
        labels = new_labels
        class_0_count = new_labels.count(0.)
        class_1_count = new_labels.count(1.)
        print(f"EBG 1: N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")

    data = init_data
    baseline = np.mean(data[..., baseline_min:baseline_max], axis=(0, -1), keepdims=True)
    data = data[..., t_min:t_max] - baseline

    return data, np.array(labels), T_vec


class EBG_all(Dataset):
    def __init__(
            self,
            root_path: str,
            tmin: float = None,
            tmax: float = None,
            binary: bool = True,
            modality: str = 'both'
    ):
        self.root_path = root_path

        self.baseline_min = -0.5
        self.baseline_max = -0.2
        self.data = None
        self.labels = None
        self.subject_id = None
        self.fs = 512
        self.time_vec = None
        self.class_weight = None

        # ebg1
        ebg1_data, ebg1_labels, ebg1_tvec = add_ebg1(
            root_path=root_path, modality=modality, tmin=tmin, tmax=tmax, baseline_min=self.baseline_min,
            baseline_max=self.baseline_max, binary=binary)
        # ebg4
        ebg4_data, ebg4_labels, ebg4_tvec = add_ebg4(
            root_path=root_path, modality=modality, tmin=tmin, tmax=tmax, baseline_min=self.baseline_min,
            baseline_max=self.baseline_max, binary=binary)

        assert ebg4_data.shape[-1] == ebg1_data.shape[-1]
        assert ebg4_data.shape[-2] == ebg1_data.shape[-2]

        self.data = np.vstack((ebg1_data, ebg4_data))
        self.labels = np.vstack((np.expand_dims(ebg1_labels, axis=1), np.expand_dims(ebg4_labels, axis=1)))
        self.labels = self.labels.squeeze()
        self.time_vec = ebg1_tvec

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = torch.from_numpy(self.data[item, ...])
        return sample, self.labels[item]


if __name__ == "__main__":
    path = "/Volumes/T5 EVO/Smell/"
    dataset = EBG_all(root_path=path, tmin=-0.2, tmax=0.3, binary=True, modality="ebg")
