import numpy as np
import torch
from torch.utils.data import Dataset
import os
import math
from dataset.data_utils import load_ebg4


class EBG4(Dataset):
    def __init__(
            self, root_path: str,
            tmin: float = None,
            tmax: float = None,
            w: float = None,
            binary: bool = True,
            data_type: str = 'source',
            modality: str = 'both',
            intensity: bool = False,
            pick_subjects: int = 0,
            normalize: bool = False,
            fs_new: int = None
    ):

        self.root_path = root_path
        if pick_subjects == 0:
            # if data_type != 'source':
            #     subjects = [subject_id for subject_id in range(1, 38) if subject_id != 10]
            # else:
            #     subjects = [subject_id for subject_id in range(1, 26) if subject_id != 10]
            subjects = [subject_id for subject_id in range(1, 54) if subject_id != 10]
            print("***** Training On All Available Subject *****")
        else:
            subjects = [pick_subjects]
            print(f"***** Training On Subject {pick_subjects} *****")

        self.baseline_min = -1.0
        self.baseline_max = -0.6
        self.normalize = normalize
        self.source_data = None
        self.labels = None
        self.subject_id = None
        self.fs = None
        self.time_vec = None
        self.class_weight = None
        self.modality = modality

        for i, subject in enumerate(subjects):
            source_data, label, time_vec, fs = load_ebg4(root_path, subject, data_type, fs_new=fs_new)

            if self.fs is None:
                self.fs = float(fs)

            if self.time_vec is None:
                self.time_vec = time_vec

            if data_type == 'sensor' or data_type == 'sensor_ica':
                if modality == 'eeg':
                    source_data = source_data[:, :63, :]
                elif modality == 'eeg-sniff':
                    sniff_data, _, _, _ = load_ebg4(
                        root_path,
                        subject,
                        data_type="sniff",
                        fs_new=fs_new if fs_new is not None else self.fs
                    )
                    sniff_data = np.expand_dims(sniff_data, axis=1)
                    source_data = source_data[:, :63, :]    # extract EEG
                    source_data = np.concatenate((source_data, sniff_data), axis=1)
                elif modality == 'ebg':
                    source_data = source_data[:, 63:-1, :]
                elif modality == 'ebg-sniff':
                    sniff_data, _, _, _ = load_ebg4(
                        root_path,
                        subject,
                        data_type="sniff",
                        fs_new=fs_new if fs_new is not None else self.fs
                    )
                    sniff_data = np.expand_dims(sniff_data, axis=1)
                    source_data = source_data[:, 63:-1, :]    # extract EBG
                    source_data = np.concatenate((source_data, sniff_data), axis=1)
                elif modality == 'both-sniff':
                    sniff_data, _, _, _ = load_ebg4(
                        root_path,
                        subject,
                        data_type="sniff",
                        fs_new=fs_new if fs_new is not None else self.fs
                    )
                    sniff_data = np.expand_dims(sniff_data, axis=1)
                    source_data = np.concatenate((source_data, sniff_data), axis=1)
                elif modality == 'sniff':
                    sniff_data, _, _, _ = load_ebg4(
                        root_path,
                        subject,
                        data_type="sniff",
                        fs_new=fs_new if fs_new is not None else self.fs
                    )
                    sniff_data = np.expand_dims(sniff_data, axis=1)
                    source_data = sniff_data
                else:
                    pass

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

        if intensity:
            self.source_data = self.source_data[self.labels.squeeze() != 64, ...]
            self.labels = self.labels[self.labels != 64]
            new_labels = [0. if (y == 1 or y == 2 or y == 4) else 1. for y in self.labels]
            self.labels = new_labels
            class_0_count = new_labels.count(0.)
            class_1_count = new_labels.count(1.)
            print(f"N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")
            self.class_weight = torch.tensor(class_0_count / class_1_count)
        elif binary:
            # only consider high intensity odors
            mask = np.logical_not(np.isin(self.labels.squeeze(), [1, 2, 4]))
            self.source_data = self.source_data[mask, ...]
            self.labels = self.labels[mask]
            new_labels = [1. if y == 64 else 0. for y in self.labels]
            self.labels = new_labels
            class_0_count = new_labels.count(0.)
            class_1_count = new_labels.count(1.)
            print(f"N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")
            self.class_weight = torch.tensor(class_0_count / class_1_count)
        else:
            new_labels = [math.log2(y) for y in self.labels]
            self.labels = new_labels
            print(f"new_labels = {set(new_labels)}")

        self.data = self.source_data
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max], axis=(0, -1), keepdims=True)
        self.data = self.data[..., self.t_min:self.t_max] - self.baseline
        self.percentile_95 = np.percentile(np.abs(self.data), 95, axis=-1, keepdims=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = self.data[item, ...]
        if self.normalize or self.modality == "sniff":
            sample = sample / self.percentile_95[item, ...]
        if self.modality == 'ebg-sniff' or self.modality == 'eeg-sniff':
            sample_normalized = sample / self.percentile_95[item, ...]
            sample[-1, ...] = sample_normalized[-1, ...]

        sample = torch.from_numpy(sample)
        return sample, self.labels[item]


if __name__ == "__main__":
    data_args = {'tmin': -0.2, 'tmax': 0.3, 'data_type': 'sensor_ica', 'modality': 'ebg', 'intensity': True}
    ebg_dataset = EBG4(root_path='/Volumes/T5 EVO/Smell/ebg4/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]
