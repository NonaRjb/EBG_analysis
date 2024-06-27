import pickle
from typing import Any
# from pywt import wavedec
import mne
import torch
import scipy
import scipy.io as scio
from scipy.signal import resample
import mat73
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Union
import mne
import os


def load_ebg1_mat(filename, trials_to_keep):
    print(f"********** loading sensor data from {filename} **********")
    data_struct = scio.loadmat(filename)

    data = np.asarray(list(data_struct['data_eeg']['trial'][0][0][0]))
    time = data_struct['data_eeg']['time'][0][0][0][0].squeeze()
    channels = [ch[0] for ch in list(data_struct['data_eeg']['label'][0][0].squeeze())]
    labels = data_struct['data_eeg']['trialinfo'][0][0].squeeze()
    fs = data_struct['data_eeg']['fsample'][0][0][0][0]

    indices_air = np.array(trials_to_keep['air'][0][0])
    indices_odor = np.array(trials_to_keep['odor'][0][0])
    indices_all = np.vstack((indices_air, indices_odor))
    indices_all -= 1  # convert MATLAB indices to Python

    channels_to_remove = ['Mstd_L', 'Mstd_R', 'Status', 'BR3', 'BL3']
    new_channels = [channels.index(ch) for ch in channels if ch not in channels_to_remove]

    data = data[indices_all, new_channels, :]
    labels = labels[indices_all].squeeze()
    eeg_data = data[:, :64, :]
    ebg_data = data[:, 64:, :]

    return data, labels, time, fs


def load_ebg1_tfr(filename_air, filename_odor, n_subjects=29):
    """
    Load pre-computed time-frequency representations by Iravani et al. (2020) (Original EBG paper)
    :param filename_air:
    :param filename_odor:
    :param n_subjects:
    :return:
    """
    data_struct_air = scio.loadmat(filename_air)
    data_struct_odor = mat73.loadmat(filename_odor)

    time = data_struct_air['Air_trials'][0][0][0]['time'][0][0]
    freq = data_struct_air['Air_trials'][0][0][0]['freq'][0][0]

    ebg_data = None
    labels = None
    subject_ids = []

    for s in range(n_subjects):
        air_tfr = data_struct_air['Air_trials'][0][s][0]['powspctrm'][0]
        odor_tfr = data_struct_odor['Odor_trials'][s]['powspctrm']
        data_subject = np.vstack((air_tfr, odor_tfr))
        if ebg_data is None:
            ebg_data = data_subject
            labels = np.vstack((np.zeros((len(air_tfr), 1)), np.ones((len(odor_tfr), 1))))
        else:
            ebg_data = np.vstack((ebg_data, data_subject))
            labels = np.vstack((labels, np.vstack((np.zeros((len(air_tfr), 1)), np.ones((len(odor_tfr), 1))))))
        subject_ids.extend(len(data_subject) * [s])

    return ebg_data, labels, np.array(subject_ids), time, freq


def load_ebg3_tfr(filename):
    data_struct = scio.loadmat(filename)

    # there are some NaN values in the raw TFRs (should be cut off maybe until 10 Hz and beyond 1.3 s)
    air_tfr = data_struct['CNT']['TFR'][0][0]['AIR'][0][0]['powspctrm'][0][0]
    odor_tfr = data_struct['CNT']['TFR'][0][0]['ODOR'][0][0]['powspctrm'][0][0]
    ebg_data = np.vstack((air_tfr, odor_tfr))
    labels = np.vstack((np.zeros((len(air_tfr), 1)), np.ones((len(odor_tfr), 1))))

    freq = data_struct['CNT']['TFR'][0][0]['AIR'][0][0]['freq'][0][0].squeeze()
    time = data_struct['CNT']['TFR'][0][0]['AIR'][0][0]['time'][0][0].squeeze()

    return ebg_data, labels, time, freq


def load_ebg4(root, subject_id, data_type, **kwargs):
    if data_type == 'source':
        filename = "source_data.mat"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_source_ebg4(file, kwargs['fs_new'])
    elif data_type == "sensor":
        filename = "preprocessed_data.mat"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_sensor_ebg4(file, kwargs['fs_new'])
    elif data_type == "sensor_ica":
        filename = "preprocessed_data_ica.mat"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_sensor_ica_ebg4(file, kwargs['fs_new'])
    elif data_type == "sniff":
        filename = "s" + str("{:02d}".format(subject_id)) + "_sniff.pkl"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_ebg4_sniff(file, kwargs['fs_new'])
    else:
        raise NotImplementedError

    return data, labels, time, fs


def load_sensor_ebg4(filename, fs_new=None):
    print(f"********** loading sensor data from {filename} **********")
    data_struct = mat73.loadmat(filename)
    data = np.asarray(data_struct['data_eeg']['trial'])
    time = data_struct['data_eeg']['time'][0]
    time = np.asarray(time)
    labels = data_struct['data_eeg']['trialinfo'].squeeze()
    labels = labels[:, 0]
    channels = data_struct['data_eeg_ica']['label']
    channels = [ch[0] for ch in channels]
    fs = 512

    if fs_new is None:
        return data, labels, time, fs
    else:
        # low-pass filter
        data_filtered = mne.filter.filter_data(data, sfreq=fs, l_freq=None, h_freq=min(120, fs_new/2))
        times_new = np.arange(-1., 4., 1 / fs_new)
        data_resampled = resample(data_filtered, num=int(fs_new / fs * data_filtered.shape[-1]), axis=-1)
        return data_resampled, labels, times_new, fs_new


def load_sensor_ica_ebg4(filename, fs_new=None):
    print(f"********** loading sensor data from {filename} **********")
    data_struct = mat73.loadmat(filename)
    data = np.asarray(data_struct['data_eeg_ica']['trial'])
    time = data_struct['data_eeg_ica']['time'][0]
    time = np.asarray(time)
    labels = data_struct['data_eeg_ica']['trialinfo'].squeeze()
    labels = labels[:, 0]
    channels = data_struct['data_eeg_ica']['label']
    channels = [ch[0] for ch in channels]
    fs = 512

    if fs_new is None:
        return data, labels, time, fs
    else:
        # low-pass filter
        data_filtered = mne.filter.filter_data(data, sfreq=fs, l_freq=None, h_freq=min(120, fs_new/2))
        times_new = np.arange(-1., 4., 1 / fs_new)
        data_resampled = resample(data_filtered, num=int(fs_new / fs * data_filtered.shape[-1]), axis=-1)
        return data_resampled, labels, times_new, fs_new


def load_source_ebg4(filename, fs_new=None):
    print(f"********** loading source data from {filename} **********")

    mat73_files = [i for i in range(21, 54)]
    mat73_files.append(3)
    print(mat73_files)
    if int(filename.split("/")[-2]) in mat73_files:
        data_struct = mat73.loadmat(filename)

        data = np.asarray(data_struct['source_data_ROI']['trial'])
        time = data_struct['source_data_ROI']['time'][0]
        labels = data_struct['source_data_ROI']['trialinfo'].squeeze()
    else:
        data_struct = scio.loadmat(filename)

        data = np.asarray(list(data_struct['source_data_ROI']['trial'][0][0][0]))
        time = data_struct['source_data_ROI']['time'][0][0][0][0].squeeze()
        labels = data_struct['source_data_ROI']['trialinfo'][0][0].squeeze()
    fs = 512
    labels = labels[:, 0]
    time = np.asarray(time)

    if fs_new is None:
        return data, labels, time, fs
    else:
        # low-pass filter
        data_filtered = mne.filter.filter_data(data, sfreq=fs, l_freq=None, h_freq=min(120, fs_new/2))
        times_new = np.arange(-1., np.max(time), 1 / fs_new)
        data_resampled = resample(data_filtered, num=int(fs_new / fs * data_filtered.shape[-1]), axis=-1)
        return data_resampled, labels, times_new, fs_new


def load_ebg4_sniff(filename, fs_new=None):
    fs = 400
    orig_times = np.arange(-1., 4., 1 / fs)

    with open(filename, "rb") as f:
        data_dict = pickle.load(f)

    trials = data_dict['trials']
    labels = data_dict['labels']

    # Low-pass filter and downsample TODO
    if fs_new is None:
        return trials, labels, orig_times, fs
    else:
        # low-pass filter
        trials_filtered = mne.filter.filter_data(trials, sfreq=fs, l_freq=None, h_freq=50)
        times_new = np.arange(-1., 4., 1 / fs_new)
        trials_resampled = resample(trials_filtered, num=int(fs_new / fs * trials_filtered.shape[-1]), axis=-1)
        return trials_resampled, labels, times_new, fs_new


def load_ebg4_sniff_mat(path, save_path):
    # Function to load the sniff signal MATLAB struct and split it into per s signals

    labels_map = {
        '1-Butanol low  ': 1,
        '5-Nonanone low ': 2,
        'Undecanal low  ': 4,
        '1-Butanol high ': 8,
        '5-Nonanone high': 16,
        'Undecanal high ': 32,
        'Air            ': 64
    }
    data_struct = scio.loadmat(path)
    data_all = data_struct['data'][0][0]
    for s in range(1, 54):
        data_subject = data_all['subject' + str("{:02d}".format(s))][0]
        trials_subject = data_subject['trials'][0]
        labels_subject = data_subject['odors'][0].squeeze()
        labels_subject = [labels_map[i[0]] for i in labels_subject]
        data_dict = {
            'trials': trials_subject,
            'labels': labels_subject
        }
        with open(os.path.join(save_path, "s" + str("{:02d}".format(s)) + "_sniff.pkl"), "wb") as f:
            pickle.dump(data_dict, f)
    return


def crop_temporal(data, tmin, tmax, tvec, w=None, fs=512):
    if tmin is None:
        t_min = 0
    else:
        t_min = np.abs(tvec - tmin).argmin()

    if w is None:
        if tmax is None:
            t_max = len(tvec)
        else:
            t_max = np.abs(tvec - tmax).argmin()
    else:
        if tmin is None:
            t_max = int(w * fs)
        else:
            tmax = tmin + w
            t_max = np.abs(tvec - tmax).argmin()
    return data[..., t_min:t_max]


def crop_tfr(tfr, tmin, tmax, fmin, fmax, tvec, freqs, w=None, fs=512) -> np.ndarray:
    """
    :param tfr: 4-d data array with the shape (n_trials, n_channels, n_freqs, n_samples)
    :return: cropped array
    """

    if tmin is None:
        t_min = 0
    else:
        t_min = np.abs(tvec - tmin).argmin()

    if w is None:
        if tmax is None:
            t_max = len(tvec)
        else:
            t_max = np.abs(tvec - tmax).argmin()
    else:
        if tmin is None:
            t_max = int(w * fs)
        else:
            tmax = tmin + w
            t_max = np.abs(tvec - tmax).argmin()

    # print(f"Number of Time Samples: {t_max - t_min}")

    if fmin is None:
        f_min = 0
    else:
        f_min = np.abs(freqs - fmin).argmin()
    if fmax is None:
        f_max = len(freqs)
    else:
        f_max = np.abs(freqs - fmax).argmin()

    return tfr[:, :, f_min:f_max, t_min:t_max]


def apply_baseline(tfr, bl_lim, tvec, mode):
    if bl_lim[0] is None:
        baseline_min = 0
    else:
        baseline_min = np.abs(tvec - bl_lim[0]).argmin()
    if bl_lim[1] is None:
        baseline_max = len(tvec)
    else:
        baseline_max = np.abs(tvec - bl_lim[1]).argmin()

    baseline = np.mean(tfr[..., baseline_min:baseline_max], axis=-1, keepdims=True)

    if mode == "mean":
        def fun(d, m):
            d -= m
    elif mode == "ratio":
        def fun(d, m):
            d /= m
    elif mode == "logratio":
        def fun(d, m):
            d /= m
            np.log10(d, out=d)
    elif mode == "percent":
        def fun(d, m):
            d -= m
            d /= m
    elif mode == "zscore":
        def fun(d, m):
            d -= m
            d /= np.std(d[..., baseline_min:baseline_max], axis=-1, keepdims=True)
    elif mode == "zlogratio":
        def fun(d, m):
            d /= m
            np.log10(d, out=d)
            d /= np.std(d[..., baseline_min:baseline_max], axis=-1, keepdims=True)
    else:
        raise NotImplementedError

    fun(tfr, baseline)
    return tfr


# def wavelet_decompose(data, level, wavelet='db4'):
#     coeffs = wavedec(data, wavelet=wavelet, level=level)
#     return coeffs


def apply_tfr(in_data: np.ndarray, fs: float, freqs: np.ndarray, n_cycles: Union[np.ndarray, int] = 3.0,
              method: str = 'morlet'):
    if method == 'morlet':
        tfr_power = mne.time_frequency.tfr_array_morlet(
            in_data, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
            zero_mean=False, use_fft=True, decim=1, output='power', n_jobs=None, verbose=None
        )
    elif method == 'dpss':
        tfr_power = mne.time_frequency.tfr_array_multitaper(
            in_data, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
            zero_mean=False, use_fft=True, decim=1, output='power', n_jobs=None, verbose=None
        )
    else:
        raise NotImplementedError

    return tfr_power


def extract_stat_features(signal, axis=2):
    mean = np.mean(signal, axis=axis, keepdims=True)
    std = np.std(signal, axis=axis, keepdims=True)
    skew = scipy.stats.skew(signal, axis=axis, keepdims=True)
    kurtosis = scipy.stats.kurtosis(signal, axis=axis, keepdims=True)
    entropy = scipy.stats.entropy(signal, axis=axis, keepdims=True)
    entropy = np.nan_to_num(entropy, posinf=10, neginf=-10)
    final_features = np.concatenate((mean, std, skew, kurtosis, entropy), axis=axis)
    return final_features


def tfr_feature_extract(tfr):
    fb1 = tfr[:, :, :20, :]
    fb2 = tfr[:, :, 20:40, :]
    fb3 = tfr[:, :, 40:, :]

    fb1_feutures = extract_stat_features(fb1, axis=2)
    fb2_features = extract_stat_features(fb2, axis=2)
    fb3_features = extract_stat_features(fb3, axis=2)

    return np.concatenate((fb1_feutures, fb2_features, fb3_features), axis=1)


def get_channel_split(modality):
   
    if modality in ["ebg-sniff", "sniff-ebg"]:
        # n_channels = 5
        return 4
    elif modality in ["eeg-sniff", "sniff-eeg"]:
        # n_channels = 64
        return 63
    elif modality in ['source-sniff', 'sniff-source']:
        # n_channels = 5
        return 4
    elif modality in ['source-ebg', 'ebg-source']:
        # n_channels = 8
        return 4
    elif modality in ['source-eeg', 'eeg-source']:
        # n_channels = 67
        return 63
    elif modality in ['eeg-ebg', 'ebg-eeg']:
        # n_channels = 67
        return 63
    else:
        raise NotImplementedError


def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape
    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)
    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))


class RandomNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0, p: float = 0.5) -> None:
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): The input EEG signal.

        Returns:
            torch.Tensor: The output EEG signal after adding random noise.
        '''
        if self.p < torch.rand(1):
            return x
        if self.std is None:
            self.std = x.std(axis=-1, keepdims=True) / 4.
        if self.mean is None:
            self.mean = x.mean(dim=-1, keepdims=True)
        noise = torch.randn_like(x)
        noise = (noise + self.mean) * self.std
        return x + noise

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TemporalJitter(object):
    def __init__(self, max_jitter: int = 20, p: float = 0.5, padding: str = 'zero') -> None:
        self.max_jitter = max_jitter
        self.p = p
        self.padding = padding

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < torch.rand(1):
            return x
        channels, length = x.shape
        shifts = torch.randint(-self.max_jitter, self.max_jitter + 1, (channels,)).to(x.device)
        if self.padding == 'same':
            pass
        elif self.padding == 'zero':
            jittered_x = torch.zeros_like(x)
        else:
            raise NotImplementedError
        for i, shift in enumerate(shifts):
            if shift == 0:
                jittered_x[i, :] = x[i, :]
            elif shift > 0:
                jittered_x[i, shift:] = x[i, :-shift]
            else:  # shift < 0
                jittered_x[i, :shift] = x[i, -shift:]
        return jittered_x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomMask(object):
    def __init__(self, ratio: float = 0.5, p: float = 0.5) -> None:
        self.ratio = ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): The input EEG signal.

        Returns:
            torch.Tensor: The output EEG signal after applying a random mask.
        '''
        if self.p < torch.rand(1):
            return x
        mask = torch.rand_like(x)
        mask = (mask < self.ratio).to(x.dtype)
        return x * mask

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:
            x = self.map(self.dataset[index][0])
        else:
            x = self.dataset[index][0]  # eeg
        y = self.dataset[index][1]  # label
        return x, y

    def __len__(self):
        return len(self.dataset)


class MeanStdNormalize:
    '''
    Perform z-score normalization on the input data. This class allows the user to define the dimension of normalization and the used statistic.

    .. code-block:: python

        transform = Concatenate([
            MeanStdNormalize(axis=0)
        ])
        # normalize along the first dimension (electrode dimension)
        transform(torch.randn(32, 128)).shape(32, 128)

        transform = Concatenate([
            MeanStdNormalize(axis=1)
        ])
        # normalize along the second dimension (temproal dimension)
        transform(torch.randn(32, 128)).shape(32, 128)

    Args:
        mean (np.array, optional): The mean used in the normalization process, allowing the user to provide mean statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        std (np.array, optional): The standard deviation used in the normalization process, allowing the user to provide tandard deviation statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized.
    
    .. automethod:: __call__
    '''

    def __init__(self, mean: Union[np.ndarray, None] = None, std: Union[np.ndarray, None] = None,
                 axis: Union[int, None] = None):
        self.mean = mean
        self.std = std
        self.axis = axis

    def __call__(self, x: np.ndarray):
        '''
        Args:
            x (np.ndarray): The input EEG signals or features.

        Returns:
            np.ndarray: The normalized results.
        '''
        if (self.mean is None) or (self.std is None):
            if self.axis is None:
                mean = x.mean()
                std = x.std()
            else:
                mean = x.mean(axis=self.axis, keepdims=True)
                std = x.std(axis=self.axis, keepdims=True)
        elif not self.axis is None:
            shape = [-1] * len(x.shape)
            shape[self.axis] = 1
            mean = self.mean.reshape(*shape)
            std = self.std.reshape(*shape)
        return (x - mean) / std

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class MinMaxNormalize:
    '''
    Perform min-max normalization on the input data. This class allows the user to define the dimension of normalization and the used statistic.

    .. code-block:: python

        transform = Concatenate([
            MinMaxNormalize(axis=0)
        ])
        # normalize along the first dimension (electrode dimension)
        transform(torch.randn(32, 128)).shape(32, 128)

        transform = Concatenate([
            MinMaxNormalize(axis=1)
        ])
        # normalize along the second dimension (temproal dimension)
        transform(torch.randn(32, 128)).shape(32, 128)

    Args:
        min (np.array, optional): The minimum used in the normalization process, allowing the user to provide minimum statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        max (np.array, optional): The maximum used in the normalization process, allowing the user to provide maximum statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized.
    
    .. automethod:: __call__
    '''

    def __init__(self,
                 min: Union[np.ndarray, None, float] = None,
                 max: Union[np.ndarray, None, float] = None,
                 axis: Union[int, None] = None):
        self.min = min
        self.max = max
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Args:
            x (np.ndarray): The input EEG signals or features.
            
        Returns:
            np.ndarray: The normalized results.
        '''
        if (self.min is None) or (self.max is None):
            if self.axis is None:
                min = x.min()
                max = x.max()
            else:
                min = x.min(axis=self.axis, keepdims=True)
                max = x.max(axis=self.axis, keepdims=True)
        elif not self.axis is None:
            shape = [-1] * len(x.shape)
            shape[self.axis] = 1
            min = self.min.reshape(*shape)
            max = self.max.reshape(*shape)

        return (x - min) / (max - min)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


if __name__ == "__main__":

    # load_ebg4_sniff_mat("/Volumes/T5 EVO/Smell/ebg4_sniff/trials_all_subjects_ICA_cut.mat",
    #                     "/Volumes/T5 EVO/Smell/ebg4_sniff/ebg4_sniff_subject")

    root_path = "/Volumes/T5 EVO/Smell/ebg4"
    # subject = 1
    # data_ebg, labels_ebg, times_ebg, fs_ebg = load_ebg4(root_path, subject, data_type="sensor_ica")
    # data_sniff, labels_sniff, times_sniff, fs_sniff = load_ebg4(root_path, subject, data_type="sniff")

    subjects = [i for i in range(1, 54) if i != 10]
    mismatches = {'sensor_sniff': [], 'source_sniff': [], 'sensor_source': []}
    for s in subjects:
        sensor_data, sensor_labels, sensor_time, sensor_fs = \
            load_ebg4(root=root_path, subject_id=s, data_type="sensor_ica", fs_new=256)

        sniff_data, sniff_labels, sniff_time, sniff_fs = \
            load_ebg4(root=root_path, subject_id=s, data_type="sniff", fs_new=256)

        source_data, source_labels, source_time, source_fs = \
            load_ebg4(root=root_path, subject_id=s, data_type="source", fs_new=256)

        if not np.array_equal(sensor_labels, source_labels):
            mismatches['sensor_source'].append(s)
        if not np.array_equal(sensor_labels, sniff_labels):
            mismatches['sensor_sniff'].append(s)
        if not np.array_equal(source_labels, sniff_labels):
            mismatches['source_sniff'].append(s)

        # find longest match difflib
