import mne
import scipy.io as scio
import mat73
import numpy as np
from numpy.lib.stride_tricks import as_strided
import mne


def load_ebg1_mat(filename, trials_to_keep):
    data_struct = scio.loadmat(filename)

    data = np.asarray(list(data_struct['data_eeg']['trial'][0][0][0]))
    time = data_struct['data_eeg']['time'][0][0][0][0].squeeze()
    channels = [ch[0] for ch in list(data_struct['data_eeg']['label'][0][0].squeeze())]
    labels = data_struct['data_eeg']['trialinfo'][0][0].squeeze()
    fs = data_struct['data_eeg']['fsample'][0][0][0][0]

    indices_air = np.array(trials_to_keep['air'][0][0])
    indices_odor = np.array(trials_to_keep['odor'][0][0])
    indices_all = np.vstack((indices_air, indices_odor))
    indices_all -= 1

    channels_to_remove = ['Mstd_L', 'Mstd_R', 'Status', 'BR3', 'BL3']
    new_channels = [channels.index(ch) for ch in channels if ch not in channels_to_remove]

    data = data[indices_all, new_channels, :]
    labels = labels[indices_all].squeeze()
    eeg_data = data[:, :64, :]
    ebg_data = data[:, 64:, :]

    return eeg_data, ebg_data, labels, time, fs


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


def load_source_data_recent(filename):
    print(f"********** loading source data from {filename} **********")

    mat73_files = [21, 22, 23, 24, 25]
    if int(filename.split("/")[-2]) in mat73_files:
        data_struct = mat73.loadmat(filename)

        data = np.asarray(data_struct['source_data_ROI']['trial'])
        time = data_struct['source_data_ROI']['time']
        labels = data_struct['source_data_ROI']['trialinfo'].squeeze()
    else:
        data_struct = scio.loadmat(filename)

        data = np.asarray(list(data_struct['source_data_ROI']['trial'][0][0][0]))
        time = data_struct['source_data_ROI']['time'][0][0][0][0].squeeze()
        labels = data_struct['source_data_ROI']['trialinfo'][0][0].squeeze()
    fs = 512
    labels = labels[:, 0]

    return data, labels, time, fs


def apply_tfr(in_data: np.ndarray, fs: float, freqs: np.ndarray, n_cycles: float = 3.0, method: str = 'morlet'):
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


def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape
    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)
    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))
