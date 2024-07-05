import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import data_utils
from dataset.data_utils import apply_tfr, apply_baseline, crop_tfr
from train_logistic_reg import load_ebg4_array

# cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
# cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Volumes/T5 EVO/Smell/"

if __name__ == "__main__":
    subjects = [s for s in np.arange(1, 53) if s != 10]
    fmin = 10
    fmax = 70
    tmin = 0.0
    tmax = 1.0

    tfr_0s = None
    tfr_1s = None
    for subject in subjects:
        data_array, labels, time_vec, fs = load_ebg4_array(
            os.path.join(local_data_path, "ebg4"),
            subject_id=subject,
            data_type="sensor_ica",
            modality="ebg",
            tmin=None, tmax=None,
            bl_lim=None)

        labels = np.asarray(labels)

        freqs = np.arange(5, 100)
        tfr = apply_tfr(data_array, fs, freqs=freqs, n_cycles=7, method='morlet')
        tfr = apply_baseline(tfr, bl_lim=(-1.0, -0.6), tvec=time_vec, mode='logratio')
        tfr = crop_tfr(tfr, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, tvec=time_vec,
                       freqs=freqs, w=None)
        tfr = np.mean(tfr, axis=1)  # average over channels

        class0 = np.where(labels == 0.)
        class1 = np.where(labels == 1.)

        tfr_1 = np.squeeze(tfr[class1, ...], axis=0)  # clean air trials
        tfr_0 = np.squeeze(tfr[class0, ...], axis=0)  # odor trials

        tfr_1_mean = np.mean(tfr_1, axis=0)
        tfr_0_mean = np.mean(tfr_0, axis=0)

        if tfr_0s is None:
            tfr_0s = np.expand_dims(tfr_0_mean, axis=0)
        else:
            tfr_0s = np.vstack((tfr_0s, np.expand_dims(tfr_0_mean, axis=0)))

        if tfr_1s is None:
            tfr_1s = np.expand_dims(tfr_1_mean, axis=0)
        else:
            tfr_1s = np.vstack((tfr_1s, np.expand_dims(tfr_1_mean, axis=0)))

    tfr_0s_mean = np.squeeze(np.mean(tfr_0s, axis=0))
    tfr_1s_mean = np.squeeze(np.mean(tfr_1s, axis=0))

    diff = tfr_0s_mean - tfr_1s_mean

    yticks = np.arange(fmin, fmax + 1, 10)
    yticks = [str(t) for t in yticks]
    # xticks = np.arange(tmin, tmax+0.1, 0.1)
    # xticks = [str(t) for t in xticks]

    plt.imshow(diff, cmap='seismic')
    plt.colorbar()
    plt.yticks(ticks=np.arange(0, len(diff)+1, 10), labels=yticks)
    # plt.xticks(ticks=np.arange(0, diff.shape[-1]+1, ))
    plt.show()
