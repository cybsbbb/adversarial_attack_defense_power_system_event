import numpy as np
import tensorflow as tf
from scipy import signal


def low_pass_filtering_seq(seq, sos):
    # sos = signal.butter(10, 10, 'lowpass', fs=30, output='sos')
    filtered = signal.sosfilt(sos, seq)
    return filtered


def low_pass_filtering(train_data):
    res = np.zeros_like(train_data)
    shape = train_data.shape
    sos = signal.butter(10, 10, 'lowpass', fs=30, output='sos')
    for sample_idx in range(shape[0]):
        for pmu_idx in range(shape[2]):
            for measurements in range(shape[3]):
                res[sample_idx, :, pmu_idx, measurements] = low_pass_filtering_seq(train_data[sample_idx, :, pmu_idx, measurements], sos)
    return res
