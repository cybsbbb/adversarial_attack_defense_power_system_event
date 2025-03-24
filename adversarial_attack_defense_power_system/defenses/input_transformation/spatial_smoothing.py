import numpy as np


def spatial_smoothing(train_data, window_size=5):
    res = np.zeros_like(train_data)
    half_window = window_size//2
    for idx in range(train_data.shape[0]):
        for measurement_idx in range(4):
            matrix = train_data[idx, :, :, measurement_idx]
            for i in range(360):
                left = max(i - half_window, 0)
                right = min(i + half_window, 360)
                res[idx, i, :, measurement_idx] = np.mean(matrix[left: right, :], axis=0)
    return res
