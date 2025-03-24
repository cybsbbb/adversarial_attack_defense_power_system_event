import numpy as np
import tensorflow as tf
from numpy import linalg

tf.random.set_seed(0)
np.random.seed(0)


def svd_decomposition(train_data, svd_remaining=6):
    train_data_svd = np.zeros_like(train_data)
    for idx in range(train_data.shape[0]):
        for measurement_idx in range(4):
            matrix = train_data[idx, :, :, measurement_idx]
            U, sigma, VT = linalg.svd(matrix, full_matrices=False)
            sigma[svd_remaining:] = 0
            sigma_first = np.diag(sigma)
            train_data_svd[idx, :, :, measurement_idx] = np.dot(U, np.dot(sigma_first, VT))
    return train_data_svd
