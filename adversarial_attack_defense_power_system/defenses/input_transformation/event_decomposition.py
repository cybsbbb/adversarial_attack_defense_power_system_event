import numpy as np
import tensorflow as tf
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA


def event_decomposition(train_data):
    train_data_reconstruct = np.zeros_like(train_data)
    for idx in range(train_data.shape[0]):
        if idx % 100 == 0:
            print(idx)
        X = train_data[idx]
        train_data_reconstruct[idx, :, :, :] = single_event_decomposition(X)
        # visualization(X)
        # visualization(train_data_reconstruct[idx, :, :, :])
    return train_data_reconstruct


def single_event_decomposition(X):
    X_new = np.zeros_like(X)
    measurement_p = X[:, :, 0]
    measurement_q = X[:, :, 1]
    measurement_v = X[:, :, 2]
    measurement_f = X[:, :, 3]
    # Inter-Event 5 participation
    pca_p = PCA(n_components=5)
    pca_q = PCA(n_components=5)
    pca_v = PCA(n_components=5)
    pca_f = PCA(n_components=5)
    inter_p = pca_p.fit_transform(measurement_p)
    inter_q = pca_q.fit_transform(measurement_q)
    inter_v = pca_v.fit_transform(measurement_v)
    inter_f = pca_f.fit_transform(measurement_f)
    # Residual of the Inter-Event
    residual_p = measurement_p - (pca_p.inverse_transform(inter_p))
    residual_q = measurement_q - (pca_q.inverse_transform(inter_q))
    residual_v = measurement_v - (pca_v.inverse_transform(inter_v))
    residual_f = measurement_f - (pca_f.inverse_transform(inter_f))
    # Intra_Event (5 sparsePCA + 15 SVD)
    # (5 sparsePCA)
    sparse_pca_p = SparsePCA(n_components=5)
    sparse_pca_q = SparsePCA(n_components=5)
    sparse_pca_v = SparsePCA(n_components=5)
    sparse_pca_f = SparsePCA(n_components=5)
    intra_pca_p = sparse_pca_p.fit_transform(residual_p)
    intra_pca_q = sparse_pca_q.fit_transform(residual_q)
    intra_pca_v = sparse_pca_v.fit_transform(residual_v)
    intra_pca_f = sparse_pca_f.fit_transform(residual_f)
    # Residual of the 5 SparsePCA
    residual_sparse_p = residual_p - (intra_pca_p @ sparse_pca_p.components_) + sparse_pca_p.mean_
    residual_sparse_q = residual_q - (intra_pca_q @ sparse_pca_q.components_) + sparse_pca_q.mean_
    residual_sparse_v = residual_v - (intra_pca_v @ sparse_pca_v.components_) + sparse_pca_v.mean_
    residual_sparse_f = residual_f - (intra_pca_f @ sparse_pca_f.components_) + sparse_pca_f.mean_
    # (10 PCA)
    U_p, sigma_p, VT_p = linalg.svd(residual_sparse_p, full_matrices=False)
    U_q, sigma_q, VT_q = linalg.svd(residual_sparse_q, full_matrices=False)
    U_v, sigma_v, VT_v = linalg.svd(residual_sparse_v, full_matrices=False)
    U_f, sigma_f, VT_f = linalg.svd(residual_sparse_f, full_matrices=False)
    sigma_p[10:] = 0
    sigma_q[10:] = 0
    sigma_v[10:] = 0
    sigma_f[10:] = 0
    sigma_first_p = np.diag(sigma_p)
    sigma_first_q = np.diag(sigma_q)
    sigma_first_v = np.diag(sigma_v)
    sigma_first_f = np.diag(sigma_f)
    # Reconstruct Event from Event participation
    svd_reconstruct_p = np.dot(U_p, np.dot(sigma_first_p, VT_p))
    sparse_reconstruct_p = svd_reconstruct_p + (intra_pca_p @ sparse_pca_p.components_) + sparse_pca_p.mean_
    reconstruct_p = sparse_reconstruct_p + (pca_p.inverse_transform(inter_p))
    svd_reconstruct_q = np.dot(U_q, np.dot(sigma_first_q, VT_q))
    sparse_reconstruct_q = svd_reconstruct_q + (intra_pca_q @ sparse_pca_q.components_) + sparse_pca_q.mean_
    reconstruct_q = sparse_reconstruct_q + (pca_q.inverse_transform(inter_q))
    svd_reconstruct_v = np.dot(U_v, np.dot(sigma_first_v, VT_v))
    sparse_reconstruct_v = svd_reconstruct_v + (intra_pca_v @ sparse_pca_v.components_) + sparse_pca_v.mean_
    reconstruct_v = sparse_reconstruct_v + (pca_v.inverse_transform(inter_v))
    svd_reconstruct_f = np.dot(U_f, np.dot(sigma_first_f, VT_f))
    sparse_reconstruct_f = svd_reconstruct_f + (intra_pca_f @ sparse_pca_f.components_) + sparse_pca_f.mean_
    reconstruct_f = sparse_reconstruct_f + (pca_f.inverse_transform(inter_f))
    # Generate reconstructed event
    X_new[:, :, 0] = reconstruct_p
    X_new[:, :, 1] = reconstruct_q
    X_new[:, :, 2] = reconstruct_v
    X_new[:, :, 3] = reconstruct_f
    return X_new