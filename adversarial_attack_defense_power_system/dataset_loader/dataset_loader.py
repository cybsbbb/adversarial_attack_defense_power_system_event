import numpy as np
from pathlib import Path


def load_dataset_npy(interconnection='b', num_pmus=None):
    if interconnection == 'b' or interconnection == 'B':
        interconnection = 'b'
        num_pmus = 40 if num_pmus is None else num_pmus
    if interconnection == 'c' or interconnection == 'C':
        interconnection = 'c'
        num_pmus = 176 if num_pmus is None else num_pmus
    script_path = Path(__file__).resolve().parent
    dataset = dict()
    if interconnection == 'b':
        train_data = np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/train_data.npy')[:, :, :num_pmus, :]
    else:
        train_data = np.concatenate((
            np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/train_data_1.npy'),
            np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/train_data_2.npy')))[:, :, :num_pmus, :]
    train_label = np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/train_label.npy')
    val_data = np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/val_data.npy')[:, :, :num_pmus, :]
    val_label = np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/val_label.npy')
    test_data = np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/test_data.npy')[:, :, :num_pmus, :]
    test_label = np.load(f'{script_path}/../../data/datasets/ic_{interconnection}/test_label.npy')
    dataset['interconnection'] = interconnection
    dataset['train_data'] = train_data
    dataset['train_label'] = train_label
    dataset['val_data'] = val_data
    dataset['val_label'] = val_label
    dataset['test_data'] = test_data
    dataset['test_label'] = test_label
    print_dataset_info(dataset)
    return dataset


def print_dataset_info(dataset):
    print(f"Dataset of the interconnection: {dataset['interconnection']}")
    print(f"Shape of the train data: {dataset['train_data'].shape}")
    print(f"Shape of the train label: {dataset['train_label'].shape}")
    print(f"Shape of the validation data: {dataset['val_data'].shape}")
    print(f"Shape of the validation label: {dataset['val_label'].shape}")
    print(f"Shape of the test data: {dataset['test_data'].shape}")
    print(f"Shape of the test label: {dataset['test_label'].shape}")
    return 0


if __name__ == '__main__':
    load_dataset_npy(interconnection='b')
    print()
    load_dataset_npy(interconnection='c')
