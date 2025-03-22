import numpy as np
from torch.utils.data import Dataset
from adversarial_attack_defense_power_system.dataset_loader.dataset_loader import load_dataset_npy


class PMUEventDataset(Dataset):
    def __init__(self, interconnection='b', train=True, num_pmus=None):
        self.dataset = load_dataset_npy(interconnection=interconnection, num_pmus=num_pmus)
        self.data = None
        self.label = None
        if train is True:
            self.data = np.concatenate((np.transpose(self.dataset['train_data'], (0, 3, 1, 2)),
                                        np.transpose(self.dataset['val_data'], (0, 3, 1, 2)))).astype(np.float32)
            self.label = np.concatenate((self.dataset['train_label'], self.dataset['val_label'])).astype(np.float32)
        else:
            self.data = np.transpose(self.dataset['test_data'], (0, 3, 1, 2)).astype(np.float32)
            self.label = self.dataset['test_label'].astype(np.float32)
        print(f"The final dataset shape: {self.data.shape}")
        print(f"The final label shape: {self.label.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


if __name__ == '__main__':
    trainset = PMUEventDataset(train=True)
    testset = PMUEventDataset(train=False)
