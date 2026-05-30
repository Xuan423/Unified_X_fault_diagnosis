from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class Default_dataset(Dataset):
    def __init__(self, args, flag):
        self.flag = flag
        self.data_loader(args.data_dir, args.target)
        self.data_create()

    def data_loader(self, data_dir, target):
        base = Path(data_dir)
        self.data = np.load(base / f"{target}_data.npy").astype(np.float32)
        self.labels = np.load(base / f"{target}_label.npy").astype(np.int64)

    def data_create(self):
        train_ratio = 0.6
        val_ratio = 0.1

        train_indices, val_indices, test_indices = [], [], []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            n_train = int(len(label_indices) * train_ratio)
            n_val = int(len(label_indices) * val_ratio)

            train_indices.extend(label_indices[:n_train])
            val_indices.extend(label_indices[n_train : n_train + n_val])
            test_indices.extend(label_indices[n_train + n_val :])

        if self.flag == "train":
            selected_indices = train_indices
        elif self.flag == "val":
            selected_indices = val_indices
        elif self.flag == "test":
            selected_indices = test_indices
        else:
            raise ValueError("Invalid flag. Please choose from 'train', 'val', or 'test'.")

        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)
        self.selected_data = self.data[self.selected_indices]
        self.selected_labels = self.labels[self.selected_indices]

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, idx):
        return self.selected_data[idx], self.selected_labels[idx]
