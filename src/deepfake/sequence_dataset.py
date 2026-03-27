# src/deepfake/sequence_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()

        features = torch.tensor(data["features"], dtype=torch.float32)
        label = torch.tensor([data["label"]], dtype=torch.float32)

        return features, label