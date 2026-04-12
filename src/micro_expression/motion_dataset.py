import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class MotionDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label_name in ["real", "fake"]:
            label = 1 if label_name == "real" else 0
            label_path = os.path.join(root_dir, label_name)

            for video in os.listdir(label_path):
                video_path = os.path.join(label_path, video)

                frames = sorted(os.listdir(video_path))

                if len(frames) < 5:
                    continue

                imgs = []

                for f in frames[:40]:
                    img = cv2.imread(os.path.join(video_path, f))
                    if img is None:
                        continue

                    img = cv2.resize(img, (224, 224))
                    img = img.astype("float32") / 255.0
                    imgs.append(img)

                for i in range(1, len(imgs)):
                    diff = np.abs(imgs[i] - imgs[i-1]) * 5.0  # amplify motion
                    diff = np.clip(diff, 0, 1)

                    diff = diff.transpose(2, 0, 1)
                    self.samples.append((diff, label))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32)
        )