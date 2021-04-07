from pathlib import Path
from random import randint
from typing import List

import matplotlib.pyplot as plt
import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    SmallestMaxSize,
    CenterCrop
)

from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset


new_size = 256


class LandscapesDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            transform: List = [
                SmallestMaxSize(new_size),
                CenterCrop(new_size, new_size),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                HorizontalFlip(),
                ToTensorV2(),
            ],
        ):
        self.filenames = list(str(p) for p in data_path.glob('**/*.jpg')) + list(str(p) for p in data_path.glob('**/*.png'))
        self.transform = Compose(transform)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(
            self,
            idx,
        ):
        filename = self.filenames[idx]
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        random_idx = randint(0, len(self.filenames) - 1)
        random_filename = self.filenames[random_idx]
        random_image = cv2.imread(random_filename)
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            random_image = self.transform(image=random_image)['image']

        return image, random_image
