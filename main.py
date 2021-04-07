import torch
import os
from pathlib import Path
from random import randint
from typing import List

import matplotlib.pyplot as plt
import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    Resize,
    SmallestMaxSize,
    CenterCrop
)
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.Dataset import LandscapesDataset
from src.hidt_model import HiDTModel
from trainer.trainer import Trainer
from config.hidt_config import config


def main():
    model = HiDTModel(config=config,
            device=torch.device(config["device"]))
    model.to(torch.device(config["device"]))
    path = Path(config["data_path"])
    dataset = LandscapesDataset(path)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=config["batch_size"],
                              num_workers=4)
    trainer = Trainer(model=model,
                      config=config,
                      train_loader=train_loader)
#     print(trainer.optimizers, type(trainer.optimizers))
    trainer.fit()
    
if __name__ == "__main__":
    main()
