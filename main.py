import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
from random import randint
from typing import List

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
                              num_workers=4,
                              drop_last=True)
    trainer = Trainer(model=model,
                      config=config,
                      train_loader=train_loader)

    if config["checkpoint_path"] is not None:
        trainer.load_checkpoint(Path(config["checkpoint_path"]))
    
    trainer.fit()
    
if __name__ == "__main__":
    main()
