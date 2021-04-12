import torch
from pathlib import Path

from torch.utils.data import DataLoader

from src.Dataset import LandscapesDataset, train_test_split
from src.hidt_model import HiDTModel
from trainer.trainer import Trainer
from config.hidt_config import config


def main():
    model = HiDTModel(config=config,
                      device=torch.device(config["device"]))
    model.to(torch.device(config["device"]))
    path = Path(config["data_path"])
    dataset = LandscapesDataset(path)

    train_dataset, val_dataset = train_test_split(dataset, 
                                                  config["test_size"])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config["batch_size"],
                              num_workers=config["num_workers"],
                              drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            drop_last=True)
    trainer = Trainer(model=model,
                      config=config,
                      train_loader=train_loader,
                      val_loader=val_loader)
    if config["from_pretrained"]:
        trainer.load_checkpoint(Path(config["checkpoint_path"]))
    trainer.fit()


if __name__ == "__main__":
    main()
