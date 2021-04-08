from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb


class Trainer():
    def __init__(self, 
                 model: nn.Module,
                 config: dict,
                 train_loader: DataLoader,
                 val_loader: DataLoader=None,
                 scheduler=None):
        self.model = model
        self.config = config
        self.optimizers = self.model.configure_optimizers()[0]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.schedulers = scheduler
        self.global_step = 0

    def save_checkpoint(self,
                        checkpoint_path: Path,
                        ) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
        }

        for opt in self.optimizers:
            label = opt["label"]
            optimizer = opt["value"]

            checkpoint[f"optimizer_{label}_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, 
                        checkpoint_path: Path,
                        ) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        for opt in self.optimizers:
            label = opt["label"]
            optimizer = opt["value"]
            optimizer.load_state_dict(checkpoint[f"optimizer_{label}_state_dict"])

    @torch.enable_grad()
    def train_epoch(self) -> None:
        self.model.train()
        
        pbar = tqdm(self.train_loader, position=1, leave=False)
        
        for batch in pbar:
            current_iter_info = dict()
            for opts in self.optimizers:

                step = opts["label"]
                optimizer = opts["value"]

                info = self.model.training_step(batch=batch, 
                                                step=step)
                current_iter_info = {**current_iter_info, **info}
                loss = info[step]['loss']
                loss.backward()
                utils.clip_grad_norm_(parameters=self.model.parameters(),
                                      max_norm=10)

                optimizer.step()
                optimizer.zero_grad()

            if self.global_step % self.config["picture_frequency"] == 0:
                self._show_picture()
                
            if self.global_step % self.config["save_period"] == 0:
                checkpoint_path = \
                    Path.cwd() / "notcheckpoints" / f"step={self.global_step}.pt"
                self.save_checkpoint(checkpoint_path)

            self._update_logs(current_iter_info, pbar)
            self.global_step += 1

    def _update_logs(self, info: dict, pbar: tqdm):
        current = dict()
        for key in info:
            for inner_key in info[key]:
                value = info[key][inner_key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                current[key + "-" + inner_key] = value
        
        pbar.set_postfix(current)

        if self.global_step > self.config["send_wandb"]:
            wandb.log(current)

    def _show_picture(self):
        batch = next(iter(self.train_loader))
        sample = self.model.sample(batch).cpu()

        images = (torchvision.utils.make_grid(sample, 
                                              nrow=self.config["batch_size"],
                                              normalize=False)
                  .permute(1, 2, 0) * Tensor([0.229, 0.224, 0.225]) 
                  + Tensor([0.485, 0.456, 0.406])).numpy().clip(0, 1)

        wandb.log({"generated images": [wandb.Image(images)]})

    def fit(self):
        n_epochs = self.config["n_epochs"]
        wandb.init(project="test-drive", config=self.config)
        wandb.watch(self.model)
                
        for epoch in tqdm(range(n_epochs), position=0):
            self.train_epoch()

        wandb.finish()
