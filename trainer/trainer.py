from collections import defaultdict
from typing import Dict, Any
from pathlib import Path
from typing import List, Optional
import sys

import numpy as np
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
                 val_loader: DataLoader = None,
                 scheduler=None):
        self.model = model
        self.config = config
        self.optimizers = self.model.configure_optimizers()[0]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.schedulers = scheduler
        self.global_step = 0

        print("Created model. Validate is",
              "on" if self.val_loader is not None else "off")

    def save_checkpoint(self,
                        save: bool = True
                        ) -> Dict[Any, Any]:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
        }
        path = Path() / self.config["save_checkpoint_path"] \
               / f"step={self.global_step}.pt"
        for opt in self.optimizers:
            label = opt["label"]
            optimizer = opt["value"]

            checkpoint[
                f"optimizer_{label}_state_dict"] = optimizer.state_dict()
        if save:
            torch.save(checkpoint, path)
        return checkpoint

    def load_checkpoint(self,
                        checkpoint_path: Path,
                        ) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        for opt in self.optimizers:
            label = opt["label"]
            optimizer = opt["value"]
            optimizer.load_state_dict(
                checkpoint[f"optimizer_{label}_state_dict"])

        self.global_step = checkpoint["global_step"] + 1

    def setup_hooks(self, problems: dict):
        if not self.config["debug"]:
            return

        def hook_fn(layer, input, output):
            if isinstance(output, tuple):
                output = output[0]
            output = output.detach().cpu()
            if not output.isfinite().all():
                if isinstance(input, tuple):
                    input = input[0]
                input = input.detach().cpu()
                problems[str(type(layer))] = {"input": input.numpy(),
                                              "output": output.numpy()}

        def backward_hook(layer, grad_input, grad_output):
            if isinstance(grad_input, tuple):
                grad_input = grad_input[0]
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            grad_input = grad_input.detach().cpu()
            grad_output = grad_output.detach().cpu()
            if not grad_input.isfinite().all() or \
                    not grad_output.isfinite().all():
                problems[str(type(layer))] = {'grad_input': grad_input.numpy(),
                                              'grad_output': grad_output.numpy()}

        for name, layer in self.model._modules.items():
            if isinstance(layer, nn.Sequential):
                pass
            else:
                layer.register_backward_hook(backward_hook)
                layer.register_forward_hook(hook_fn)

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return

        if self.global_step < self.config["validate_start"] \
                or self.global_step % self.config["validate_period"] != 0:
            return

        val_info = defaultdict(float)

        for batch in self.val_loader:
            current_iter_info = dict()
            for opts in self.optimizers:
                step = opts["label"]

                info = self.model.validation_step(batch=batch,
                                                  step=step)

                current_iter_info = {**current_iter_info, **info}

            for key, value in current_iter_info.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                val_info[key] += value

        for key in val_info:
            val_info[key] /= len(self.val_loader)

        wandb.log(val_info)

    @torch.enable_grad()
    def train_epoch(self) -> None:
        pbar = tqdm(self.train_loader, position=1, leave=False)

        problems = dict()
        self.setup_hooks(problems)
        prev_step = self.save_checkpoint(Path('/'), save=False)

        for batch in pbar:
            current_iter_info = dict()
            for opts in self.optimizers:

                step = opts["label"]
                optimizer = opts["value"]

                info = self.model.training_step(batch=batch,
                                                step=step)
                if info.get('loss', 0) > 50:
                    import pdb;
                    pdb.set_trace()
                if len(problems) > 0:
                    torch.save(prev_step,
                               Path(self.config['checkpoint_path'] + '_last'))
                    with open("logs.txt", 'w') as f:
                        for layer, values in problems.items():
                            f.write(" ".join(layer) + '\n')
                            f.write("input\n")
                            f.write(" ".join(values['input']))
                            f.write('output\n')
                            f.write(" ".join(values['output']))
                            f.write('\n\n\n')
                    import pdb;
                    pdb.set_trace()

                prev_step = self.save_checkpoint(save=False)
                current_iter_info = {**current_iter_info, **info}

                if step == "generator":
                    loss = info["train-gen: loss"]
                else:
                    loss = info["train-dis: loss"]

                loss.backward()
                utils.clip_grad_norm_(parameters=self.model.parameters(),
                                      max_norm=10)

                optimizer.step()
                optimizer.zero_grad()

            self.validate()

            if self.global_step % self.config["picture_frequency"] == 0:
                self._show_picture()

            if self.global_step % self.config["save_period"] == 0:
                self.save_checkpoint()

            self._update_logs(current_iter_info, pbar)
            self.global_step += 1

    def _update_logs(self, info: dict, pbar: tqdm):
        info_values = dict()
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            info_values[key] = value

        pbar.set_postfix(info_values)

        if self.global_step > self.config["send_wandb"]:
            wandb.log(info)

    def _show_picture(self):
        batch = next(iter(self.val_loader))
        sample = self.model.sample(batch).cpu()

        images = (torchvision.utils.make_grid(sample,
                                              nrow=self.config["batch_size"],
                                              normalize=False)
                  .permute(1, 2, 0) * Tensor([0.229, 0.224, 0.225])
                  + Tensor([0.485, 0.456, 0.406])).numpy()

        np.nan_to_num(images, copy=False, nan=1)
        images = images.clip(0, 1)

        wandb.log({"generated images": [wandb.Image(images)]})

    def fit(self):
        n_epochs = self.config["n_epochs"]
        wandb.init(project="test-drive", config=self.config)
        wandb.watch(self.model)

        for _ in tqdm(range(n_epochs), position=0):
            self.train_epoch()

        wandb.finish()
