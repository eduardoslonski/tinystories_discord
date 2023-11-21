from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from utils_training import format_magnitude, format_lr
import wandb
from dataclasses import asdict
import math
import os

class Trainer():
    def __init__(self, model, optimizer, loss_fn, dataloader, config, dataloader_val=None, tokenizer=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.config = config
        self.n_epochs = config.n_epochs
        self.device = config.device
        self.dataloader_val = dataloader_val
        self.eval_every = config.eval_every
        self.tokenizer = tokenizer
    
    def train(self):
        wandb.init(
            project = "discord_test",
            name = "dim-128 n_head-2 n_layers-8 ctx-256 btch-128",
            config = asdict(self.config),
            notes = f"{self.model.num_parameters:,} parameters"
        )

        wandb_dict = {}

        pbar = tqdm("Training", total=len(self.dataloader)*self.config.n_epochs)

        total_steps = (len(self.dataloader) * self.config.n_epochs)
        lr_lambda = lambda step: 0.5 * (1 + math.cos(math.pi * step / total_steps))
        scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.optimizer.zero_grad()

        loss_val_print = ''
        acc_val_print = ''

        tokens_seen = 0

        for i_epoch in range(self.config.n_epochs):
            for i, batch in enumerate(self.dataloader):
                wandb_dict = {}
                input_ids, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)

                logits, _ = self.model(input_ids)
                
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

                tokens_seen += torch.sum(labels.ne(-100)).item()
                wandb_dict["loss_train"] = loss.item()
                wandb_dict["lr"] = scheduler.get_last_lr()[0]
                wandb_dict["tokens_seen"] = tokens_seen
                
                if i % self.eval_every == 0:
                    loss_val_positions = self._eval()
                    loss_val_positions_dict = {"loss_val_pos/pos" + str(i): value for i, value in enumerate(loss_val_positions.tolist())}
                    loss_val_positions_dict_agg = self.get_metrics_position(loss=loss_val_positions, batch_size=self.config.batch_size, agg=16, use_wandb=True)

                    loss_val_mean = torch.mean(loss_val_positions)
                    loss_val_print = f"  Val {loss_val_mean:.4f}"

                    wandb_dict["loss_val"] = loss_val_mean
                    wandb_dict.update(loss_val_positions_dict)
                    wandb_dict.update(loss_val_positions_dict_agg)

                # If you need to test the training process on Mac with MPS
                # MPS backend has a weird bug with gradients computation instability
                if self.config.device == 'mps': torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                if i % self.config.checkpoint_every == 0 and i > 0:
                    self._save_checkpoint(scheduler, i, loss)

                pbar.update(1)
                pbar.set_description(f"[{i_epoch}][{i}]  Train {loss.item():.4f} {loss_val_print} {acc_val_print}   Tokens seen {format_magnitude(tokens_seen)}   LR {format_lr(scheduler.get_last_lr()[0])}")
                wandb.log(wandb_dict)

        self._save_checkpoint(scheduler, i, loss)

    def _eval(self):
        self.model.eval()
        with torch.no_grad():
            loss_fn_no_reduction = torch.nn.CrossEntropyLoss(reduction="none")
            loss_sum_positions = torch.zeros(self.config.context_length).to(self.config.device)
            loops = 0

            for batch in tqdm(self.dataloader_val, leave=False, desc="Evaluating"):
                input_ids, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
                if labels.shape[0] != self.config.batch_size_val:
                    continue
                
                logits, _ = self.model(input_ids)
                
                loss = loss_fn_no_reduction(logits.view(-1, logits.shape[-1]), labels.view(-1))
                losses_position = loss.view(self.config.batch_size_val, -1).mean(dim=0)
                loss_sum_positions += losses_position

                loops += 1

        self.model.train()
        return loss_sum_positions / loops

    def get_metrics_position(self, loss, batch_size, agg=1, use_wandb=False):
        losses_agg = loss.view(self.config.context_length//agg, -1).mean(dim=-1)
        
        if use_wandb:
            dict_log = {}
            for i, loss_position in enumerate(losses_agg):
                if agg > 1:
                    dict_log[f'loss_val_pos_agg/pos_{i*agg}-{i*agg+agg-1}'] = loss_position.item()
                else:
                    dict_log[f'loss_val_pos_agg/pos_{i}'] = loss_position.item()
                
            return dict_log

    def _save_checkpoint(self, scheduler, i, loss):
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': i,
            'loss': loss,
        }, f"checkpoints/checkpoint_{i}.pt")