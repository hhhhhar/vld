from pathlib import Path
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf
from omegaconf import MISSING
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


class xiaoduxiaodu(pl.LightningModule):
    def __init__(self, actor_lr, critic_lr, alpha_lr, target_entropy, 
                 tau, gamma, batch_size, frequency, env=None) -> None:
        super(xiaoduxiaodu, self).__init__()
        self.backbone = backbone


    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            nextObs = self.env.reset()
            state = nextObs["observation"]
            self.episode_reward = 0
            self.episode_step = 0
            self.turn = 0
        else:
            state = self.state

        for _ in range(self.frequency):
            self.turn += 1
            action = self.take_action(state)[0].squeeze(
                0).detach().cpu().numpy()
            action = action.clip(self.env.action_space.low,
                                 self.env.action_space.high).reshape(3)
            nextObs, reward, done, info = self.env.step(action)
            self.dataPool.pool.extend(
                [(state, action, reward, nextObs["observation"], done)])
            torch.cuda.empty_cache()
            state = nextObs["observation"]

            self.episode_reward += reward
            self.episode_step += 1
            if done:
                nextObs = self.env.reset()
                state = nextObs["observation"]
        else:
            self.state = state
            self.log('ep_reward', self.episode_reward, on_step=False,
                     on_epoch=True, prog_bar=True, logger=True)
            self.episode_reward = 0
            self.episode_step = 0
            
    def train_dataloader(self):
        dataset = RLDataset(self.dataPool, self.device, sample_size=self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)

        return dataloader
    
    def configure_optimizers(self):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.actor_lr)
        critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                              lr=self.critic_lr)
        critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                              lr=self.critic_lr)
        log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                               lr=self.alpha_lr)
        return actor_optimizer, critic_1_optimizer, \
            critic_2_optimizer, log_alpha_optimizer

    def training_step(self, batch, batch_idx):
        pass