from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoad

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    net = SAMRoad(config)

    val_ds = SatMapDataset(config, is_train=False, dev_run=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler
        )

    trainer.test(net, dataloaders=val_loader, ckpt_path=args.checkpoint)