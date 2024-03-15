from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from dataset import CityScaleDataset
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
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam_road",
        # track hyperparameters and run metadata
        config=config,
        # disable wandb if debugging
        mode='disabled' if (args.fast_dev_run or args.dev_run) else None
    )


    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    net = SAMRoad(config)

    train_ds, val_ds = CityScaleDataset(config, is_train=True), CityScaleDataset(config, is_train=False)

    # verify data
    """
    from triage import visualize_image_and_graph
    import cv2
    for i in range(20):
        sample = train_ds[i]
        img_tensor, coords, edges = sample[0], sample[2].numpy(), sample[3].numpy()
        img = recover_img(img_tensor)
        viz = visualize_image_and_graph(img[:, :, ::-1], coords.reshape((-1, 2)), edges, 128)
        cv2.imshow('viz', viz)
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() # destroys the window showing image
    import pdb
    pdb.set_trace()
    """ 
    # data stats
    """
    n_coords, n_edges = [], []
    for sample in val_ds:
        coords, edges = sample[2], sample[3]
        n_coords.append(coords.shape[0])
        n_edges.append(edges.shape[0])
    n_coords, n_edges = np.array(n_coords), np.array(n_edges)
    print(f'max N coord: {np.max(n_coords)}')
    print(f'max N edge: {np.max(n_edges)}')
    print(f'avg N coord: {np.mean(n_coords)}')
    print(f'avg N edge: {np.mean(n_edges)}')
    import pdb
    pdb.set_trace()
    """

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger()

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath='profile', filename='result_fast_matcher')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler
        )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)