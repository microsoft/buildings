# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
import os
import warnings
import torch

torch.set_float32_matmul_precision("medium")
os.environ.update(
    {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_USE_HEAD": "No",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
        "GDAL_CACHEMAX": "200",
    }
)

import lightning.pytorch as pl
import pandas as pd

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*"
)
warnings.filterwarnings(
    "ignore", message="Default grid_sample and affine_grid behavior has changed"
)

import yaml
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from tempo.datamodules import SegmentationDataModule
from tempo.trainers import SegmentationTask
from tempo.config import TrainerConfig
from tempo.utils import apply_temperature, weighted_train_val_split


class CosineAnnealingCheckpointCallback(Callback):
    """
    Custom checkpoint callback that saves checkpoints around absolute minimum learning rate.

    In cosine annealing with patience T:
    - Epoch 0 to T: LR decreases from max to min
    - Epoch T to 2*T: LR increases from min to max
    - Epoch 2*T to 3*T: LR decreases from max to min
    - And so on...

    This callback saves 3 checkpoints per plateau around the minimum LR points:
    - At epoch T-1, T, T+1 (around first minimum)
    - At epoch 3*T-1, 3*T, 3*T+1 (around second minimum)
    - At epoch 5*T-1, 5*T, 5*T+1 (around third minimum)
    - etc. (left, center, right around odd multiples of patience)

    Additionally saves:
    - Best checkpoint based on validation loss
    - Latest checkpoint after each epoch
    """

    def __init__(
        self, patience: int, max_epochs: int, dirpath: str, save_last: bool = True
    ):
        super().__init__()
        self.patience = patience
        self.max_epochs = max_epochs
        self.dirpath = Path(dirpath)
        self.save_last = save_last
        self.best_val_loss = float("inf")

        # Calculate epochs for 3 checkpoints per plateau (left, center, right)
        self.target_epochs = {}  # epoch -> (cycle, position)
        cycle = 0
        while True:
            # Minimum LR occurs at odd multiples of patience
            center_epoch = (2 * cycle + 1) * patience
            if center_epoch >= max_epochs:
                break

            # Add left, center, right checkpoints for this plateau
            for offset, position in [(-1, "left"), (0, "center"), (1, "right")]:
                epoch = center_epoch + offset
                if 0 <= epoch < max_epochs:
                    self.target_epochs[epoch] = (cycle, position)
            cycle += 1

        print(
            f"Will save {len(self.target_epochs)} checkpoints around minimum LR epochs: {sorted(self.target_epochs.keys())}"
        )

        # Ensure output directory exists
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss")

        if current_val_loss is not None:
            current_val_loss = float(current_val_loss)
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                best_path = self.dirpath / "best.ckpt"
                trainer.save_checkpoint(best_path)

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        # Save at target epochs (left, center, right around minimum LR)
        if current_epoch in self.target_epochs:
            cycle, position = self.target_epochs[current_epoch]
            checkpoint_path = (
                self.dirpath
                / f"min_lr_cycle{cycle:02d}_{position}_epoch_{current_epoch:03d}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(
                f"Saved minimum LR checkpoint (cycle {cycle}, {position}) at epoch {current_epoch}: {checkpoint_path}"
            )

        # Save last checkpoint after every epoch
        if self.save_last:
            last_path = self.dirpath / "last.ckpt"
            trainer.save_checkpoint(last_path)
            if current_epoch == self.max_epochs - 1:
                print(f"Saved final checkpoint: {last_path}")


def get_parser():
    parser = argparse.ArgumentParser()

    # pass specify options using yaml instead of cli
    parser.add_argument(
        "--config",
        type=str,
        help="Pass a yaml instead of using the CLI. Uses train_config.yaml by default.",
    )
    return parser


def main(args):
    """Sets Trainer configs based on train_config.yaml."""

    with Path(args.config).open() as f:
        config_dict = yaml.safe_load(f)

    # instantiate TrainerConfig to validate inputs
    train_config = TrainerConfig(**config_dict)

    # seed everything
    pl.seed_everything(train_config.seed)
    df = pd.read_csv(train_config.index)
    quads = df["image"].apply(lambda x: x.split("/")[-1][:-4]).tolist()

    # Vectorized temperature scaling for training weights
    raw_weights = df["weight"].values + train_config.base_weight
    weights = apply_temperature(raw_weights, train_config.weight_temperature).tolist()

    # Keep validation weights unchanged
    val_weights = (df["val_weight"] + 2.0).values.tolist()

    all_image_fns = df["image"].tolist()
    all_overture_fns = df["overture"].tolist()
    all_mask_fns = df["mask"].tolist()
    fn_to_weight = {quad: weights[i] for i, quad in enumerate(quads)}

    (train_image_fns, train_overture_fns, train_mask_fns), (
        val_image_fns,
        val_overture_fns,
        val_mask_fns,
    ) = weighted_train_val_split(
        (all_image_fns, all_overture_fns, all_mask_fns),
        weights=val_weights,
        val_size=train_config.val_ratio,
        random_state=train_config.seed,
    )
    total_val_weight_sum = 0
    for fn in val_mask_fns:
        total_val_weight_sum += fn_to_weight[str(fn).split(".tif")[0].split("/")[-1]]
    print(f"Using {len(val_image_fns)} images and masks for validation.")
    print(f"Total validation weight sum: {total_val_weight_sum}")
    print(f"Average of {total_val_weight_sum / len(val_mask_fns)} per image.")

    # setup training
    dm = SegmentationDataModule(
        image_fns={"train": train_image_fns, "valid": val_image_fns},
        overture_fns={"train": train_overture_fns, "valid": val_overture_fns},
        mask_fns={"train": train_mask_fns, "valid": val_mask_fns},
        fn_sample_weights=fn_to_weight,
        batch_size=train_config.batch_size,
        patch_size=train_config.patch_size,
        num_workers=train_config.num_workers,
        train_batches_per_epoch=train_config.batches_per_epoch,
        valid_batches_per_epoch=int(train_config.batches_per_epoch / 8),
        band_normalizers=train_config.band_normalizers,
        target_bands=train_config.target_bands,
        target_normalizers=train_config.target_normalizers,
    )

    task = SegmentationTask(
        segmentation_model=train_config.segmentation_model_name,
        encoder_name=train_config.backbone_name,
        encoder_weights=train_config.weight_init,
        in_channels=train_config.in_channels,
        losses=train_config.losses,
        learning_rate=train_config.lr,
        learning_rate_schedule_patience=train_config.patience,
        scheduler=train_config.scheduler,
        optimizer=train_config.optimizer,
        weight_decay=train_config.weight_decay,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
        activation=train_config.activation,
        erase_blocks_in_priors=train_config.erase_blocks_in_priors,
        priors_bands=train_config.priors_bands,
        erase_priors_p=train_config.erase_priors_p,
        erase_priors_scale=train_config.erase_priors_scale,
        erase_priors_ratio=train_config.erase_priors_ratio,
        target_bands=train_config.target_bands,
        unet_heads=train_config.unet_heads,
        band_normalizers=train_config.band_normalizers,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        train_config.log_dir,
        name=train_config.experiment_name,
    )

    # Use custom checkpoint callback for cosine annealing plateau middle epochs
    if train_config.scheduler == "cosine":
        checkpoint_callback = CosineAnnealingCheckpointCallback(
            patience=train_config.patience,
            max_epochs=train_config.max_epochs,
            dirpath=train_config.output_dir,
            save_last=True,
        )
    else:
        # Use default ModelCheckpoint for non-cosine schedulers
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=train_config.output_dir,
            save_top_k=5,
            save_last=True,
        )

    trainer_args = {
        "accelerator": "gpu",
        "devices": train_config.gpu_ids,
        "callbacks": [checkpoint_callback],
        "logger": tb_logger,
        "default_root_dir": train_config.output_dir,
        "max_epochs": train_config.max_epochs,
        "precision": train_config.precision,
    }

    trainer = pl.Trainer(**trainer_args)

    checkpoint_path = Path(train_config.output_dir) / "last.ckpt"
    if checkpoint_path.exists():
        trainer.fit(model=task, datamodule=dm, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model=task, datamodule=dm)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
