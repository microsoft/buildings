# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict
import kornia.augmentation as K
import lightning.pytorch as pl
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from tempo.models import DownsampledRegressionUnet


class HuberLoss(torch.nn.Module):
    """Huber loss that ignores specified values (e.g., -99 for no data)."""

    def __init__(self, val=-99, delta=0.7):
        super().__init__()
        self.val = val
        self.delta = delta

    def forward(self, input, target):
        input = input.reshape(-1)
        target = target.reshape(-1)
        mask = target != self.val
        if not mask.any():
            return input.new_zeros(1, requires_grad=True)
        input = input[mask].float()
        target = target[mask].float()
        return F.huber_loss(input, target, delta=self.delta)


class SegmentationTask(pl.LightningModule):

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        num_output_bands = len(self.hparams.get("target_bands"))

        # Initialize UNet model
        self.model = DownsampledRegressionUnet(
            encoder_name=self.hparams["encoder_name"],
            encoder_weights=self.hparams["encoder_weights"],
            in_channels=self.hparams.get("in_channels"),
            num_output_bands=num_output_bands,
            activation=self.hparams.get("activation"),
            decoder_use_batchnorm=True,
            unet_heads=self.hparams.get("unet_heads"),
        )

        # Initialize loss functions for each band (only ignored_nan_huber is supported)
        self.losses = {}
        target_bands = self.hparams.get("target_bands")

        for i, band_idx in enumerate(target_bands):
            self.losses[band_idx] = HuberLoss(delta=0.7)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function."""
        super().__init__()
        self.save_hyperparameters()
        self.config_task()

        # Setup augmentations
        augmentation_list = [K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5)]
        self.train_augmentations = K.AugmentationSequential(
            *augmentation_list,
            data_keys=["input", "mask"],
        )

        # Setup overture masking
        self.mask_priors = (
            K.RandomErasing(
                p=self.hparams.get("erase_priors_p"),
                scale=self.hparams.get("erase_priors_scale"),
                ratio=self.hparams.get("erase_priors_ratio"),
                same_on_batch=False,
            )
            if kwargs.get("erase_blocks_in_priors")
            else None
        )

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:

        x, y = batch
        target_bands = self.hparams.get("target_bands")
        epoch = self.trainer.current_epoch

        # Apply augmentations
        if self.train_augmentations is not None:
            x, y = self.train_augmentations(x, y)

        # Apply random block masking to prior bands if enabled
        if self.hparams.get("erase_blocks_in_priors"):
            if self.hparams.get("priors_bands") is not None:
                for band_idx in self.hparams["priors_bands"]:
                    x[:, band_idx : band_idx + 1] = self.mask_priors(
                        x[:, band_idx : band_idx + 1]
                    )

        # Forward pass
        y_hat = self.model(x)

        # Calculate supervised loss for each band
        band_losses = []

        # Target bands loss calculation
        for i, band_idx in enumerate(target_bands):
            y_hat_band = y_hat[:, i].contiguous()
            y_band = y[:, i].contiguous()
            band_loss = self.losses[band_idx](y_hat_band, y_band)
            band_losses.append(band_loss)
            self.log(f"train_loss_band_{band_idx}", band_loss)

        # Sum all losses without weighting
        total_loss = sum(band_losses)

        self.log("train_loss", total_loss)

        # Visualize training examples
        if (epoch < 8) and (batch_idx == 0):
            num_samples = min(10, x.shape[0])
            # First band is Overture, last 3 are RGB
            num_extra_bands = 1

            # Calculate number of columns: RGB + (target, pred) pairs for each band + Overture band
            n_cols = 1 + 2 * len(target_bands) + num_extra_bands

            fig, axs = plt.subplots(
                num_samples, n_cols, figsize=(5 * n_cols, 5 * num_samples)
            )

            try:
                for i in range(num_samples):

                    # Plot RGB input (last 3 bands)
                    img = np.rollaxis(x[i, -3:].cpu().numpy(), 0, 3)
                    axs[i, 0].imshow(np.clip(img, 0, 1))

                    # Plot target-prediction pairs for each band
                    for j, band_idx in enumerate(target_bands):

                        # Min-max normalize target
                        target = y[i, j].cpu().numpy()
                        pred = y_hat[i, j].detach().cpu().numpy()

                        # Set vmax based on band_idx for regression
                        if band_idx == 2:
                            vmax = 1.0
                        elif band_idx == 1:
                            vmax = 0.3
                        else:
                            vmax = 0.5
                        vmin = 0
                        cmap = "magma"

                        # visualize
                        axs[i, 2 * j + 1].imshow(
                            target,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            interpolation="none",
                        )
                        # Prediction
                        axs[i, 2 * j + 2].imshow(
                            pred,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            interpolation="none",
                        )

                    # Plot Overture band (first band) with min-max normalization
                    band = x[i, 0].cpu().numpy()
                    band_min, band_max = band.min(), band.max()
                    if band_max > band_min:
                        band = (band - band_min) / (band_max - band_min)
                    axs[i, 1 + 2 * len(target_bands)].imshow(
                        band, cmap="magma", interpolation="none"
                    )

                    # Turn off axes for all subplots in this row
                    for j in range(n_cols):
                        axs[i, j].axis("off")

                # Set titles for first row
                axs[0, 0].set_title("RGB Input")
                for j, band_idx in enumerate(target_bands):
                    axs[0, 2 * j + 1].set_title(f"Target Band {band_idx}")
                    axs[0, 2 * j + 2].set_title(f"Pred Band {band_idx}")
                axs[0, 1 + 2 * len(target_bands)].set_title("Overture")

                plt.tight_layout()
                self.logger.experiment.add_figure(
                    f"train_predictions_{epoch}", fig, self.global_step
                )
            finally:
                plt.close(fig)
                del fig, axs

        return total_loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_id: int
    ) -> None:
        """Run validation step."""
        img, mask = batch
        pred = self(img)
        target_bands = self.hparams.get("target_bands")

        # Calculate loss for each band
        band_losses = []

        for i, band_idx in enumerate(target_bands):
            y_hat_band = pred[:, i]
            y_band = mask[:, i]
            band_loss = self.losses[band_idx](y_hat_band, y_band)
            band_losses.append(band_loss)
            self.log(f"val_loss_band_{band_idx}", band_loss)

            # Calculate classification metrics for first target band only
            if i == 0:
                valid_data_mask = y_band >= 0
                y_band_valid = y_band[valid_data_mask]
                y_hat_band_valid = y_hat_band[valid_data_mask]

                # Regression mode with existing threshold
                threshold = 2 / 255
                pred_binary = (y_hat_band_valid > threshold).float()
                target_binary = (y_band_valid > threshold).float()

                # Calculate binary metrics (convert to boolean for bitwise operations)
                pred_bool = pred_binary.bool()
                target_bool = target_binary.bool()
                tps = (pred_bool & target_bool).sum()
                fps = (pred_bool & ~target_bool).sum()
                fns = (~pred_bool & target_bool).sum()

                precision = tps / (tps + fps + 1e-5)
                recall = tps / (tps + fns + 1e-5)
                f1 = (2 * precision * recall) / (precision + recall + 1e-5)

                self.log("val_precision", precision, sync_dist=True)
                self.log("val_recall", recall, sync_dist=True)
                self.log("val_f1", f1, sync_dist=True)

            # Calculate F1 score for 5 height ranges between ]0,1]
            if i == 1:
                valid_data_mask = y_band > 0
                y_band_valid = y_band[valid_data_mask]
                y_hat_band_valid = y_hat_band[valid_data_mask]

                height_ranges = [0, 0.03, 0.1, 0.2, 0.5, 1]
                for range_min, range_max in zip(height_ranges[:-1], height_ranges[1:]):
                    mask_ = (y_band_valid >= range_min) & (y_band_valid < range_max)
                    if torch.any(mask_):
                        y_band_valid_range = y_band_valid[mask_]
                        y_hat_band_valid_range = y_hat_band_valid[mask_]

                        tps = (
                            (y_band_valid_range >= range_min)
                            & (y_band_valid_range < range_max)
                            & (y_hat_band_valid_range >= range_min)
                            & (y_hat_band_valid_range < range_max)
                        ).sum()
                        fps = (
                            (
                                (y_band_valid_range < range_min)
                                | (y_band_valid_range >= range_max)
                            )
                            & (y_hat_band_valid_range >= range_min)
                            & (y_hat_band_valid_range < range_max)
                        ).sum()
                        fns = (
                            (y_band_valid_range >= range_min)
                            & (y_band_valid_range < range_max)
                            & (
                                (y_hat_band_valid_range < range_min)
                                | (y_hat_band_valid_range >= range_max)
                            )
                        ).sum()

                        precision = tps / (tps + fps + 1e-5)
                        recall = tps / (tps + fns + 1e-5)
                        f1 = (2 * precision * recall) / (precision + recall + 1e-5)
                        self.log(f"val_height_f1_{range_min}_{range_max}", f1)

        # Sum the losses without weighting for consistent validation metric (NaN-safe, FP32)
        valid_losses = [loss.float() for loss in band_losses if not torch.isnan(loss)]
        if len(valid_losses) > 0:
            total_val_loss = sum(valid_losses)
        else:
            total_val_loss = torch.tensor(0.0, device=self.device)
        self.log("val_loss", total_val_loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""

        # Only AdamW is supported
        beta1 = self.hparams.get("beta1")
        beta2 = self.hparams.get("beta2")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            betas=(beta1, beta2),
            amsgrad=True,
        )

        # Only cosine annealing is supported
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams["learning_rate_schedule_patience"],
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
