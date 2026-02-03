# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional
from tqdm import tqdm

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from tempo.datasets import TileDataset, stack_samples
from tempo.samplers import GridGeoSampler, RandomTileSampler


class Preprocesser:
    def __init__(
        self,
        band_normalizers,
        target_bands,
        target_normalizers,
        nodata=-99,
    ):
        self.band_normalizers = band_normalizers
        self.target_bands = target_bands or [2]
        self.target_normalizers = target_normalizers or [1.0]
        self.nodata = nodata

        if len(self.target_bands) != len(self.target_normalizers):
            raise ValueError(
                "Number of target normalizers must match number of target bands"
            )

    def __call__(self, sample):
        img, mask = sample

        # Use band_normalizers for ALL bands (no RGB assumption)
        min_vals = torch.tensor(
            [min_val for min_val, _ in self.band_normalizers],
            device=img.device,
        )
        max_vals = torch.tensor(
            [max_val for _, max_val in self.band_normalizers],
            device=img.device,
        )

        # Apply normalization
        img = (img - min_vals[:, None, None]) / (
            max_vals[:, None, None] - min_vals[:, None, None]
        )
        img = img.float()

        if mask is not None:
            target_mask = mask[self.target_bands]
            normalizers = torch.tensor(self.target_normalizers, device=mask.device)

            # Create valid data mask (non-nodata pixels)
            valid_mask = target_mask != self.nodata

            # Only normalize valid data
            normalized_mask = target_mask.clone().float()
            normalized_mask[valid_mask] = (
                normalized_mask[valid_mask]
                / normalizers[:, None, None].expand_as(normalized_mask)[valid_mask]
            )
            mask = normalized_mask.float()

        return img, mask


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_fns: Dict[str, List[str]],
        overture_fns: Dict[str, List[str]],
        mask_fns: Dict[str, List[str]],
        fn_sample_weights: Optional[dict[str, float]],
        batch_size,
        patch_size,
        num_workers,
        train_batches_per_epoch,
        valid_batches_per_epoch,
        band_normalizers,
        target_bands,
        target_normalizers,
        **kwargs: Any,
    ) -> None:
        """Initialize the SegmentationDataModule.

        Args:
            image_fns: dictionary of lists of image filenames for train, valid, and test.
            overture_fns: dictionary of lists of overture filenames for train, valid, and test.
            mask_fns: dictionary of lists of mask filenames for train, valid, and test.
            fn_sample_weights: dictionary of sample weights for each mask filename.
            batch_size: number of samples per batch.
            patch_size: size of patches to extract from images.
            num_workers: number of workers for the DataLoader.
            train_batches_per_epoch: number of batches per epoch for training.
            valid_batches_per_epoch: number of batches per epoch for validation.
        """
        super().__init__()
        self.image_fns = image_fns
        self.overture_fns = overture_fns
        self.mask_fns = mask_fns
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_patches_per_epoch = train_batches_per_epoch * batch_size
        self.valid_patches_per_epoch = valid_batches_per_epoch * batch_size
        self.band_normalizers = band_normalizers
        self.target_bands = target_bands
        self.target_normalizers = target_normalizers

        self.weights = {
            "train": None,
            "valid": None,
            "test": None,
        }
        if fn_sample_weights is not None:
            self.weights = {
                "train": [],
                "valid": [],
                "test": [],
            }
            for key, fns in tqdm(self.mask_fns.items()):
                for fn in fns:
                    self.weights[key].append(
                        fn_sample_weights[str(fn).split(".tif")[0].split("/")[-1]]
                    )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """

        preprocesser = Preprocesser(
            band_normalizers=self.band_normalizers,
            target_bands=self.target_bands,
            target_normalizers=self.target_normalizers,
        )
        transforms = Compose([preprocesser])

        self.train_dataset = TileDataset(
            self.image_fns["train"],
            self.overture_fns["train"],
            self.mask_fns["train"],
            transforms=transforms,
            patch_size=self.patch_size,
            tile_size=(4096, 4096),
            downsample_factor=8,
        )

        self.val_dataset = TileDataset(
            self.image_fns["valid"],
            self.overture_fns["valid"],
            self.mask_fns["valid"],
            transforms=transforms,
            patch_size=self.patch_size,
            tile_size=(4096, 4096),
            downsample_factor=8,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        train_sampler = RandomTileSampler(
            image_fns=self.image_fns["train"],
            length=self.train_patches_per_epoch,
            weights=self.weights["train"],
        )

        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        sampler = GridGeoSampler(
            list(range(len(self.image_fns["valid"]))),
            4096,
            4096,
        )

        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_size=16,
            num_workers=16,
            collate_fn=stack_samples,
            persistent_workers=True,
            pin_memory=True,
        )
