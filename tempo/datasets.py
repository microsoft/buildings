# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import time
import numpy as np
import rasterio
import rasterio.windows
import torch
from torch.utils.data import Dataset


def stack_samples(samples):
    batch_size = len(samples)

    # Extract first sample to determine shapes - assume (C, H, W) format
    sample_img, sample_mask = samples[0]

    # Pre-allocate tensors
    img_batch = torch.zeros((batch_size, *sample_img.shape), dtype=sample_img.dtype)
    mask_batch = torch.zeros((batch_size, *sample_mask.shape), dtype=sample_mask.dtype)

    # Fill pre-allocated tensors
    for i, (img, mask) in enumerate(samples):
        img_batch[i] = img
        mask_batch[i] = mask

    return img_batch, mask_batch


class TileDataset(Dataset):

    def __init__(
        self,
        image_fns,
        overture_fns,
        mask_fns,
        patch_size,
        tile_size,
        transforms,
        downsample_factor,
    ):
        self.image_fns = image_fns
        self.overture_fns = overture_fns
        self.mask_fns = mask_fns
        assert len(image_fns) == len(overture_fns) == len(mask_fns)
        self.downsample_factor = downsample_factor
        self.patch_size = patch_size
        self.tile_size = tile_size
        self.transforms = transforms
        self.patch_size_downsampled = self.patch_size // self.downsample_factor

    def __len__(self):
        return len(self.image_fns)

    def _get_dataset(self, path):
        is_remote = path.startswith("http")
        max_retries = 3 if is_remote else 1

        for attempt in range(max_retries):
            try:
                return rasterio.open(path)
            except Exception as e:
                if is_remote and attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"Retry {attempt + 1}/{max_retries} opening {path} after {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    if is_remote:
                        print(
                            f"Failed to open {path} after {max_retries} retries, returning None: {e}"
                        )
                        return None
                    else:
                        raise RuntimeError(
                            f"Failed to open dataset at {path}: {e}"
                        ) from e

        if is_remote:
            return None
        raise RuntimeError(f"Failed to open dataset at {path}")

    def _calculate_scaled_window(self, dataset, target_window):
        """Calculate the appropriate window for a dataset based on its resolution relative to the tile size."""
        if dataset is None:
            return target_window

        # Get dataset dimensions
        dataset_height, dataset_width = dataset.height, dataset.width

        # Calculate the scaling factor between dataset and tile size
        scale_y = dataset_height / self.tile_size[0]
        scale_x = dataset_width / self.tile_size[1]

        # Scale the target window to match this dataset's resolution
        return rasterio.windows.Window(
            int(target_window.col_off * scale_x),
            int(target_window.row_off * scale_y),
            int(target_window.width * scale_x),
            int(target_window.height * scale_y),
        )

    def _read_with_retry(
        self, path, dataset, window, bands=None, indexes=None, target_shape=None
    ):
        if dataset is None:
            if target_shape is not None:
                return np.zeros(target_shape, dtype=np.float32)
            else:
                raise RuntimeError(
                    f"Dataset is None and no target_shape provided for {path}"
                )

        is_remote = path.startswith("http")
        max_retries = 3 if is_remote else 1

        for attempt in range(max_retries):
            try:
                if bands is not None:
                    return dataset.read(bands, window=window)
                elif indexes is not None:
                    return dataset.read(indexes=indexes, window=window)
                else:
                    return dataset.read(window=window)
            except Exception as e:
                if is_remote and attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"Retry {attempt + 1}/{max_retries} for {path} after {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    if is_remote and target_shape is not None:
                        print(
                            f"Failed to read {path} after {max_retries} retries, returning zeros: {e}"
                        )
                        return np.zeros(target_shape, dtype=np.float32)
                    else:
                        raise

        if is_remote and target_shape is not None:
            return np.zeros(target_shape, dtype=np.float32)
        raise RuntimeError(f"Failed to read {path}")

    def get_random_window(self):

        # Calculate maximum valid starting positions
        max_y = self.tile_size[0] - self.patch_size
        max_x = self.tile_size[1] - self.patch_size

        # Randomly sample any valid window position
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)

        return y, x

    def load_and_interpolate_extra(self, fn, target_window, target_size):

        ds = None
        try:
            ds = self._get_dataset(fn)

            # If dataset failed to open, use default single-band shape
            if ds is None:
                target_shape = (1, target_size[0], target_size[1])
                band_data = np.zeros(target_shape, dtype=np.float32)
                return band_data

            # Calculate the appropriate window for this extra band based on its resolution
            scaled_window = self._calculate_scaled_window(ds, target_window)

            # Overture is always single band
            target_shape = (1, target_size[0], target_size[1])

            # Read the single band
            band_data = self._read_with_retry(
                fn, ds, scaled_window, bands=[1], target_shape=target_shape
            )
            band = torch.from_numpy(band_data).float()

            # Close the reader
            if ds is not None:
                ds.close()

            # Only interpolate if the current size doesn't match target size
            current_size = (band.shape[-2], band.shape[-1])
            if current_size != target_size:
                with torch.no_grad():
                    interpolated = torch.nn.functional.interpolate(
                        band.unsqueeze(0),  # Add batch dimension
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    result = interpolated.squeeze(0).numpy()  # Remove batch dimension
                    del band, interpolated
                    return result
            else:
                # No interpolation needed - same resolution
                return band.numpy()

        except Exception as e:
            raise Exception(f"Error reading {fn}: {e}")

    def __getitem__(self, idx):

        # If idx is a tuple, use the provided coordinates (GeoGridSampler)
        # Otherwise, generate random coordinates (RandomTileSampler)
        if isinstance(idx, tuple):
            i, y, x = idx
        else:
            i = idx
            y, x = self.get_random_window()

        assert 0 <= i < len(self.image_fns)

        window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)

        # Get the image URL
        image_path = self.image_fns[i]

        # Load RGB with explicit file handle management
        ds_rgb = None
        try:
            ds_rgb = self._get_dataset(image_path)
            target_shape = (3, self.patch_size, self.patch_size)
            rgb_image = self._read_with_retry(
                image_path, ds_rgb, window, indexes=[1, 2, 3], target_shape=target_shape
            )
            if ds_rgb is not None:
                ds_rgb.close()
        except Exception as e:
            if ds_rgb is not None:
                ds_rgb.close()
            raise RuntimeError(
                f"Failed to load RGB image from {image_path}: {e}"
            ) from e

        # Load single Overture band
        target_size = (rgb_image.shape[1], rgb_image.shape[2])
        overture_path = self.overture_fns[i]
        overture_band = self.load_and_interpolate_extra(
            overture_path, window, target_size
        )

        # Stack: Overture band first, then RGB
        stack = np.concatenate([overture_band, rgb_image], axis=0).astype(np.float32)
        del rgb_image, overture_band

        img = torch.from_numpy(stack)
        del stack

        # Load mask with explicit error handling
        mask = None
        ds_mask = None
        try:
            ds_mask = self._get_dataset(self.mask_fns[i])
            # If dataset is None, use default single-channel mask
            if ds_mask is None:
                target_shape = (
                    1,
                    self.patch_size_downsampled,
                    self.patch_size_downsampled,
                )
                mask_data = np.zeros(target_shape, dtype=np.float32)
            else:
                # Calculate appropriate window for mask based on its resolution
                mask_window = self._calculate_scaled_window(ds_mask, window)
                target_shape = (
                    ds_mask.count,
                    self.patch_size_downsampled,
                    self.patch_size_downsampled,
                )
                mask_data = self._read_with_retry(
                    self.mask_fns[i],
                    ds_mask,
                    mask_window,
                    target_shape=target_shape,
                )
                ds_mask.close()
        except Exception as e:
            if ds_mask is not None:
                ds_mask.close()
            raise RuntimeError(
                f"Failed to load mask from {self.mask_fns[i]}: {e}"
            ) from e
        mask = torch.from_numpy(mask_data)

        if self.transforms is not None:
            img, mask = self.transforms((img, mask))

        return img, mask
