# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
import warnings

import numpy as np
import rasterio
import rasterio.transform
import torch
from torch.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from tempo.trainers import SegmentationTask

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def collate_batch(samples):
    """Collate samples into a batch, filtering out None values."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None

    batch = {}
    for key in samples[0].keys():
        values = [s[key] for s in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


class Preprocesser:
    """Normalize input imagery using band-specific min/max values."""

    def __init__(self, band_normalizers):
        self.band_normalizers = band_normalizers

    def __call__(self, sample):
        img = sample["image"]

        num_bands = img.shape[0]
        if len(self.band_normalizers) != num_bands:
            raise ValueError(
                f"Number of band normalizers ({len(self.band_normalizers)}) "
                f"must match number of bands ({num_bands})"
            )

        min_vals = torch.tensor(
            [min_val for min_val, _ in self.band_normalizers],
            device=img.device,
        )
        max_vals = torch.tensor(
            [max_val for _, max_val in self.band_normalizers],
            device=img.device,
        )

        img = (img - min_vals[:, None, None]) / (
            max_vals[:, None, None] - min_vals[:, None, None]
        )
        img = torch.clamp(img, min=0).float()

        return {
            "image": img,
            "rgb_path": sample["rgb_path"],
            "bounds": sample["bounds"],
        }


class InferenceDataset(Dataset):
    """Dataset for loading RGB imagery with Overture building priors.

    Each line in the input file should be formatted as:
        <overture_path>,<rgb_path>
    """

    def __init__(self, input_file, transforms=None):
        self.transforms = transforms

        # Read input file paths
        with open(input_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parts = self.samples[idx].split(",")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid line format: {self.samples[idx]}. "
                f"Expected format: <overture_path>,<rgb_path>"
            )

        overture_path, rgb_path = parts

        # Load RGB imagery (first 3 bands only)
        with rasterio.open(rgb_path) as f:
            bounds = tuple(f.bounds)
            rgb = f.read([1, 2, 3]).astype(np.float32)
            rgb = np.nan_to_num(rgb, nan=0.0)

        # Load Overture prior and resize to match RGB dimensions
        with rasterio.open(overture_path) as f:
            overture = f.read(1).astype(np.float32)  # Shape: (H, W)
            overture = np.nan_to_num(overture, nan=0.0)

            # Resize Overture to match RGB spatial dimensions if needed
            if overture.shape != rgb.shape[1:]:
                # Add channel and batch dimensions for interpolation: (1, 1, H, W)
                overture_tensor = torch.from_numpy(overture[None, None, :, :]).float()
                with torch.no_grad():
                    interpolated = torch.nn.functional.interpolate(
                        overture_tensor,
                        size=rgb.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Remove batch and channel dimensions: (H, W)
                    overture = interpolated[0, 0].numpy()

        # Stack: Overture, R, G, B (4 channels)
        # Add channel dimension to overture: (1, H, W)
        image = np.vstack([overture[None, :, :], rgb])

        sample = {
            "image": torch.from_numpy(image),
            "rgb_path": rgb_path,
            "bounds": bounds,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def set_up_parser():
    parser = argparse.ArgumentParser(
        description="Run inference on satellite imagery to predict building density and height."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to model checkpoint (.ckpt format)",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=str,
        help="Text file with image paths (one per line, format: 'overture_path,rgb_path')",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save prediction outputs",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use for inference (default: use CPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    return parser


def main(args):
    # Set up device
    device = torch.device(
        f"cuda:{args.gpu}"
        if args.gpu is not None and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    task = SegmentationTask.load_from_checkpoint(
        args.checkpoint, strict=False, map_location=device
    )
    task.freeze()
    model = task.model.eval()

    # Get hyperparameters from checkpoint
    hps = dict(task.hparams)

    if "band_normalizers" not in hps or hps["band_normalizers"] is None:
        raise ValueError(
            "Model checkpoint must contain 'band_normalizers' configuration. "
            "Please retrain your model with the updated config."
        )

    num_channels = len(hps["band_normalizers"])
    print(f"Model expects {num_channels} input channels")

    # Create dataset and dataloader
    preprocess = Preprocesser(band_normalizers=hps["band_normalizers"])
    transforms = Compose([preprocess])
    dataset = InferenceDataset(args.input_file, transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    print(f"Running inference on {len(dataset)} images...")

    processed = 0
    failed = 0

    for batch in tqdm(dataloader, desc="Processing"):
        if batch is None:
            continue

        images = batch["image"].to(device)
        rgb_paths = batch["rgb_path"]
        bounds = batch["bounds"]

        try:
            with torch.inference_mode(), autocast(
                "cuda" if device.type == "cuda" else "cpu"
            ):
                predictions = model(images).cpu().numpy()

            # Handle single-sample batches
            if len(predictions.shape) == 2:
                predictions = predictions[None, :, :]

            for i in range(len(images)):
                try:
                    output = predictions[i]
                    rgb_path = rgb_paths[i]
                    west, south, east, north = bounds[i]

                    # Create output filename based on input
                    input_name = Path(rgb_path).stem
                    output_path = output_dir / f"{input_name}.tif"

                    # Create GeoTIFF profile
                    profile = {
                        "driver": "GTiff",
                        "crs": "EPSG:3857",
                        "count": output.shape[0],
                        "height": output.shape[1],
                        "width": output.shape[2],
                        "dtype": "float32",
                        "compress": "lzw",
                        "predictor": 3,
                        "nodata": -99,
                        "tiled": True,
                        "blockxsize": 256,
                        "blockysize": 256,
                        "transform": rasterio.transform.from_bounds(
                            west, south, east, north, output.shape[2], output.shape[1]
                        ),
                    }

                    # Write output
                    with rasterio.open(output_path, "w", **profile) as dst:
                        for band_idx in range(output.shape[0]):
                            dst.write(output[band_idx], band_idx + 1)

                    processed += 1

                except Exception as e:
                    print(f"Error saving output for {rgb_paths[i]}: {e}")
                    failed += 1

        except Exception as e:
            print(f"Error during inference: {e}")
            failed += len(images)

    print(f"\nInference complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
