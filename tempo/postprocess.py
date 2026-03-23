# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Optional

from loguru import logger
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.io import MemoryFile
import rasterio.mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from scipy.stats import mode


# ---------------------------------------------------------------------------
# UDM era sets — used to select the correct band interpretation.
# First set (pre-2019q3): no UDM available.
# Second set (2019q3–2021q4): single-band UDM v1; band 0 == 0 means clear.
# Third set (2022q1+): multi-band UDM v2; band 0 = clear flag, band 6 = confidence.
# ---------------------------------------------------------------------------
_UDM_FIRST_SET = {
    "2018q1", "2018q2", "2018q3", "2018q4", "2019q1", "2019q2",
}
_UDM_SECOND_SET = {
    "2019q3", "2019q4",
    "2020q1", "2020q2", "2020q3", "2020q4",
    "2021q1", "2021q2", "2021q3", "2021q4",
}


def create_ranked_mask(
    udm: np.ndarray,
    confidence_threshold: int = 95,
    timestamp: Optional[str] = None,
) -> np.ndarray:
    """Create a ranked mask based on pixel clarity and confidence.

    The band interpretation depends on the UDM era:

    - **UDM v2** (2022q1 onwards, or ``timestamp=None``): band 1 is a clear
      flag (1 = clear), band 7 is a confidence map (0–100).
    - **UDM v1** (2019q3–2021q4): single-band; pixels equal to ``0`` indicate
      good/clear data. All pixels are treated as high-confidence.
    - **Pre-2019q3** timestamps: no UDM is available; passing one raises
      ``ValueError``.

    Parameters:
        udm: Array of shape ``(C, H, W)`` where C ≥ 1.
        confidence_threshold: UDM v2 Band 7 value above which a pixel is
            considered high-confidence (0–100, default 95).
        timestamp: Quarterly timestamp string, e.g. ``"2023q1"``.  Used to
            select v1 vs v2 band interpretation.  If ``None``, v2 is assumed.

    Returns:
        ranked_mask: Array of shape ``(H, W)``, integer ranks 1–4
            (4 = highest quality / most usable).

            - **4**: high confidence, clear
            - **3**: low confidence, clear
            - **2**: low confidence, not clear
            - **1**: high confidence, not clear
    """
    if timestamp is not None and timestamp in _UDM_FIRST_SET:
        raise ValueError(
            f"Timestamp {timestamp!r} predates UDM availability (earliest is 2019q3)."
        )

    if timestamp is not None and timestamp in _UDM_SECOND_SET:
        # UDM v1: band 0 == 0 means good/clear (inverted); treat all pixels as
        # high-confidence since v1 has no confidence band.
        clear_mask = (udm[0] == 0).astype(np.uint8)
        confidence_map = np.full(
            (udm.shape[1], udm.shape[2]), 100, dtype=np.uint8
        )
    else:
        # UDM v2 (or unspecified): band 1 = clear flag, band 7 = confidence.
        clear_mask = udm[0]
        confidence_map = udm[6]

    ranked_mask = np.full((udm.shape[1], udm.shape[2]), 4, dtype=np.uint8)
    ranked_mask[(confidence_map >= confidence_threshold) & (clear_mask == 0)] = 1
    ranked_mask[(confidence_map < confidence_threshold) & (clear_mask == 0)] = 2
    ranked_mask[(confidence_map < confidence_threshold) & (clear_mask == 1)] = 3
    ranked_mask[(confidence_map >= confidence_threshold) & (clear_mask == 1)] = 4

    return ranked_mask


def downsample_ranked_mask(ranked_mask: np.ndarray, algorithm: str = "mean") -> np.ndarray:
    """Downsample a (4096, 4096) ranked mask to (512, 512) via 8×8 block reduction.

    Parameters:
        ranked_mask: Array of shape (4096, 4096).
        algorithm: Reduction method — one of ``"mean"``, ``"max"``, ``"min"``,
            ``"mode"``.

    Returns:
        Downsampled array of shape (512, 512).
    """
    reshaped = ranked_mask.reshape(512, 8, 512, 8)
    if algorithm == "mean":
        return reshaped.mean(axis=(1, 3))
    if algorithm == "max":
        return reshaped.max(axis=(1, 3))
    if algorithm == "min":
        return reshaped.min(axis=(1, 3))
    if algorithm == "mode":
        mode_result = mode(reshaped, axis=(1, 3))
        return mode_result.mode.squeeze()
    raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose from mean, max, min, mode.")


def ensemble_pixels(
    model_outputs: np.ndarray,
    ranked_masks: np.ndarray,
    small_value_threshold: float,
    clarity_threshold: float,
    default_behavior: str,
) -> np.ndarray:
    """Vectorized ensemble across T timestamps for each pixel.

    Replaces the original (512, 512) nested Python loop with numpy operations,
    preserving identical semantics.

    Parameters:
        model_outputs: Shape (T, H, W) float32 — model predictions per timestamp.
        ranked_masks: Shape (T, H, W) float32 — downsampled UDM ranks per timestamp.
        small_value_threshold: Minimum value considered a building prediction.
        clarity_threshold: UDM rank cutoff above which a pixel is considered "clear".
        default_behavior: Aggregation over all T values for mixed-clarity 2-timestamp
            cases. One of ``"median"``, ``"mean"``, ``"max"``, ``"min"``.

    Returns:
        result: Shape (H, W) float32 ensemble output.
    """
    above = model_outputs > small_value_threshold        # (T, H, W)
    n_above = above.sum(axis=0)                          # (H, W)
    above_only = np.where(above, model_outputs, np.nan)  # NaN for non-building timestamps

    # Suppress all-NaN warnings: pixels where n_above == 0 produce all-NaN slices
    # in above_only, but those values are discarded by the final np.where.
    with np.errstate(all="ignore"):
        # Case A: >= 3 timestamps above threshold → median of building values
        case_a = np.nanmedian(above_only, axis=0)

        # Case C: exactly 2 timestamps above — apply UDM clarity logic
        n_clear_above = (above & (ranked_masks > clarity_threshold)).sum(axis=0)
        case_c1 = np.nanmax(above_only, axis=0)  # both clear → max of building values

    if default_behavior == "mean":
        case_c3 = np.mean(model_outputs, axis=0)
    elif default_behavior == "max":
        case_c3 = np.max(model_outputs, axis=0)
    elif default_behavior == "min":
        case_c3 = np.min(model_outputs, axis=0)
    else:  # median
        case_c3 = np.median(model_outputs, axis=0)

    case_c = np.where(
        n_clear_above == 2, case_c1,
        np.where(n_clear_above == 0, 0.0, case_c3),
    )

    result = np.where(
        n_above >= 3, case_a,
        np.where(n_above <= 1, 0.0, case_c),
    ).astype(np.float32)

    return result


def get_water_mask(
    aoi: dict,
    output_data: np.ndarray,
    output_profile: dict,
) -> Optional[np.ndarray]:
    """Query JRC-GSW via Planetary Computer STAC and return a permanent water mask.

    Transition values of 1 (permanent water) or 2 (new permanent water) are masked.

    Parameters:
        aoi: GeoJSON geometry dict for the scene's bounding box.
        output_data: Prediction array (H, W) — used for reprojection destination shape.
        output_profile: Rasterio profile of the prediction — used for reprojection.

    Returns:
        Boolean mask of shape (H, W), or None if no water data found.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = catalog.search(collections=["jrc-gsw"], intersects=aoi).item_collection()

    if len(items) == 0:
        logger.info("No water data found for AOI.")
        return None

    if len(items) == 1:
        with rasterio.open(items[0].assets["transitions"].href) as src:
            water_data, water_transform = rasterio.mask.mask(src, [aoi], crop=True, nodata=99)
            water_data = water_data[0]
            water_profile = src.profile.copy()
            water_profile.update({
                "transform": water_transform,
                "width": water_data.shape[1],
                "height": water_data.shape[0],
                "nodata": 99,
            })
    else:
        src_files = [rasterio.open(item.assets["transitions"].href) for item in items]
        mosaic, out_transform = merge(src_files)
        mosaic_profile = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": 1,
            "dtype": mosaic.dtype,
            "crs": src_files[0].crs,
            "transform": out_transform,
            "nodata": 99,
        }
        for src in src_files:
            src.close()
        with MemoryFile() as memfile:
            with memfile.open(**mosaic_profile) as src:
                src.write(mosaic)
                water_data, water_transform = rasterio.mask.mask(src, [aoi], crop=True, nodata=99)
                water_data = water_data[0]
                water_profile = src.profile.copy()
                water_profile.update({
                    "transform": water_transform,
                    "width": water_data.shape[1],
                    "height": water_data.shape[0],
                    "nodata": 99,
                })

    water_aligned = np.zeros(output_data.shape, dtype=rasterio.float32)
    water_reproj, _ = reproject(
        source=water_data,
        destination=water_aligned,
        src_transform=water_profile["transform"],
        src_crs=water_profile["crs"],
        dst_transform=output_profile["transform"],
        dst_crs=output_profile["crs"],
        resampling=Resampling.nearest,
    )
    water_mask = (water_reproj == 1) | (water_reproj == 2)

    del items, water_aligned, water_reproj
    return water_mask


def get_elevation_mask(
    aoi: dict,
    output_data: np.ndarray,
    output_profile: dict,
    threshold: int = 5100,
) -> Optional[np.ndarray]:
    """Query NASADEM via Planetary Computer STAC and return a high-elevation mask.

    Parameters:
        aoi: GeoJSON geometry dict for the scene's bounding box.
        output_data: Prediction array (H, W) — used for reprojection destination shape.
        output_profile: Rasterio profile of the prediction — used for reprojection.
        threshold: Elevation in meters above which pixels are masked (default 5100m).

    Returns:
        Boolean mask of shape (H, W), or None if no elevation data found.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = catalog.search(collections=["nasadem"], intersects=aoi).item_collection()

    if len(items) == 0:
        logger.info("No elevation data found for AOI.")
        return None

    if len(items) == 1:
        with rasterio.open(items[0].assets["elevation"].href) as src:
            elevation_data, elevation_transform = rasterio.mask.mask(
                src, [aoi], crop=True, nodata=-9999
            )
            elevation_data = elevation_data[0]
            elevation_profile = src.profile.copy()
            elevation_profile.update({
                "transform": elevation_transform,
                "width": elevation_data.shape[1],
                "height": elevation_data.shape[0],
                "nodata": -9999,
            })
    else:
        src_files = [rasterio.open(item.assets["elevation"].href) for item in items]
        mosaic, out_transform = merge(src_files)
        mosaic_profile = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": 1,
            "dtype": mosaic.dtype,
            "crs": src_files[0].crs,
            "transform": out_transform,
        }
        for src in src_files:
            src.close()
        with MemoryFile() as memfile:
            with memfile.open(**mosaic_profile) as src:
                src.write(mosaic)
                elevation_data, elevation_transform = rasterio.mask.mask(
                    src, [aoi], crop=True, nodata=-9999
                )
                elevation_data = elevation_data[0]
                elevation_profile = src.profile.copy()
                elevation_profile.update({
                    "transform": elevation_transform,
                    "width": elevation_data.shape[1],
                    "height": elevation_data.shape[0],
                    "nodata": -9999,
                })

    elevation_aligned = np.zeros(output_data.shape, dtype=rasterio.float32)
    elevation_reproj, _ = reproject(
        source=elevation_data,
        destination=elevation_aligned,
        src_transform=elevation_profile["transform"],
        src_crs=elevation_profile["crs"],
        dst_transform=output_profile["transform"],
        dst_crs=output_profile["crs"],
        resampling=Resampling.nearest,
    )
    elevation_mask = elevation_reproj >= threshold

    del items, elevation_aligned, elevation_reproj
    return elevation_mask
