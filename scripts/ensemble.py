# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Ensemble building density and height predictions across quarterly timestamps.

Reads per-scene model prediction TIFFs and UDM quality masks from blob storage,
applies a vectorized multi-timestamp ensemble, then optionally zeros out permanent
water (burned as -1) and high-elevation pixels. Outputs a 2-band float32 GeoTIFF
per scene: band 1 = density, band 2 = height.

Required environment variables (loaded from .env automatically):
    OUTPUT_SAS  — SAS token for the label prediction container.
    UDM_SAS     — SAS token for the UDM container.
    BLOB_SAS    — SAS token for the pre-computed water/elevation raster container.
                  If unset, water and elevation masks fall back to Planetary Computer
                  STAC queries.

Quick start (single machine):

    python scripts/ensemble.py \\
        --input-csv data/scenes.csv \\
        --model <model-name> \\
        --save-dir outputs/ \\
        --save-folder <run-name> \\
        --timestamps 2024q1 2024q2 2024q3 2024q4 \\
        --workers 32 \\
        --skip-existing

External storage (supply your own URL templates):

    python scripts/ensemble.py \\
        --label-url-template "https://mystorage.blob.core.windows.net/preds/{timestamp}/{model}/{scene}.tif?{sas}" \\
        --udm-url-template   "https://mystorage.blob.core.windows.net/udm/{timestamp}/{scene}_udm.tif?{sas}" \\
        --input-csv scenes.csv --model <model-name> --save-dir out/ --save-folder <run-name> \\
        --timestamps 2024q1 2024q2 2024q3 2024q4

AML at scale (submit N parallel jobs via submit_ensemble.py):

    python scripts/submit_ensemble.py \\
        --input-csv data/scenes.csv \\
        --model <model-name> \\
        --save-folder <run-name> \\
        --timestamps 2024q1 2024q2 2024q3 2024q4 \\
        --num-shards 128 \\
        --workers 32 \\
        --aml-output-path "azureml://datastores/<datastore>/paths/<prefix>/"
"""

import os

# Must be set before rasterio is imported so GDAL picks them up.
os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
    "GDAL_SWATH_SIZE": "200000000",
    "VSI_CURL_CACHE_SIZE": "200000000",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_USE_HEAD": "No",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
})

import argparse
import concurrent.futures
from pathlib import Path
import time
import warnings

from dotenv import load_dotenv

load_dotenv()

from loguru import logger
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, box
from tqdm import tqdm

from tempo.postprocess import (
    create_ranked_mask,
    downsample_ranked_mask,
    ensemble_pixels,
    get_water_mask,
    get_elevation_mask,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Internal URL builders (Microsoft blob storage layout).
# External users should pass --label-url-template / --udm-url-template instead.
# ---------------------------------------------------------------------------

_INTERNAL_LABEL_TEMPLATE = (
    "https://researchlabwuplanetstg.blob.core.windows.net/outputs"
    "/predictions/{timestamp}/{model}/{scene}.tif?{sas}"
)

# Pre-computed static rasters (water / elevation) stored in ihme-population-mapping.
_INTERNAL_WATER_TEMPLATE = (
    "https://researchlabwusstorage.blob.core.windows.net/ihme-population-mapping"
    "/datasets/permanent_or_seasonal_water/{scene}.tif?{sas}"
)
_INTERNAL_ELEVATION_TEMPLATE = (
    "https://researchlabwusstorage.blob.core.windows.net/ihme-population-mapping"
    "/datasets/nasadem_elevation_above_5100/{scene}.tif?{sas}"
)

_UDM_SECOND_SET = {
    "2019q3", "2019q4",
    "2020q1", "2020q2", "2020q3", "2020q4",
    "2021q1", "2021q2", "2021q3", "2021q4",
}
_UDM_THIRD_SET = {
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
}


def _internal_udm_url(scene: str, timestamp: str, sas: str) -> str:
    """Return the internal UDM blob URL, selecting UDM v1 or v2 by timestamp era.

    Returns an empty string for timestamps predating UDM availability.
    """
    if timestamp in _UDM_SECOND_SET:
        return (
            f"https://researchlabwuplanetstg.blob.core.windows.net/udm"
            f"/global_quarterly_{timestamp}_mosaic/udm/{scene}_ortho_udm.tif?{sas}"
        )
    if timestamp in _UDM_THIRD_SET:
        return (
            f"https://researchlabwuplanetstg.blob.core.windows.net/udm"
            f"/global_quarterly_{timestamp}_mosaic/udm2/{scene}_ortho_udm2.tif?{sas}"
        )
    return ""


def process_scene(
    scene: str,
    timestamps: list,
    model: str,
    save_fp: Path,
    output_sas: str,
    udm_sas: str,
    blob_sas: str = None,
    label_url_template: str = None,
    udm_url_template: str = None,
    udm_url_template_v2: str = None,
    udm_v2_cutoff: str = "2022q1",
    water_url_template: str = None,
    elevation_url_template: str = None,
    skip_existing: bool = False,
    confidence_threshold: int = 95,
    averaging_algorithm: str = "mean",
    small_value_threshold: float = 2 / 255,
    small_height_threshold: float = 0.024,
    clarity_threshold: float = 3.5,
    default_behavior: str = "median",
    elevation_threshold: int = 5100,
    water: bool = True,
    elevation: bool = True,
    stac_fallback: bool = False,
) -> None:
    """Run the full ensemble + post-processing pipeline for a single scene.

    Reads label TIFFs (band 1 = density, band 2 = height) and UDM masks for
    each timestamp, ensembles each band independently, applies water and
    elevation masks, enforces density–height consistency, then writes a 2-band
    float32 GeoTIFF.

    **Timestamp handling**

    - **4 timestamps available**: full ensemble logic (vectorised across T).
    - **1–3 timestamps available**: the most recent successfully-loaded
      prediction is used as-is (no ensembling).
    - **0 timestamps available**: scene is skipped.

    **Masking**

    - Water pixels are burned to ``-1`` (not 0) so downstream tools can
      distinguish missing-data from non-building.  ``profile["nodata"]`` is
      set to ``-1`` in the output.
    - Elevation pixels above *elevation_threshold* are set to ``0.0``.

    **Water / elevation raster source**

    Pre-computed binary rasters are used when available (preferred):

    1. If *water_url_template* / *elevation_url_template* are supplied, those
       are used.
    2. Else if *blob_sas* is supplied, the internal Microsoft blob storage
       templates are used.
    3. Otherwise, Planetary Computer STAC queries are used as a fallback.

    **URL templates**

    *label_url_template* and *udm_url_template* support the placeholders
    ``{scene}``, ``{timestamp}``, ``{model}``, ``{sas}``.
    Water/elevation templates support ``{scene}`` and ``{sas}``.

    Args:
        scene: Scene identifier (matches the ``scene`` column in the input CSV).
        timestamps: Ordered list of quarterly timestamps, e.g.
            ``["2023q3", "2023q4", "2024q1", "2024q2"]``.
        model: Model folder name used to resolve label URLs, e.g.
            ``"9-37-best_practices_p3"``.
        save_fp: Full output path for the ensembled float32 GeoTIFF.
        output_sas: SAS token for the label prediction container.
        udm_sas: SAS token for the UDM container.
        blob_sas: SAS token for the pre-computed water/elevation raster
            container.  If ``None`` and no explicit URL templates are given,
            masking falls back to STAC.
        label_url_template: Optional URL template for label TIFFs.
            Placeholders: ``{scene}``, ``{timestamp}``, ``{model}``, ``{sas}``.
        udm_url_template: URL template for UDM TIFFs (primary / older era).
        udm_url_template_v2: URL template for UDM TIFFs (newer era, for
            timestamps >= ``udm_v2_cutoff``).  Falls back to
            ``udm_url_template`` if omitted.
        udm_v2_cutoff: First timestamp that should use ``udm_url_template_v2``
            (default ``"2022q1"``).
        water_url_template: URL template for pre-computed water rasters.
            Placeholders: ``{scene}``, ``{sas}``.
        elevation_url_template: URL template for pre-computed elevation rasters.
            Placeholders: ``{scene}``, ``{sas}``.
        skip_existing: If ``True`` and ``save_fp`` already exists, return
            immediately.
        confidence_threshold: UDM Band 7 value above which a pixel is
            high-confidence (0–100, default 95).
        averaging_algorithm: Method for 8×8 UDM block downsampling.
            One of ``"mean"``, ``"max"``, ``"min"``, ``"mode"``.
        small_value_threshold: Density values at or below this are treated as
            non-building (default ``2/255``).
        small_height_threshold: Height values at or below this are treated as
            non-building (default ``0.024`` ≈ 3 m, anything smaller is
            unlikely to be a real building).
        clarity_threshold: UDM rank cutoff for "clear" pixel classification
            (default 3.5).
        default_behavior: Aggregation for mixed-clarity 2-timestamp pixels.
            One of ``"median"``, ``"mean"``, ``"max"``, ``"min"``.
        elevation_threshold: NASADEM elevation (m) above which pixels are
            zeroed (default 5100 m).
        water: If ``True``, apply permanent-water mask.
        elevation: If ``True``, apply high-elevation mask.
    """
    if skip_existing and save_fp.exists():
        logger.info(f"Skipping {scene} — output already exists at {save_fp}.")
        return

    start_time = time.time() 
    current_url = None
    try:
        # ------------------------------------------------------------------
        # Load label TIFFs + UDMs — all timestamps read concurrently.
        # ------------------------------------------------------------------
        def _read_timestamp(timestamp):
            """Read label + UDM for one timestamp. Returns (model_out, mask, profile, aoi)."""
            if label_url_template:
                label_url = label_url_template.format(
                    scene=scene, timestamp=timestamp, model=model, sas=output_sas
                )
            else:
                label_url = _INTERNAL_LABEL_TEMPLATE.format(
                    scene=scene, timestamp=timestamp, model=model, sas=output_sas
                )
            logger.debug(f"{scene} {timestamp}: opening label {label_url}")
            try:
                with rasterio.open(label_url) as src:
                    density = src.read(1).astype(np.float32)
                    height = src.read(2).astype(np.float32) if src.count >= 2 else None
                    ts_profile = src.profile.copy()
                    bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    ts_aoi = mapping(box(*bounds))
                model_out = (density, height)
            except Exception as e:
                logger.warning(f"{scene} {timestamp}: failed to read label — {e}")
                return None, None, None, None

            if udm_url_template:
                if udm_url_template_v2 and timestamp >= udm_v2_cutoff:
                    udm_url = udm_url_template_v2.format(
                        scene=scene, timestamp=timestamp, sas=udm_sas
                    )
                else:
                    udm_url = udm_url_template.format(
                        scene=scene, timestamp=timestamp, sas=udm_sas
                    )
            else:
                udm_url = _internal_udm_url(scene, timestamp, udm_sas)

            if udm_url:
                logger.debug(f"{scene} {timestamp}: opening UDM  {udm_url}")
                try:
                    with rasterio.open(udm_url) as src:
                        udm = src.read()
                    ranked_mask = create_ranked_mask(
                        udm, confidence_threshold=confidence_threshold, timestamp=timestamp,
                    )
                    downsampled = downsample_ranked_mask(ranked_mask, algorithm=averaging_algorithm)
                except Exception as e:
                    logger.warning(f"{scene} {timestamp}: failed to read UDM — {e}. Using default clarity.")
                    downsampled = np.full(density.shape, 4.0, dtype=np.float32)
            else:
                logger.warning(f"{scene} {timestamp}: no UDM available, using default clarity.")
                downsampled = np.full(density.shape, 4.0, dtype=np.float32)

            return model_out, downsampled, ts_profile, ts_aoi

        ts_results = [None] * len(timestamps)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(timestamps)) as ts_pool:
            fs = {ts_pool.submit(_read_timestamp, ts): i for i, ts in enumerate(timestamps)}
            for f in concurrent.futures.as_completed(fs):
                ts_results[fs[f]] = f.result()

        model_outputs = []
        ranked_masks = []
        profile = None
        aoi = None
        for model_out, downsampled, ts_profile, ts_aoi in ts_results:
            model_outputs.append(model_out)
            ranked_masks.append(downsampled)
            if model_out is not None and profile is None:
                profile = ts_profile
                aoi = ts_aoi

        # ------------------------------------------------------------------
        # Determine how many timestamps loaded successfully.
        # ------------------------------------------------------------------
        available_indices = [j for j, mo in enumerate(model_outputs) if mo is not None]

        if len(available_indices) == 0:
            logger.warning(f"{scene}: no predictions available. Skipping.")
            return

        has_height = all(model_outputs[j][1] is not None for j in available_indices)

        if len(available_indices) < 4:
            # Fewer than 4 timestamps — use the most recent available prediction as-is.
            logger.info(
                f"{scene}: only {len(available_indices)}/4 timestamps available "
                f"— using most recent prediction directly."
            )
            last_idx = available_indices[-1]
            result = model_outputs[last_idx][0].copy()
            result_height = model_outputs[last_idx][1].copy() if has_height else None
        else:
            # Full 4-timestamp ensemble.
            densities = np.stack([model_outputs[j][0] for j in available_indices][::-1])
            masks_arr = np.stack([ranked_masks[j] for j in available_indices][::-1])

            result = ensemble_pixels(
                densities, masks_arr,
                small_value_threshold=small_value_threshold,
                clarity_threshold=clarity_threshold,
                default_behavior=default_behavior,
            )
            result[result < small_value_threshold] = 0.0

            if has_height:
                heights = np.stack([model_outputs[j][1] for j in available_indices][::-1])
                result_height = ensemble_pixels(
                    heights, masks_arr,
                    small_value_threshold=small_height_threshold,
                    clarity_threshold=clarity_threshold,
                    default_behavior=default_behavior,
                )
                result_height[result_height < small_height_threshold] = 0.0
            else:
                result_height = None

        # ------------------------------------------------------------------
        # Water + elevation masks — loaded concurrently.
        # ------------------------------------------------------------------
        water_mask = None
        elevation_mask = None

        def _load_water():
            return _load_mask(
                scene=scene, mask_type="water", blob_sas=blob_sas,
                url_template=water_url_template, internal_template=_INTERNAL_WATER_TEMPLATE,
                aoi=aoi, result=result, profile=profile, elevation_threshold=None,
                stac_fallback=stac_fallback,
            )

        def _load_elevation():
            return _load_mask(
                scene=scene, mask_type="elevation", blob_sas=blob_sas,
                url_template=elevation_url_template, internal_template=_INTERNAL_ELEVATION_TEMPLATE,
                aoi=aoi, result=result, profile=profile, elevation_threshold=elevation_threshold,
                stac_fallback=stac_fallback,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as mask_pool:
            f_water     = mask_pool.submit(_load_water)     if water     else None
            f_elevation = mask_pool.submit(_load_elevation) if elevation else None
            if f_water is not None:
                water_mask = f_water.result()
            if f_elevation is not None:
                elevation_mask = f_elevation.result()

        if water_mask is not None:
            logger.info(f"{scene}: applying water mask ({water_mask.sum():,} px).")
            result[water_mask] = -1.0
            if result_height is not None:
                result_height[water_mask] = -1.0

        if elevation_mask is not None:
            logger.info(f"{scene}: applying elevation mask ({elevation_mask.sum():,} px).")
            result[elevation_mask] = 0.0
            if result_height is not None:
                result_height[elevation_mask] = 0.0

        # ------------------------------------------------------------------
        # Step 5: enforce density–height consistency.
        # If density > 0, height must be at least small_height_threshold.
        # If density == 0, height must also be 0.
        # If height > 0, density must be at least small_value_threshold.
        # ------------------------------------------------------------------
        if result_height is not None:
            result_height[result > 0] = np.maximum(
                result_height[result > 0], small_height_threshold
            )
            result_height[result == 0] = 0.0
            result[result_height > 0] = np.maximum(
                result[result_height > 0], small_value_threshold
            )

        # ------------------------------------------------------------------
        # Write output.
        # ------------------------------------------------------------------
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        out_count = 2 if result_height is not None else 1
        profile.update({"dtype": "float32", "count": out_count, "nodata": -1})
        with rasterio.open(save_fp, "w", **profile) as dst:
            dst.write(result, 1)
            if result_height is not None:
                dst.write(result_height, 2)

        elapsed = time.time() - start_time
        logger.info(f"{scene}: complete in {elapsed:.1f}s → {save_fp}")

    except Exception as e:
        logger.warning(f"{scene}: failed, skipping. {e}")
        if current_url:
            logger.warning(f"{scene}: last attempted URL → {current_url}")


def _load_mask(
    scene: str,
    mask_type: str,
    blob_sas: str,
    url_template: str,
    internal_template: str,
    aoi: dict,
    result: np.ndarray,
    profile: dict,
    elevation_threshold,
    stac_fallback: bool = False,
):
    """Return a boolean mask for water or elevation, trying sources in priority order.

    Priority:
        1. Explicit URL template (external users).
        2. Internal blob raster (if ``blob_sas`` available).
        3. Planetary Computer STAC (only if ``stac_fallback=True``).

    For pre-computed rasters the mask is simply ``raster == 1``.
    For STAC-derived masks, the existing ``get_water_mask`` /
    ``get_elevation_mask`` functions are used.
    """
    # Priority 1 or 2: pre-computed raster.
    raster_url = None
    if url_template:
        raster_url = url_template.format(scene=scene, sas=blob_sas or "")
    elif blob_sas:
        raster_url = internal_template.format(scene=scene, sas=blob_sas)

    if raster_url:
        logger.debug(f"{scene}: opening {mask_type} raster {raster_url}")
        try:
            with rasterio.open(raster_url) as src:
                data = src.read(1)
            return data == 1
        except Exception as e:
            if stac_fallback:
                logger.warning(f"{scene}: could not read {mask_type} raster — {e}. Falling back to STAC.")
            else:
                logger.warning(f"{scene}: could not read {mask_type} raster — {e}. Skipping mask (pass --stac-fallback to query STAC instead).")
                return None

    if not stac_fallback:
        logger.debug(f"{scene}: no {mask_type} raster source available and --stac-fallback is off — skipping mask.")
        return None

    # Priority 3: STAC fallback (opt-in).
    logger.debug(f"{scene}: querying {mask_type} mask via Planetary Computer STAC.")
    if mask_type == "water":
        return get_water_mask(aoi, result, profile)
    else:
        return get_elevation_mask(aoi, result, profile, threshold=elevation_threshold)


def run_ensemble(args) -> None:
    """Read the input CSV and dispatch ``process_scene`` calls via a thread pool.

    Reads ``OUTPUT_SAS``, ``UDM_SAS``, and optionally ``BLOB_SAS`` from the
    environment.  When ``--num-shards`` > 1, each process works on an
    interleaved slice of the scene list so that N AML jobs cover the full CSV
    without overlap.
    """
    output_sas = os.environ["OUTPUT_SAS"]
    udm_sas = os.environ["UDM_SAS"]
    blob_sas = os.environ.get("BLOB_SAS")  # optional; enables pre-computed water/elevation

    df = pd.read_csv(args.input_csv)
    scenes = list(df["scene"].unique())

    if args.num_shards > 1:
        scenes = scenes[args.shard_index::args.num_shards]
        logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(scenes)} scenes.")

    last_timestamp = args.timestamps[-1]

    scene_kwargs = dict(
        timestamps=args.timestamps,
        model=args.model,
        output_sas=output_sas,
        udm_sas=udm_sas,
        blob_sas=blob_sas,
        label_url_template=args.label_url_template,
        udm_url_template=args.udm_url_template,
        water_url_template=args.water_url_template,
        elevation_url_template=args.elevation_url_template,
        skip_existing=args.skip_existing,
        confidence_threshold=args.confidence_threshold,
        averaging_algorithm=args.averaging_algorithm,
        small_value_threshold=args.small_value_threshold,
        small_height_threshold=args.small_height_threshold,
        clarity_threshold=args.clarity_threshold,
        default_behavior=args.default_behavior,
        elevation_threshold=args.elevation_threshold,
        water=not args.no_water,
        elevation=not args.no_elevation,
        stac_fallback=args.stac_fallback,
    )

    def _process(scene):
        save_fp = (
            Path(args.save_dir)
            / last_timestamp
            / args.save_folder
            / f"{scene}.tif"
        )
        process_scene(scene=scene, save_fp=save_fp, **scene_kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process, scene): scene for scene in scenes}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Ensembling scenes",
        ):
            future.result()


def set_up_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for ensemble.py.

    All optional arguments default to the values used in the reference
    pipeline. Override only what you need to change.
    """
    parser = argparse.ArgumentParser(
        description="Ensemble building density predictions across quarterly timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv", required=True, type=str,
        help="CSV with a 'scene' column listing scene IDs to process.",
    )
    parser.add_argument(
        "--model", required=True, type=str,
        help="Model folder name, e.g. 9-37-best_practices_p3.",
    )
    parser.add_argument(
        "--save-dir", required=True, type=str,
        help="Root output directory.",
    )
    parser.add_argument(
        "--save-folder", required=True, type=str,
        help="Sub-folder written under <save-dir>/<last-timestamp>/.",
    )
    parser.add_argument(
        "--timestamps", required=True, nargs="+", type=str,
        help="Quarterly timestamps to ensemble, e.g. 2023q3 2023q4 2024q1 2024q2.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel threads.",
    )
    parser.add_argument(
        "--label-url-template", type=str, default=None,
        help=(
            "URL template for label TIFFs. "
            "Placeholders: {scene}, {timestamp}, {model}, {sas}. "
            "If omitted, uses the internal Microsoft blob storage layout."
        ),
    )
    parser.add_argument(
        "--udm-url-template", type=str, default=None,
        help=(
            "URL template for UDM TIFFs. "
            "Placeholders: {scene}, {timestamp}, {sas}. "
            "If omitted, uses the internal layout with automatic v1/v2 selection."
        ),
    )
    parser.add_argument(
        "--water-url-template", type=str, default=None,
        help=(
            "URL template for pre-computed permanent-water rasters (value==1 → water). "
            "Placeholders: {scene}, {sas}. "
            "If omitted, uses the internal BLOB_SAS template or falls back to STAC."
        ),
    )
    parser.add_argument(
        "--elevation-url-template", type=str, default=None,
        help=(
            "URL template for pre-computed elevation rasters (value==1 → high elevation). "
            "Placeholders: {scene}, {sas}. "
            "If omitted, uses the internal BLOB_SAS template or falls back to STAC."
        ),
    )
    parser.add_argument(
        "--confidence-threshold", type=int, default=95,
        help="UDM confidence threshold for ranked mask (0–100).",
    )
    parser.add_argument(
        "--averaging-algorithm", type=str, default="mean",
        choices=["mean", "max", "min", "mode"],
        help="Algorithm for 8×8 UDM block downsampling.",
    )
    parser.add_argument(
        "--small-value-threshold", type=float, default=2 / 255,
        help="Minimum density value considered a building.",
    )
    parser.add_argument(
        "--small-height-threshold", type=float, default=0.024,
        help="Minimum height value considered a building (applied to band 2 when present).",
    )
    parser.add_argument(
        "--clarity-threshold", type=float, default=3.5,
        help="UDM rank cutoff for 'clear' pixel classification.",
    )
    parser.add_argument(
        "--default-behavior", type=str, default="median",
        choices=["median", "mean", "max", "min"],
        help="Aggregation method for mixed-clarity 2-timestamp cases.",
    )
    parser.add_argument(
        "--elevation-threshold", type=int, default=5100,
        help="Elevation in meters above which pixels are zeroed (NASADEM).",
    )
    parser.add_argument(
        "--no-water", action="store_true",
        help="Skip permanent water masking.",
    )
    parser.add_argument(
        "--no-elevation", action="store_true",
        help="Skip high-elevation masking.",
    )
    parser.add_argument(
        "--stac-fallback", action="store_true",
        help=(
            "If a pre-computed water/elevation raster is unavailable or unreadable, "
            "fall back to Planetary Computer STAC queries. Off by default."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip scenes whose output file already exists.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Zero-based index of this shard. Use with --num-shards for AML parallelism.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards. Set > 1 to partition the scene list across AML jobs.",
    )
    return parser


def main(args) -> None:
    """Entry point: log start/end and delegate to ``run_ensemble``."""
    logger.info("Beginning ensemble pipeline.")
    run_ensemble(args)
    logger.info("Ensemble complete.")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
