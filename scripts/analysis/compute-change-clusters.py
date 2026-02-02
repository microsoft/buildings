"""Compute change clusters between two temporal prediction rasters.

This module pulls prediction GeoTIFFs at two timestamps, computes
volumetric change (density * height delta), derives a percentile-based
threshold, vectorizes change above that threshold into polygons, and writes
results to a GeoPackage layer. Negative change percentiles represents declines.

Example usage:
1.  Compare growth between two COGs:
    python compute-change-clusters.py
        --start-cog /path/or/url/to/2020q2.tif
        --end-cog /path/or/url/to/2025q2.tif
        --change-percentile 0.95
        --output-gpkg growth.gpkg
        --output-layer clusters
        --min-cluster-pixels 8 # optional

2.  Compare decline between two COGs:
    python compute-change-clusters.py
        --start-cog /path/or/url/to/2020q2.tif
        --end-cog /path/or/url/to/2025q2.tif
        --change-percentile -0.95
        --output-gpkg decline.gpkg
        --output-layer clusters
        --min-cluster-pixels 8 # optional

3.  Compare tifs stored in a blob container, limit analysis to bbox provided:
    Expected directory structure:
        predictions_root / timestamp / data_dir / tile_id.tif
    python compute-change-clusters.py
        --account-url https://<account>.blob.core.windows.net
        --container-name <container>
        --predictions-root predictions_root
        --data-dir data_dir
        --start-ts 2020q2
        --end-ts 2025q2
        --bbox -97.5 32.4 -96.3 33.2
        --change-percentile 0.95  # switch to negative for declines
        --output-gpkg growth.gpkg
        --output-layer clusters
"""

import argparse
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Tuple, List, Optional

from azure.storage.blob import ContainerClient
import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import shape, box


class Config(BaseModel):
    """Runtime configuration for change cluster computation.

    Parameters set via CLI. If using blob storage, expected structure is:
        <predictions_root> / <timestamp> / <data_dir> / <tile_id>.tif

    Percentile semantics:
        +p (0<p<=1): select clusters of largest positive change (growth)
            (top p quantile of positive values)
        -p (-1<=p<0): select clusters of largest negative change (decline)
            (bottom p quantile of negative values)
    """

    # Parameters if using blob storage to access tifs
    account_url: Optional[str] = None
    container_name: Optional[str] = None
    predictions_root: Optional[str] = None
    data_dir: Optional[str] = None
    sas_token: Optional[str] = None  # provide via --sas-token or SAS_TOKEN env
    start_ts: str = "2020q2"
    end_ts: str = "2025q2"
    bbox: Optional[Tuple[float, float, float, float]] = (
        None  # (minx, miny, maxx, maxy) WGS84
    )

    # Optional parameters if using blob storage to list blobs more efficently
    # Pre-selects tile IDs that intersect with bbox via quad index
    max_tiles_per_timestamp: int = 200
    quad_index_gpkg: Optional[str] = None
    tile_ids_override: Tuple[str, ...] = ()  # explicit tile list

    # Parameters if providing direct links to COGs
    start_cog: Optional[str] = None
    end_cog: Optional[str] = None

    # Required for either use case
    change_percentile: float = 0.95  # -1..1 (sign indicates direction)
    height_scale_m: float = 100.0  # converts band2 normalized height to meters
    min_cluster_pixels: int = 4  # minimum connected pixels per cluster
    output_gpkg: str = "change_clusters.gpkg"
    output_layer: str = "clusters"


DEFAULT_CFG = Config()


def get_container_client(cfg: Config) -> ContainerClient:
    """Return a `ContainerClient` for listing prediction blobs.

    Uses SAS token if provided.
    """
    if not cfg.account_url or not cfg.container_name:
        raise ValueError("account-url and container-name are required for blob listing")
    credential = cfg.sas_token if cfg.sas_token else None
    return ContainerClient(cfg.account_url, cfg.container_name, credential=credential)


def blob_to_url(cfg: Config, blob_name: str) -> str:
    """Construct an HTTPS URL for opening via rasterio/GDAL."""
    base = f"{cfg.account_url}/{cfg.container_name}/{blob_name}"
    return f"{base}?{cfg.sas_token}" if cfg.sas_token else base


def transform_bbox_to_crs(
    bbox_wgs84: Tuple[float, float, float, float], target_crs
) -> Tuple[float, float, float, float]:
    """Transform a WGS84 bbox to the target CRS once.

    Returns (minx, miny, maxx, maxy) in `target_crs`.
    """
    minx, miny, maxx, maxy = bbox_wgs84
    tb = transform_bounds(
        "EPSG:4326",
        target_crs,
        minx,
        miny,
        maxx,
        maxy,
        densify_pts=21,
    )
    return tb


def compute_tile_ids_from_index(cfg: Config) -> List[str]:
    """Intersect AOI bbox with quad index polygons and return tile IDs.

    Note: this implementation requires bbox in EPSG:4326, assumes index has a
    "quad" column, and strips .tif extension from quad field values if provided.
    """
    if cfg.bbox is None:
        raise ValueError("bbox must be provided")

    if cfg.quad_index_gpkg is None:
        raise ValueError("quad_index_gpkg must be provided")

    logger.info(f"Loading quad index from {cfg.quad_index_gpkg}")
    quads = gpd.read_file(cfg.quad_index_gpkg)

    if "quad" not in quads.columns:
        raise ValueError(
            f"Quad index must contain a 'quad' column; found: {quads.columns.tolist()}"
        )

    if quads.crs is None:
        raise ValueError("Quad index must have a defined CRS")

    # Ensure quad index -> bbox crs consistency before intersection
    minx, miny, maxx, maxy = transform_bbox_to_crs(cfg.bbox, quads.crs)
    bbox = box(minx, miny, maxx, maxy)
    mask = quads.intersects(bbox)
    selected = quads.loc[mask, "quad"].unique()

    tiles = []
    for q in selected:
        q_str = str(q).strip()
        if q_str.lower().endswith(".tif"):
            tiles.append(Path(q_str).stem)
        else:
            tiles.append(q_str)
    tiles_sorted = sorted(tiles)

    logger.info(f"Found {len(tiles_sorted)} quad tiles intersecting AOI.")
    return tiles_sorted


def bbox_intersects_same_crs(
    raster_bounds, bbox_in_raster_crs: Tuple[float, float, float, float]
) -> bool:
    """Return True if raster bounds intersect AOI bbox when both are in the same CRS."""
    rb_minx, rb_miny, rb_maxx, rb_maxy = (
        raster_bounds.left,
        raster_bounds.bottom,
        raster_bounds.right,
        raster_bounds.top,
    )
    minx, miny, maxx, maxy = bbox_in_raster_crs
    return not (rb_maxx < minx or rb_minx > maxx or rb_maxy < miny or rb_miny > maxy)


def list_tiles_for_timestamp(cfg: Config, timestamp: str) -> Dict[str, str]:
    """Return mapping tile_id -> URL for timestamp; apply AOI if provided."""
    logger.info(f"Listing tiles for timestamp '{timestamp}'")
    if not cfg.predictions_root or not cfg.data_dir:
        raise ValueError("predictions-root and data-dir are required for blob listing")

    tiles = dict()
    if cfg.tile_ids_override:
        logger.info(
            f"Using override tile IDs for {timestamp}: count={len(cfg.tile_ids_override)}"
        )
        for tile_id in cfg.tile_ids_override:
            blob_name = (
                f"{cfg.predictions_root}/{timestamp}/{cfg.data_dir}/{tile_id}.tif"
            )
            tiles[tile_id] = blob_to_url(cfg, blob_name)
        return tiles

    container = get_container_client(cfg)
    prefix = f"{cfg.predictions_root}/{timestamp}/{cfg.data_dir}/"
    logger.info(
        f"Listing blobs with prefix '{prefix}' (limit {cfg.max_tiles_per_timestamp})"
    )

    # Transform AOI bbox once to the first raster CRS; reuse thereafter
    ref_crs = None
    bbox_in_ref_crs = None

    for blob in container.list_blobs(name_starts_with=prefix):
        if not blob.name.endswith(".tif"):
            continue
        tile_id = Path(blob.name).stem
        url = blob_to_url(cfg, blob.name)
        with rasterio.open(url) as src:
            if cfg.bbox is None:
                tiles[tile_id] = url
                logger.debug(f"Selected tile without AOI filter: {tile_id}")
                if len(tiles) >= cfg.max_tiles_per_timestamp:
                    logger.warning("Reached scan limit; stopping")
                    break
                continue
            if ref_crs is None:
                ref_crs = src.crs
                bbox_in_ref_crs = transform_bbox_to_crs(cfg.bbox, ref_crs)
                logger.debug(f"Initialized reference CRS for bbox transform: {ref_crs}")
            bbox_to_use = (
                bbox_in_ref_crs
                if src.crs == ref_crs and bbox_in_ref_crs is not None
                else transform_bbox_to_crs(cfg.bbox, src.crs)
            )
            if bbox_intersects_same_crs(src.bounds, bbox_to_use):
                tiles[tile_id] = url
                logger.debug(f"Selected tile intersecting AOI: {tile_id}")
                if len(tiles) >= cfg.max_tiles_per_timestamp:
                    logger.warning("Reached scan limit; stopping")
                    break
    logger.info(f"Found {len(tiles)} AOI tiles for {timestamp}")
    return tiles


def compute_tile_change(start_url: str, end_url: str, height_scale_m: float):
    """Compute per-pixel change array (end - start volume)."""
    with rasterio.open(start_url) as src_start, rasterio.open(end_url) as src_end:
        if src_start.crs != src_end.crs or src_start.transform != src_end.transform:
            raise ValueError("Start and end rasters must share CRS and transform")

        start_density = src_start.read(1, masked=True)
        start_height_norm = src_start.read(2, masked=True)

        end_density = src_end.read(1, masked=True)
        end_height_norm = src_end.read(2, masked=True)

        start_volume = start_density * (start_height_norm * height_scale_m)
        end_volume = end_density * (end_height_norm * height_scale_m)

        change = end_volume - start_volume
        combined_mask = start_volume.mask | end_volume.mask
        change = np.ma.array(change, mask=combined_mask)
        profile = src_end.profile

    return change, profile


def collect_change_values_for_threshold(
    tiles_start: Dict[str, str],
    tiles_end: Dict[str, str],
    height_scale_m: float,
    percentile: float,
) -> Tuple[float, bool]:
    """
    Derive a change or decline threshold based on percentile magnitude.

    Args:
        tiles_start: mapping tile_id -> start raster URL
        tiles_end: mapping tile_id -> end raster URL
        height_scale_m: scale factor for normalized height
        percentile: signed percentile. Positive => change, Negative => decline.

    Returns:
        (threshold, is_decline)
        threshold: numeric threshold (positive for change mode, negative for
            decline mode)
        is_decline: True if searching for largest declines
    """
    if percentile == 0 or abs(percentile) > 1:
        raise ValueError(
            "change-percentile must be in (-1,0) or (0,1] (zero not allowed)"
        )
    is_decline = percentile < 0
    p = abs(percentile)

    collected = []
    shared = sorted(set(tiles_start.keys()) & set(tiles_end.keys()))
    mode_str = "decline" if is_decline else "change"
    logger.info(f"Sampling {mode_str} values across {len(shared)} overlapping tiles")

    for tile_id in shared:
        start_url = tiles_start[tile_id]
        end_url = tiles_end[tile_id]
        change, _ = compute_tile_change(start_url, end_url, height_scale_m)
        if is_decline:
            vals = change.data[~change.mask & (change.data < 0)]
        else:
            vals = change.data[~change.mask & (change.data > 0)]
        if vals.size:
            collected.append(vals.astype("float32").ravel())

    if not collected:
        raise RuntimeError(
            f"No {'decline' if is_decline else 'change'} values inside AOI"
        )

    all_concat = np.concatenate(collected)
    if is_decline:
        # For declines we want the most negative tail: quantile at (1 - p)
        q = 1 - p
        threshold = float(np.quantile(all_concat, q))
        logger.info(
            f"Decline threshold (bottom {p*100:.1f}% => q={q:.2f}): {threshold:.4f} (samples={all_concat.size})"
        )
    else:
        # For change we want top p quantile of positive values
        threshold = float(np.quantile(all_concat, p))
        logger.info(
            f"Change threshold (p={p*100:.1f}%): {threshold:.4f} (samples={all_concat.size})"
        )
    return threshold, is_decline


def polygons_from_tile(
    change: np.ma.MaskedArray,
    profile: dict,
    threshold: float,
    tile_id: str,
    is_decline: bool,
    min_cluster_pixels: int,
) -> gpd.GeoDataFrame:
    """Vectorize contiguous growth or decline clusters.

    In change mode (is_decline=False): selects pixels with change >= threshold.
    In decline mode (is_decline=True): selects pixels with change <= threshold
        (threshold < 0).
    Clusters are 8-connected.
    """
    if is_decline:
        mask = (~change.mask) & (change.data <= threshold)
    else:
        mask = (~change.mask) & (change.data >= threshold)

    if not mask.any():
        return gpd.GeoDataFrame(
            columns=["tile_id", "mean_change", "geometry"],
            crs=profile["crs"],
        )

    structure = np.ones((3, 3), dtype=int)
    labels, n_labels = ndimage.label(mask.astype("uint8"), structure=structure)
    # Compute component sizes within the threshold mask
    if n_labels > 0:
        sizes = np.bincount(labels[mask].ravel())  # index 0 is background
        keep_ids = np.where(sizes >= min_cluster_pixels)[0]
        keep_ids = keep_ids[keep_ids != 0]
        logger.debug(
            f"Tile {tile_id}: components={n_labels}, min_pixels={min_cluster_pixels}, keep={len(keep_ids)}"
        )
        if keep_ids.size == 0:
            return gpd.GeoDataFrame(
                columns=["tile_id", "mean_change", "geometry"],
                crs=profile["crs"],
            )
        keep_map = np.zeros(sizes.shape[0], dtype=bool)
        keep_map[keep_ids] = True
        mask = mask & keep_map[labels]
        # Re-check mask after filtering
        if not mask.any():
            return gpd.GeoDataFrame(
                columns=["tile_id", "mean_change", "geometry"],
                crs=profile["crs"],
            )
    logger.debug(f"Tile {tile_id}: connected components found={n_labels}")
    if n_labels == 0:
        return gpd.GeoDataFrame(
            columns=["tile_id", "mean_change", "geometry"],
            crs=profile["crs"],
        )

    results = []
    for geom, value in shapes(
        labels.astype("int32"),
        mask=mask,
        transform=profile["transform"],
        connectivity=8,
    ):
        label_id = int(value)
        if label_id == 0:
            continue
        cluster_mask = (labels == label_id) & mask
        cluster_values = change.data[cluster_mask]
        if cluster_values.size == 0:
            continue
        mean_change = float(cluster_values.mean())
        results.append((shape(geom), mean_change))

    if not results:
        return gpd.GeoDataFrame(
            columns=["tile_id", "mean_change", "geometry"],
            crs=profile["crs"],
        )
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame(
        {"tile_id": tile_id, "mean_change": vals},
        geometry=list(geoms),
        crs=profile["crs"],
    )


def build_change_clusters_gdf(
    cfg: Config,
    tiles_start: Dict[str, str],
    tiles_end: Dict[str, str],
    threshold: float,
    is_decline: bool,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame of chagrowthnge/decline polygons beyond threshold."""
    shared = sorted(set(tiles_start.keys()) & set(tiles_end.keys()))
    gdfs = []
    for tile_id in shared:
        start_url = tiles_start[tile_id]
        end_url = tiles_end[tile_id]
        change, profile = compute_tile_change(start_url, end_url, cfg.height_scale_m)
        gdf_tile = polygons_from_tile(
            change, profile, threshold, tile_id, is_decline, cfg.min_cluster_pixels
        )
        if not gdf_tile.empty:
            gdfs.append(gdf_tile)
            logger.debug(f"Tile {tile_id}: polygons beyond threshold={len(gdf_tile)}")
        else:
            logger.debug(f"Tile {tile_id}: no polygons beyond threshold")
    if not gdfs:
        mode = "declines" if is_decline else "change"
        raise RuntimeError(f"No {mode} polygons beyond threshold in AOI")
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def main(cfg: Config):
    """Execute full change cluster workflow."""
    if cfg.start_cog and cfg.end_cog:
        logger.info(f"Using direct COGs: start={cfg.start_cog}, end={cfg.end_cog}")
        # Use a shared synthetic tile ID to ensure overlap
        tiles_start = {"cog": cfg.start_cog}
        tiles_end = {"cog": cfg.end_cog}
    else:
        logger.info(
            f"Blob listing mode: account_url={cfg.account_url}, container={cfg.container_name}, "
            f"root={cfg.predictions_root}, data_dir={cfg.data_dir}, start_ts={cfg.start_ts}, end_ts={cfg.end_ts}, "
            f"bbox={cfg.bbox}, quad_index={cfg.quad_index_gpkg}, overrides={len(cfg.tile_ids_override)}"
        )
        # Optionally derive tile IDs from a quad index if provided.
        if (not cfg.tile_ids_override) and cfg.bbox and cfg.quad_index_gpkg:
            logger.info("Deriving tile IDs from bbox via quad index")
            cfg.tile_ids_override = tuple(compute_tile_ids_from_index(cfg))

        tiles_start = list_tiles_for_timestamp(cfg, cfg.start_ts)
        tiles_end = list_tiles_for_timestamp(cfg, cfg.end_ts)
    logger.info(
        f"Tile listing complete: start_count={len(tiles_start)}, end_count={len(tiles_end)}"
    )
    sample_start = list(tiles_start.keys())[:10]
    sample_end = list(tiles_end.keys())[:10]
    logger.debug(f"Sample start tiles: {sample_start}")
    logger.debug(f"Sample end tiles: {sample_end}")
    shared = set(tiles_start) & set(tiles_end)
    if not shared:
        logger.error(
            f"No overlapping tiles: start_count={len(tiles_start)}, end_count={len(tiles_end)}"
        )
        raise RuntimeError("No overlapping tiles between timestamps")

    threshold, is_decline = collect_change_values_for_threshold(
        tiles_start, tiles_end, cfg.height_scale_m, cfg.change_percentile
    )
    mode = "decline" if is_decline else "change"
    clusters_gdf = build_change_clusters_gdf(
        cfg, tiles_start, tiles_end, threshold, is_decline
    )
    logger.info(
        f"Writing {len(clusters_gdf)} {mode} polygons to {cfg.output_gpkg} (layer={cfg.output_layer})"
    )
    clusters_gdf.to_file(cfg.output_gpkg, layer=cfg.output_layer, driver="GPKG")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Compute volumetric change clusters between two prediction timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Data directory name",
    )
    p.add_argument("--start-ts", default="2020q2", help="Start timestamp (folder name)")
    p.add_argument("--end-ts", default="2025q2", help="End timestamp (folder name)")
    p.add_argument(
        "--start-cog",
        default=None,
        help="Path or URL to start (early) timestamp COG (skips blob listing)",
    )
    p.add_argument(
        "--end-cog",
        default=None,
        help="Path or URL to end (later) timestamp COG (skips blob listing)",
    )
    p.add_argument(
        "--change-percentile",
        type=float,
        default=0.95,
        help="Signed percentile magnitude: +p for change, -p for decline (0<p<=1)",
    )
    p.add_argument(
        "--bbox",
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        type=float,
        nargs=4,
        help="AOI bbox WGS84 (minx miny maxx maxy)",
    )
    p.add_argument(
        "--output-gpkg", default="change_clusters.gpkg", help="Output GeoPackage path"
    )
    p.add_argument("--output-layer", default="clusters", help="Output layer name")
    p.add_argument(
        "--sas-token",
        default=os.environ.get("SAS_TOKEN", None),
        help="SAS token (without leading '?') or env SAS_TOKEN",
    )
    p.add_argument(
        "--account-url",
        default=os.environ.get("AZ_ACCOUNT_URL", None),
        help="Azure storage account URL (e.g., https://<account>.blob.core.windows.net)",
    )
    p.add_argument(
        "--container-name",
        default=os.environ.get("AZ_CONTAINER", None),
        help="Azure storage container name",
    )
    p.add_argument(
        "--quad-index-gpkg",
        default=None,
        help="Quad index GeoPackage path (optional)",
    )
    p.add_argument(
        "--quad-layer", default=None, help="Quad index layer name (optional)"
    )
    p.add_argument(
        "--predictions-root",
        default=None,
        help="Root folder for predictions inside the container",
    )
    p.add_argument(
        "--height-scale-m",
        type=float,
        default=100.0,
        help="Scale factor to convert normalized height to meters",
    )
    p.add_argument(
        "--min-cluster-pixels",
        type=int,
        default=4,
        help="Minimum number of connected pixels to keep a cluster",
    )
    p.add_argument(
        "--max-tiles-per-timestamp",
        type=int,
        default=200,
        help="Maximum tiles to scan per timestamp",
    )
    p.add_argument(
        "--tile-ids-override",
        nargs="*",
        help="Explicit tile IDs (skip bbox intersection)",
    )
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Convert parsed arguments into a `Config` object."""
    cfg = Config()
    cfg.account_url = args.account_url
    cfg.container_name = args.container_name
    cfg.predictions_root = args.predictions_root
    cfg.data_dir = args.data_dir
    cfg.sas_token = args.sas_token
    cfg.start_ts = args.start_ts
    cfg.end_ts = args.end_ts
    cfg.start_cog = args.start_cog
    cfg.end_cog = args.end_cog
    cfg.change_percentile = args.change_percentile
    cfg.height_scale_m = args.height_scale_m
    cfg.min_cluster_pixels = args.min_cluster_pixels
    cfg.output_gpkg = args.output_gpkg
    cfg.output_layer = args.output_layer
    cfg.quad_index_gpkg = args.quad_index_gpkg
    cfg.max_tiles_per_timestamp = args.max_tiles_per_timestamp
    if args.bbox:
        minx, miny, maxx, maxy = args.bbox
        cfg.bbox = (minx, miny, maxx, maxy)
    if args.tile_ids_override:
        cfg.tile_ids_override = tuple(args.tile_ids_override)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = build_config_from_args(args)
    main(config)
