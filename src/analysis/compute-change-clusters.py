"""Compute growth clusters between two temporal prediction rasters.

This module pulls prediction GeoTIFFs at two timestamps, computes
volumetric growth (density * height), derives a percentile-based threshold,
vectorizes growth above that threshold into polygons, and writes results to
a GeoPackage layer.

Example usage:
    python compute-high-growth-clusters.py
        --ensemble-dir my_ensemble
        --start-ts 2020q2
        --end-ts 2025q2
        --bbox -97.5 32.4 -96.3 33.2
        --growth-percentile 0.95
        --output-gpkg growth.gpkg
        --output-layer clusters
"""

import os
import argparse
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Tuple, List, Optional

from azure.storage.blob import ContainerClient
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import shape, box


class Config(BaseModel):
    """Runtime configuration for growth cluster computation.

    Parameters set via CLI. Tile structure assumed:
    outputs/predictions/<timestamp>/<ensemble_dir>/<tile_id>.tif

    growth_percentile semantics:
        +p (0<p<=1): select clusters of largest positive growth
            (top p quantile of positive values)
        -p (-1<=p<0): select clusters of largest declines
            (most negative values; bottom p quantile of negative values)
    """

    account_url: str = ""
    container_name: str = ""
    predictions_root: str = ""
    ensemble_dir: str = ""
    sas_token: str = ""  # provide via --sas-token or SAS_TOKEN env

    start_ts: str = "2020q2"
    end_ts: str = "2025q2"
    growth_percentile: float = 0.95  # -1..1 (sign indicates direction)
    height_scale_m: float = 100.0  # converts band2 normalized height to meters

    bbox: Optional[Tuple[float, float, float, float]] = (
        None  # (minx, miny, maxx, maxy) WGS84
    )

    output_gpkg: str = "growth_clusters.gpkg"
    output_layer: str = "clusters"

    max_tiles_per_timestamp: int = 200
    quad_index_gpkg: str = ""
    quad_layer: str = ""
    tile_ids_override: Tuple[str, ...] = ()  # explicit tile list


DEFAULT_CFG = Config()


def get_container_client(cfg: Config) -> ContainerClient:
    """Return a `ContainerClient` for listing prediction blobs.

    Uses SAS token if provided.
    """
    credential = cfg.sas_token if cfg.sas_token else None
    return ContainerClient(cfg.account_url, cfg.container_name, credential=credential)


def blob_to_vsicurl(cfg: Config, blob_name: str) -> str:
    """Construct a /vsicurl/ URL for streaming a blob via GDAL/rasterio."""
    base_url = f"{cfg.account_url}/{cfg.container_name}/{blob_name}"
    if cfg.sas_token:
        base_url = f"{base_url}?{cfg.sas_token}"
    return f"/vsicurl/{base_url}"


def compute_tile_ids_from_index(cfg: Config) -> List[str]:
    """Intersect AOI bbox with quad index polygons and return tile IDs.

    Requires bbox; strips .tif extension from quad field values.
    """
    if cfg.bbox is None:
        raise ValueError("bbox must be provided either via --bbox or configuration")

    print(
        f"Loading quad index from {cfg.quad_index_gpkg}" f" (layer='{cfg.quad_layer}')"
    )
    quads = gpd.read_file(cfg.quad_index_gpkg, layer=cfg.quad_layer)

    if "quad" not in quads.columns:
        raise ValueError("Quad layer must contain a 'quad' column")

    if quads.crs is None:
        raise ValueError("Quad layer must have a defined CRS")

    minx, miny, maxx, maxy = cfg.bbox
    aoi = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    if aoi.crs != quads.crs:
        aoi = aoi.to_crs(quads.crs)
    mask = quads.intersects(aoi.geometry.union_all())
    selected = quads.loc[mask, "quad"].unique()

    tiles = []
    for q in selected:
        q_str = str(q).strip()
        if q_str.lower().endswith(".tif"):
            tiles.append(Path(q_str).stem)
        else:
            tiles.append(q_str)
    tiles_sorted = sorted(tiles)

    print(f"Found {len(tiles_sorted)} quad tiles intersecting AOI.")
    return tiles_sorted


def bbox_intersects(raster_bounds, raster_crs, bbox_wgs84) -> bool:
    """Return True if raster bounds intersect AOI bbox (in WGS84)."""
    rb_wgs = transform_bounds(
        raster_crs,
        "EPSG:4326",
        raster_bounds.left,
        raster_bounds.bottom,
        raster_bounds.right,
        raster_bounds.top,
        densify_pts=21,
    )
    rb_minx, rb_miny, rb_maxx, rb_maxy = rb_wgs
    minx, miny, maxx, maxy = bbox_wgs84
    return not (rb_maxx < minx or rb_minx > maxx or rb_maxy < miny or rb_miny > maxy)


def list_tiles_for_timestamp(cfg: Config, timestamp: str) -> Dict[str, str]:
    """Return mapping tile_id -> URL for timestamp; apply AOI if provided."""
    tiles = dict()
    if cfg.tile_ids_override:
        print(
            f"Using {len(cfg.tile_ids_override)} override tile IDs" f" for {timestamp}"
        )
        for tile_id in cfg.tile_ids_override:
            blob_name = (
                f"{cfg.predictions_root}/{timestamp}/{cfg.ensemble_dir}/{tile_id}.tif"
            )
            tiles[tile_id] = blob_to_vsicurl(cfg, blob_name)
        return tiles

    container = get_container_client(cfg)
    prefix = f"{cfg.predictions_root}/{timestamp}/{cfg.ensemble_dir}/"
    print(
        f"Listing blobs with prefix '{prefix}' "
        f"(limit {cfg.max_tiles_per_timestamp})"
    )

    for blob in container.list_blobs(name_starts_with=prefix):
        if not blob.name.endswith(".tif"):
            continue
        tile_id = Path(blob.name).stem
        url = blob_to_vsicurl(cfg, blob.name)
        try:
            with rasterio.open(url) as src:
                if bbox_intersects(src.bounds, src.crs, cfg.bbox):
                    tiles[tile_id] = url
                    if len(tiles) >= cfg.max_tiles_per_timestamp:
                        print("Reached scan limit; stopping")
                        break
        except Exception as e:
            print(f"Failed to open {blob.name}: {e}")
    print(f"Found {len(tiles)} AOI tiles for {timestamp}")
    return tiles


def compute_tile_growth(start_url: str, end_url: str, height_scale_m: float):
    """Compute per-pixel growth array (end - start volume)."""
    with rasterio.open(start_url) as src_start, rasterio.open(end_url) as src_end:
        if src_start.crs != src_end.crs or src_start.transform != src_end.transform:
            raise ValueError("Start and end rasters must share CRS and transform")

        start_density = src_start.read(1, masked=True)
        start_height_norm = src_start.read(2, masked=True)

        end_density = src_end.read(1, masked=True)
        end_height_norm = src_end.read(2, masked=True)

        start_volume = start_density * (start_height_norm * height_scale_m)
        end_volume = end_density * (end_height_norm * height_scale_m)

        growth = end_volume - start_volume
        combined_mask = start_volume.mask | end_volume.mask
        growth = np.ma.array(growth, mask=combined_mask)
        profile = src_end.profile

    return growth, profile


def collect_growth_values_for_threshold(
    tiles_start: Dict[str, str],
    tiles_end: Dict[str, str],
    height_scale_m: float,
    percentile: float,
) -> Tuple[float, bool]:
    """
    Derive a growth or decline threshold based on percentile magnitude.

    Args:
        tiles_start: mapping tile_id -> /vsicurl start raster
        tiles_end: mapping tile_id -> /vsicurl end raster
        height_scale_m: scale factor for normalized height
        percentile: signed percentile. Positive => growth, Negative => decline.

    Returns:
        (threshold, is_decline)
        threshold: numeric threshold (positive for growth mode, negative for
            decline mode)
        is_decline: True if searching for largest declines
    """
    if percentile == 0 or abs(percentile) > 1:
        raise ValueError(
            "growth-percentile must be in (-1,0) or (0,1] (zero not allowed)"
        )
    is_decline = percentile < 0
    p = abs(percentile)

    collected: List[np.ndarray] = []
    shared = sorted(set(tiles_start.keys()) & set(tiles_end.keys()))
    mode_str = "decline" if is_decline else "growth"
    print(f"Sampling {mode_str} values across {len(shared)} overlapping tiles")

    for tile_id in shared:
        start_url = tiles_start[tile_id]
        end_url = tiles_end[tile_id]
        growth, _ = compute_tile_growth(start_url, end_url, height_scale_m)
        if is_decline:
            # collect negative values (declines)
            vals = growth.data[~growth.mask & (growth.data < 0)]
        else:
            vals = growth.data[~growth.mask & (growth.data > 0)]
        if vals.size:
            collected.append(vals.astype("float32").ravel())

    if not collected:
        raise RuntimeError(
            f"No {'decline' if is_decline else 'growth'} values inside AOI"
        )

    all_concat = np.concatenate(collected)
    if is_decline:
        # For declines we want the most negative tail: quantile at (1 - p)
        q = 1 - p
        threshold = float(np.quantile(all_concat, q))
        print(
            f"Decline threshold (bottom {p*100:.1f}% => q={q:.2f}): " f"{threshold:.4f}"
        )
    else:
        # For growth we want top p quantile of positive values
        threshold = float(np.quantile(all_concat, p))
        print(f"Growth threshold (p={p*100:.1f}%): {threshold:.4f}")
    return threshold, is_decline


def polygons_from_tile(
    growth: np.ma.MaskedArray,
    profile: dict,
    threshold: float,
    tile_id: str,
    is_decline: bool,
) -> gpd.GeoDataFrame:
    """Vectorize contiguous growth or decline clusters.

    In growth mode (is_decline=False): selects pixels with growth >= threshold.
    In decline mode (is_decline=True): selects pixels with growth <= threshold
        (threshold < 0).
    Clusters are 8-connected.
    """
    if is_decline:
        mask = (~growth.mask) & (growth.data <= threshold)
    else:
        mask = (~growth.mask) & (growth.data >= threshold)

    if not mask.any():
        return gpd.GeoDataFrame(
            columns=["tile_id", "mean_growth", "geometry"],
            crs=profile["crs"],
        )

    structure = np.ones((3, 3), dtype=int)
    labels, n_labels = ndimage.label(mask.astype("uint8"), structure=structure)
    if n_labels == 0:
        return gpd.GeoDataFrame(
            columns=["tile_id", "mean_growth", "geometry"],
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
        cluster_values = growth.data[cluster_mask]
        if cluster_values.size == 0:
            continue
        mean_growth = float(cluster_values.mean())
        results.append((shape(geom), mean_growth))

    if not results:
        return gpd.GeoDataFrame(
            columns=["tile_id", "mean_growth", "geometry"],
            crs=profile["crs"],
        )
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame(
        {"tile_id": tile_id, "mean_growth": vals},
        geometry=list(geoms),
        crs=profile["crs"],
    )


def build_growth_clusters_gdf(
    cfg: Config,
    tiles_start: Dict[str, str],
    tiles_end: Dict[str, str],
    threshold: float,
    is_decline: bool,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame of growth/decline polygons beyond threshold."""
    shared = sorted(set(tiles_start.keys()) & set(tiles_end.keys()))
    gdfs: List[gpd.GeoDataFrame] = []
    for tile_id in shared:
        start_url = tiles_start[tile_id]
        end_url = tiles_end[tile_id]
        growth, profile = compute_tile_growth(start_url, end_url, cfg.height_scale_m)
        gdf_tile = polygons_from_tile(growth, profile, threshold, tile_id, is_decline)
        if not gdf_tile.empty:
            gdfs.append(gdf_tile)
    if not gdfs:
        mode = "declines" if is_decline else "growth"
        raise RuntimeError(f"No {mode} polygons beyond threshold in AOI")
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def main(cfg: Config):
    """Execute full growth cluster workflow."""
    if not cfg.tile_ids_override:
        print("Deriving tile IDs from bbox.")
        cfg.tile_ids_override = tuple(compute_tile_ids_from_index(cfg))

    tiles_start = list_tiles_for_timestamp(cfg, cfg.start_ts)
    tiles_end = list_tiles_for_timestamp(cfg, cfg.end_ts)
    shared = set(tiles_start) & set(tiles_end)
    if not shared:
        raise RuntimeError("No overlapping tiles between timestamps")

    threshold, is_decline = collect_growth_values_for_threshold(
        tiles_start, tiles_end, cfg.height_scale_m, cfg.growth_percentile
    )
    mode = "decline" if is_decline else "growth"
    clusters_gdf = build_growth_clusters_gdf(
        cfg, tiles_start, tiles_end, threshold, is_decline
    )
    print(
        f"Writing {len(clusters_gdf)} {mode} polygons to {cfg.output_gpkg}"
        f" (layer={cfg.output_layer})"
    )
    clusters_gdf.to_file(cfg.output_gpkg, layer=cfg.output_layer, driver="GPKG")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Compute volumetric growth clusters between two prediction timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ensemble-dir",
        default=DEFAULT_CFG.ensemble_dir,
        help="Ensemble directory name",
    )
    p.add_argument(
        "--start-ts", default=DEFAULT_CFG.start_ts, help="Start timestamp (folder name)"
    )
    p.add_argument(
        "--end-ts", default=DEFAULT_CFG.end_ts, help="End timestamp (folder name)"
    )
    p.add_argument(
        "--growth-percentile",
        type=float,
        default=DEFAULT_CFG.growth_percentile,
        help="Signed percentile magnitude: +p for growth, -p for decline (0<p<=1)",
    )
    p.add_argument(
        "--bbox",
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        type=float,
        nargs=4,
        help="AOI bbox WGS84 (minx miny maxx maxy)",
    )
    p.add_argument(
        "--output-gpkg", default=DEFAULT_CFG.output_gpkg, help="Output GeoPackage path"
    )
    p.add_argument(
        "--output-layer", default=DEFAULT_CFG.output_layer, help="Output layer name"
    )
    p.add_argument(
        "--sas-token",
        default=os.environ.get("SAS_TOKEN", DEFAULT_CFG.sas_token),
        help="SAS token (without leading '?') or env SAS_TOKEN",
    )
    p.add_argument(
        "--quad-index-gpkg",
        default=DEFAULT_CFG.quad_index_gpkg,
        help="Quad index GeoPackage path",
    )
    p.add_argument(
        "--quad-layer", default=DEFAULT_CFG.quad_layer, help="Quad index layer name"
    )
    p.add_argument(
        "--predictions-root",
        default=DEFAULT_CFG.predictions_root,
        help="Root folder for predictions",
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
    cfg.ensemble_dir = args.ensemble_dir
    cfg.start_ts = args.start_ts
    cfg.end_ts = args.end_ts
    cfg.growth_percentile = args.growth_percentile
    cfg.output_gpkg = args.output_gpkg
    cfg.output_layer = args.output_layer
    cfg.sas_token = args.sas_token
    cfg.quad_index_gpkg = args.quad_index_gpkg
    cfg.quad_layer = args.quad_layer
    cfg.predictions_root = args.predictions_root
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
