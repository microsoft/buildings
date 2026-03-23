# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build a scenes CSV from a spatial tile index GeoPackage.

Reads the index (local path or blob URL), extracts scene IDs, and writes
a one-column CSV suitable for use with ensemble.py / submit_ensemble.py.

Scenes in the index that have no predictions for the requested timestamps are
automatically skipped by ensemble.py at runtime — no pre-filtering needed.

Required environment variables (loaded from .env automatically):
    BLOB_SAS  — SAS token for the index blob (if --index is a URL).
                Override with --sas-env if a different token is needed.

Usage:
    # from blob URL (cached locally after first download)
    python scripts/build_scene_csv.py \\
        --index https://<storage-account>.blob.core.windows.net/<container>/<index>.gpkg \\
        --output data/scenes.csv

    # from a local copy
    python scripts/build_scene_csv.py \\
        --index data/<index>.gpkg \\
        --output data/scenes.csv
"""

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _download_blob(url: str, sas_token: str, dest: Path) -> None:
    from azure.storage.blob import BlobClient
    parsed        = urlparse(url)
    account_url   = f"{parsed.scheme}://{parsed.netloc}"
    container, blob = parsed.path.lstrip("/").split("/", 1)
    client = BlobClient(
        account_url=f"{account_url}?{sas_token.lstrip('?')}",
        container_name=container,
        blob_name=blob,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {blob} → {dest} ...")
    with open(dest, "wb") as f:
        client.download_blob().readinto(f)
    logger.info("Download complete.")


def main(args) -> None:
    index_path = Path(args.index) if not args.index.startswith("http") else None

    if index_path is None:
        # Blob URL — cache locally so re-runs are instant
        cache_path = Path("data") / Path(urlparse(args.index).path).name
        if not cache_path.exists():
            sas_token = os.environ[args.sas_env]
            _download_blob(args.index, sas_token, cache_path)
        else:
            logger.info(f"Using cached index at {cache_path}")
        index_path = cache_path

    logger.info(f"Reading {index_path} ...")
    gdf = gpd.read_file(index_path)

    if args.scene_col not in gdf.columns:
        logger.error(
            f"Column '{args.scene_col}' not found in index. "
            f"Available columns: {list(gdf.columns)}"
        )
        raise SystemExit(1)

    scenes = sorted(
        s.removesuffix(".tif") for s in gdf[args.scene_col].dropna().unique()
    )
    df = pd.DataFrame({"scene": scenes})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(df):,} scenes to {output_path}")


def set_up_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a scenes CSV from a tile index GeoPackage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index", required=True, type=str,
        help="Local path or blob URL to the index GeoPackage.",
    )
    parser.add_argument(
        "--scene-col", type=str, default="quad",
        help="Column name in the GeoPackage that holds scene IDs.",
    )
    parser.add_argument(
        "--sas-env", type=str, default="BLOB_SAS",
        help="Name of the env var containing the SAS token (used when --index is a URL).",
    )
    parser.add_argument(
        "--output", type=str, default="data/scenes.csv",
        help="Output CSV path.",
    )
    return parser


if __name__ == "__main__":
    parser = set_up_parser()
    args   = parser.parse_args()
    main(args)
