# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path

from google.cloud import storage
from tqdm import tqdm


def list_tif_urls(
    bucket_name: str,
    prefix: str,
    output_txt: str,
    year: int | None = None,
) -> None:
    client = storage.Client.create_anonymous_client()

    blobs = client.list_blobs(
        bucket_name,
        prefix=prefix,
        match_glob="**.tif",
        fields="items/name,nextPageToken",
        page_size=10_000,
    )

    out_path = Path(output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f, tqdm(desc="Writing URLs", total=650000) as pbar:
        for blob in blobs:
            name = blob.name  # only 'name' was fetched
            if year is not None:
                year_from_blob = extract_year_from_blob_name(name)
                if year_from_blob != year:
                    continue
            url = f"https://storage.googleapis.com/{bucket_name}/{name}"
            f.write(url + "\n")
            pbar.update(1)


def extract_year_from_blob_name(blob_name: str) -> int | None:
    # Example: v1/geotiffs/00824_2016_06_30/tile_3FYdP6L109o.tif
    try:
        parts = blob_name.split("/")
        # Find the folder with pattern XXXXX_YYYY_MM_DD
        for part in parts:
            if "_" in part:
                tokens = part.split("_")
                if len(tokens) >= 4 and len(tokens[1]) == 4:
                    return int(tokens[1])
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="List URLs for Google Open Buildings Temporal data from GCS"
    )
    parser.add_argument(
        "--url-list",
        default="/mnt/tempo/google/urls.txt",
        help="Path to save the list of GeoTIFF URLs (default: /mnt/tempo/google/urls.txt)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional: filter to only list tiles from a specific year (e.g., 2016)",
    )

    args = parser.parse_args()

    bucket_name = "open-buildings-temporal-data"
    prefix = "v1/geotiffs/"  # everything under this, including subdirs like 00824_2016_06_30/

    print("Listing GeoTIFF URLs from GCS...")
    list_tif_urls(bucket_name, prefix, args.url_list, year=args.year)

    print(f"Done: {args.url_list}")


if __name__ == "__main__":
    main()
