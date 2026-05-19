# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
import glob
import shutil
import logging
import argparse
import tempfile
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import requests
import pandas as pd
import subprocess
import rasterio
import geopandas as gpd
from shapely import wkb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("process_quad.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def find_proj_lib():
    """Find and set PROJ_LIB environment variable."""
    possible_paths = glob.glob("/azureml-envs/**/proj.db", recursive=True)
    if possible_paths:
        os.environ["PROJ_LIB"] = os.path.dirname(possible_paths[0])
    else:
        # Try to find it in conda environment or system
        conda_paths = glob.glob("/opt/conda/**/proj.db", recursive=True)
        if conda_paths:
            os.environ["PROJ_LIB"] = os.path.dirname(conda_paths[0])


def extract_tile_id(tile_path):
    """
    Extract tile ID from tile path.
    Example: 824_2018_06_30/tile_SrivkTzhats.tif -> tile_SrivkTzhats
    """
    if "/" in tile_path:
        filename = tile_path.split("/")[-1]
        # Remove .tif extension and return tile ID
        return filename.replace(".tif", "")
    return tile_path.replace(".tif", "")


def load_urls_file(urls_file_path):
    """
    Load URLs from file and create a mapping from tile ID to URL.

    Args:
        urls_file_path: Path to text file containing URLs (one per line)

    Returns:
        dict: Mapping of tile_id -> full URL
    """
    tile_id_to_url = {}

    with open(urls_file_path, "r") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue

            # Extract tile ID from URL
            # Example URL: https://...../tile_SrivkTzhats.tif
            if "tile_" in url:
                tile_id = url.split("tile_")[-1].replace(".tif", "")
                tile_id_to_url[f"tile_{tile_id}"] = url

    logger.info(f"Loaded {len(tile_id_to_url)} URLs from {urls_file_path}")
    return tile_id_to_url


def download_file_http(url, download_path, retries=1, delay=3):
    """Download a file from HTTP URL with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()

            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except (requests.exceptions.RequestException, ConnectionError) as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.warning(
                    f"Failed to download {url} after {retries} attempts: {e}"
                )
                return False
    return False


def reproject_file(file, processed_dir, majority_crs):
    """Reproject a raster file to the majority CRS."""
    output_file = processed_dir / f"reprojected_{file.name}"

    try:
        # Run the gdalwarp command and capture stdout/stderr
        subprocess.run(
            f'gdalwarp -overwrite -of GTIFF -srcnodata "-99" -dstnodata "-99" '
            f'-t_srs "{majority_crs}" --config GDAL_NUM_THREADS ALL_CPUS '
            f"-multi -wo NUM_THREADS=ALL_CPUS -co TILED=YES -co BIGTIFF=IF_SAFER "
            f'-co PREDICTOR=2 -co COMPRESS=LZW "{file}" "{output_file}"',
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during reprojection of {file}: {e}")
        logger.error(f"Command error: {e.stderr}")
        raise e

    # Replace the original file with the reprojected file
    os.remove(file)
    shutil.move(output_file, file)


def process_quad(
    quad_name,
    google_tile_paths,
    quad_geom,
    temp_path,
    output_dir,
    year,
    tile_id_to_url,
    majority_crs_epsg="EPSG:3857",
):
    """Process a single quad by downloading, reprojecting, and merging Google tiles."""
    import time as time_module

    start_time = time_module.time()
    logger.info(
        f"Starting processing for quad {quad_name} ({year}) with {len(google_tile_paths)} tiles"
    )

    quad_dir = temp_path / quad_name
    try:
        # Create directories
        quad_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = quad_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Download the images
        download_start = time_module.time()
        logger.info(f"[{quad_name}] Starting download phase")
        downloaded_files = []
        failed_downloads = 0
        skipped_no_url = 0

        for tile_path in google_tile_paths:
            tile_id = extract_tile_id(tile_path)

            # Search for URL containing this tile ID
            url = tile_id_to_url.get(tile_id)

            if not url:
                skipped_no_url += 1
                logger.warning(
                    f"No URL found for tile {tile_id} (from path {tile_path})"
                )
                continue

            filename = Path(tile_path).name
            download_path = quad_dir / filename

            success = download_file_http(url, download_path)
            if success:
                downloaded_files.append(download_path)
            else:
                failed_downloads += 1
                logger.warning(f"Skipping tile {tile_path} - download failed")

        download_elapsed = time_module.time() - download_start
        logger.info(
            f"[{quad_name}] Download phase completed in {download_elapsed:.2f}s - {len(downloaded_files)} successful, {failed_downloads} failed, {skipped_no_url} without URLs"
        )

        if failed_downloads > 0 or skipped_no_url > 0:
            logger.info(
                f"Quad {quad_name}: {failed_downloads} failed downloads, {skipped_no_url} tiles without URLs (out of {len(google_tile_paths)} total)"
            )

        if not downloaded_files:
            logger.warning(f"No files downloaded for quad {quad_name} - skipping")
            if quad_dir.exists():
                shutil.rmtree(quad_dir)
            return

        # Get the majority CRS
        crs_start = time_module.time()
        logger.info(f"[{quad_name}] Reading CRS from {len(downloaded_files)} files")
        crs_list = []
        for img in downloaded_files:
            try:
                with rasterio.open(img) as src:
                    crs_list.append(src.crs)
            except Exception as e:
                logger.warning(f"Failed to read CRS from {img}: {e}")
                continue

        if not crs_list:
            logger.warning(f"No valid CRS found for quad {quad_name} - skipping")
            if quad_dir.exists():
                shutil.rmtree(quad_dir)
            return

        majority_crs = Counter(crs_list).most_common(1)[0][0]
        crs_elapsed = time_module.time() - crs_start
        logger.info(
            f"[{quad_name}] CRS analysis completed in {crs_elapsed:.2f}s - majority CRS: {majority_crs}"
        )

        # Create a list of files that don't share the majority CRS
        files_to_reproject = [
            img for img, crs in zip(downloaded_files, crs_list) if crs != majority_crs
        ]

        # Reproject files sequentially
        if files_to_reproject:
            reproject_start = time_module.time()
            logger.info(
                f"[{quad_name}] Reprojecting {len(files_to_reproject)} files to {majority_crs}"
            )
            for file in files_to_reproject:
                reproject_file(file, processed_dir, majority_crs)
            reproject_elapsed = time_module.time() - reproject_start
            logger.info(
                f"[{quad_name}] Reprojection completed in {reproject_elapsed:.2f}s"
            )
        else:
            logger.info(
                f"[{quad_name}] No reprojection needed - all files use {majority_crs}"
            )

        # Reproject quad geometry to EPSG:3857 and get bounds
        geom_start = time_module.time()
        reprojected_geom = gpd.GeoDataFrame(
            geometry=[quad_geom], crs="EPSG:4326"
        ).to_crs(majority_crs_epsg)
        minx, miny, maxx, maxy = reprojected_geom.geometry.values[0].bounds
        geom_elapsed = time_module.time() - geom_start
        logger.info(
            f"[{quad_name}] Geometry reprojection completed in {geom_elapsed:.2f}s"
        )

        # Merge files using GDAL
        merge_start = time_module.time()
        logger.info(f"[{quad_name}] Starting GDAL merge operations")
        vrt_file = processed_dir / f"{quad_name}.vrt"
        merged_file = processed_dir / f"{quad_name}.tif"

        try:
            # Create VRT file from input images, only using band 3 and band 2
            vrt_start = time_module.time()
            tif_files_str = " ".join([f'"{file}"' for file in downloaded_files])
            command = f'gdalbuildvrt -b 3 -b 2 "{vrt_file}" {tif_files_str}'
            logger.info(
                f"[{quad_name}] Building VRT from {len(downloaded_files)} files"
            )
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            vrt_elapsed = time_module.time() - vrt_start
            logger.info(f"[{quad_name}] VRT build completed in {vrt_elapsed:.2f}s")

            # Run gdalwarp to reproject and merge the files
            warp_start = time_module.time()
            command = (
                f'gdalwarp -overwrite -of GTIFF -t_srs {majority_crs_epsg} -srcnodata "-99" -dstnodata "-99" '
                f"-te {minx} {miny} {maxx} {maxy} -ts 512 512 -r average "
                f"-multi -wo NUM_THREADS=ALL_CPUS -co TILED=YES -co BIGTIFF=IF_SAFER -co PREDICTOR=2 -co COMPRESS=LZW "
                f'"{vrt_file}" "{merged_file}"'
            )
            logger.info(f"[{quad_name}] Running gdalwarp to create final merged file")
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            warp_elapsed = time_module.time() - warp_start
            merge_elapsed = time_module.time() - merge_start
            logger.info(
                f"[{quad_name}] gdalwarp completed in {warp_elapsed:.2f}s (total merge: {merge_elapsed:.2f}s)"
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during GDAL processing for quad {quad_name}: {e}")
            logger.error(f"Command stderr: {e.stderr}")
            if quad_dir.exists():
                shutil.rmtree(quad_dir)
            raise e

        # Save the merged file to output directory
        save_start = time_module.time()
        output_year_dir = Path(output_dir) / str(year)
        output_year_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_year_dir / f"{quad_name}.tif"
        shutil.copy(merged_file, output_file)
        save_elapsed = time_module.time() - save_start
        logger.info(f"[{quad_name}] File copy completed in {save_elapsed:.2f}s")

        # Clean up
        cleanup_start = time_module.time()
        if quad_dir.exists():
            shutil.rmtree(quad_dir)
        cleanup_elapsed = time_module.time() - cleanup_start

        total_elapsed = time_module.time() - start_time
        logger.info(f"[{quad_name}] ✓ Successfully processed and saved {output_file}")
        logger.info(
            f"[{quad_name}] Total processing time: {total_elapsed:.2f}s (cleanup: {cleanup_elapsed:.2f}s)"
        )

    except Exception as e:
        logger.error(f"Exception occurred while processing quad {quad_name}: {e}")
        if quad_dir.exists():
            shutil.rmtree(quad_dir)
        raise e


def load_and_filter_data(available_planet_csv, google_index, planet_index):
    """
    Load index files and filter Google tiles based on Planet quad availability.

    Returns:
        dict: Mapping of (quad_name, year) -> list of Google tile paths
    """
    # Load available planet images
    logger.info(f"Loading available Planet images from {available_planet_csv}")
    planet_df = pd.read_csv(available_planet_csv)

    # Extract year from date column (format: YYYY-qQ)
    planet_df["year"] = planet_df["date"].str.split("-").str[0].astype(int)

    # Load planet index to get geometries
    logger.info(f"Loading Planet index from {planet_index}")
    planet_gdf = gpd.read_file(planet_index)

    # Create 'quad' column from 'filename' (new schema) or 'data' (legacy schema).
    if "filename" in planet_gdf.columns:
        planet_gdf["quad"] = planet_gdf["filename"].apply(
            lambda x: Path(x).stem if isinstance(x, str) else None
        )
    elif "data" in planet_gdf.columns:
        planet_gdf["quad"] = planet_gdf["data"].apply(
            lambda x: Path(x).stem if isinstance(x, str) else None
        )
    elif "quad" not in planet_gdf.columns:
        raise ValueError("Planet index must have 'filename', 'data', or 'quad' column.")

    # Ensure geometry is properly loaded
    if planet_gdf.crs is None:
        planet_gdf = planet_gdf.set_crs("EPSG:4326")

    # Merge to get available quads with geometries and years
    available_quads = planet_df.merge(
        planet_gdf, left_on="file", right_on="quad", how="inner"
    )
    logger.info(f"Found {len(available_quads)} available Planet quads")

    # Load Google index
    logger.info(f"Loading Google index from {google_index}")
    google_gdf = pd.read_feather(google_index)
    google_gdf["geometry"] = google_gdf["geometry"].apply(wkb.loads)
    google_gdf = gpd.GeoDataFrame(google_gdf, geometry="geometry")

    # Extract year from tile_path (format: XXX_YYYY_MM_DD/tile_*.tif)
    # First split by '/' to get directory, then split by '_' to get year component
    google_gdf["year"] = (
        google_gdf["tile_path"].str.split("/").str[0].str.split("_").str[1]
    )
    google_gdf["year"] = pd.to_numeric(google_gdf["year"], errors="coerce")

    # Filter out tiles without valid years (0.01% of data)
    valid_year_mask = google_gdf["year"].notna()
    if not valid_year_mask.all():
        logger.warning(
            f"Filtering out {(~valid_year_mask).sum()} tiles without valid year information"
        )
        google_gdf = google_gdf[valid_year_mask].copy()

    google_gdf["year"] = google_gdf["year"].astype(int)

    # Group by quad and year
    quad_year_to_tiles = {}

    for _, planet_row in tqdm(
        available_quads.iterrows(),
        total=len(available_quads),
        desc="Matching Google tiles to Planet quads",
    ):
        quad_name = planet_row["file"]
        quad_year = planet_row["year"]
        quad_geom = planet_row["geometry"]

        # Filter Google tiles by year
        google_year = google_gdf[google_gdf["year"] == quad_year].copy()

        if len(google_year) == 0:
            logger.warning(f"No Google tiles found for year {quad_year}")
            continue

        # Optimize: Group by CRS and reproject in batches
        google_year_4326_parts = []
        for crs, group in google_year.groupby("crs"):
            try:
                # Batch reproject all tiles with the same CRS
                group_gdf = gpd.GeoDataFrame(group, geometry="geometry", crs=crs)
                group_4326 = group_gdf.to_crs("EPSG:4326")
                google_year_4326_parts.append(group_4326[["tile_path", "geometry"]])
            except Exception as e:
                logger.warning(f"Failed to reproject tiles with CRS {crs}: {e}")
                continue

        if not google_year_4326_parts:
            logger.warning(f"No Google tiles could be reprojected for year {quad_year}")
            continue

        # Concatenate all reprojected tiles
        google_year_4326_gdf = pd.concat(google_year_4326_parts, ignore_index=True)
        google_year_4326_gdf = gpd.GeoDataFrame(
            google_year_4326_gdf, geometry="geometry", crs="EPSG:4326"
        )

        # Find intersecting Google tiles
        intersecting = google_year_4326_gdf[google_year_4326_gdf.intersects(quad_geom)]

        if len(intersecting) > 0:
            key = (quad_name, quad_year)
            quad_year_to_tiles[key] = {
                "tile_paths": intersecting["tile_path"].tolist(),
                "geometry": quad_geom,
            }
            logger.info(
                f"Quad {quad_name} ({quad_year}): {len(intersecting)} Google tiles"
            )

    logger.info(f"Total quad-year combinations to process: {len(quad_year_to_tiles)}")
    return quad_year_to_tiles


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Process Google 2.5D tiles for Planet quad intersections"
    )
    parser.add_argument(
        "--available-planet-csv",
        required=True,
        help="Path to CSV file listing available Planet quads",
    )
    parser.add_argument(
        "--google-index",
        required=True,
        help="Path to feather file with Google tile index",
    )
    parser.add_argument(
        "--planet-index",
        required=True,
        help="Path to GPKG file with Planet quad geometries",
    )
    parser.add_argument(
        "--urls-file",
        required=True,
        help="Path to text file containing full URLs (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save processed output files",
    )

    args = parser.parse_args()

    # Set the PROJ_LIB environment variable
    find_proj_lib()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_path = Path(tempfile.mkdtemp(prefix="google_processing_"))

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Temporary directory: {temp_path}")

    # Load URLs file
    tile_id_to_url = load_urls_file(args.urls_file)

    # Load and filter data
    quad_year_to_tiles = load_and_filter_data(
        args.available_planet_csv, args.google_index, args.planet_index
    )

    if not quad_year_to_tiles:
        logger.error("No quad-year combinations to process")
        return

    # Check which quads are already processed
    items_to_process = []
    for (quad_name, year), data in quad_year_to_tiles.items():
        output_file = output_dir / str(year) / f"{quad_name}.tif"
        if not output_file.exists():
            items_to_process.append((quad_name, year, data))
        else:
            logger.info(f"Skipping {quad_name} ({year}) - already processed")

    logger.info(f"Quads to process: {len(items_to_process)}")

    # Process quads sequentially
    for quad_name, year, data in tqdm(items_to_process, desc="Processing Quads"):
        try:
            process_quad(
                quad_name,
                data["tile_paths"],
                data["geometry"],
                temp_path,
                output_dir,
                year,
                tile_id_to_url,
            )
        except Exception as e:
            logger.error(f"Error processing quad {quad_name} ({year}): {e}")

    # Clean up temp directory
    if temp_path.exists():
        shutil.rmtree(temp_path)

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
