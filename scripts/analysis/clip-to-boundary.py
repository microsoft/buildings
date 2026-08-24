# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Clip the global building density & height layer to an area of interest.

This script takes an area of interest (any OGR-readable vector file, or a country
ISO3 code), finds the Planet L15 quads covering it in the public GeoPackage tile
index, mosaics those Cloud-Optimized GeoTIFFs directly over HTTP with GDAL's
``/vsicurl/`` driver, clips the mosaic to the boundary, and writes a single
two-band COG. Only the tiles overlapping the area of interest are read, and
nothing except the index is downloaded to disk.

The output keeps the source CRS (EPSG:3857), resolution and pixel grid, and is
resampled with nearest neighbour onto a grid-aligned extent, so it is an exact
copy of the source pixels plus boundary masking.

Output bands match the published dataset:
    Band 1 = building density (0 to 1, fraction of pixel covered by buildings)
    Band 2 = building height (0 to 1, multiply by 100 to get meters)
    NoData = -1

Example usage:
1.  Clip to a country by ISO3 code (boundary is fetched automatically):
    python clip-to-boundary.py
        --iso3 LSO
        --quarter 2023q4
        --output lesotho_2023q4.tif

2.  Clip to your own vector file (any OGR format; reprojected as needed):
    python clip-to-boundary.py
        --boundary aoi.gpkg
        --layer districts
        --quarter 2020q4
        --output aoi_2020q4.tif

3.  Add a 1 km margin around the boundary so that focal or zonal statistics near
    the edge are not computed against NoData:
    python clip-to-boundary.py
        --iso3 RWA
        --quarter 2023q4
        --buffer-m 1000
        --output rwanda_2023q4.tif

4.  Reuse an already downloaded copy of the tile index:
    python clip-to-boundary.py
        --boundary aoi.geojson
        --quarter 2023q4
        --index data/tile_index.gpkg
        --output aoi_2023q4.tif

Run with ``--list-quarters`` to see which quarters the tile index exposes.
"""

import argparse
import io
import math
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
import pyogrio
from loguru import logger
from osgeo import gdal
from pyproj import CRS, Geod
from shapely.geometry import LineString

gdal.UseExceptions()

TILE_INDEX_URL = "https://opendata.aiforgood.ai/building-density/tile_index.gpkg"
DEFAULT_INDEX_PATH = os.path.join("data", "tile_index.gpkg")

# fieldmaps.io redistributes geoBoundaries as ready-to-use GeoPackages. The
# "originals" archives hold the true national borders; the "extended" archives are
# edge-matched and deliberately extend past them, so they are not used here.
GEOBOUNDARIES_URL = "https://data.fieldmaps.io/geoboundaries/originals/{code}.gpkg.zip"

WORKING_CRS = 3857
NODATA = -1
BAND_NAMES = ["building_density", "building_height"]
QUARTER_COLUMN_PREFIX = "data_"


def download(url: str, timeout: int = 900) -> bytes:
    """Fetch a URL, sending an explicit User-Agent.

    Some data hosts reject Python's default urllib User-Agent with HTTP 403.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def ensure_tile_index(path: str) -> str:
    """Return a local path to the tile index, downloading it if necessary."""
    if os.path.exists(path):
        logger.info(f"Using tile index at {path}")
        return path

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    logger.info(f"Downloading tile index from {TILE_INDEX_URL} (this is a large file)")
    payload = download(TILE_INDEX_URL)
    with open(path, "wb") as f:
        f.write(payload)
    logger.info(f"Saved tile index to {path}")
    return path


def index_layer(path: str) -> str:
    """Return the name of the single layer in the tile index GeoPackage."""
    layers = pyogrio.list_layers(path)
    if len(layers) != 1:
        raise SystemExit(f"Expected exactly one layer in {path}, found {len(layers)}")
    return layers[0][0]


def available_quarters(path: str, layer: str) -> list:
    """List the quarters the tile index exposes, newest last.

    Quarters are discovered from the index schema rather than hard coded, so new
    releases are picked up without changing this script.
    """
    fields = pyogrio.read_info(path, layer=layer)["fields"]
    quarters = [
        field[len(QUARTER_COLUMN_PREFIX):]
        for field in fields
        if field.startswith(QUARTER_COLUMN_PREFIX)
    ]
    return sorted(quarters)


def load_geoboundaries_adm0(iso3: str) -> gpd.GeoDataFrame:
    """Download a country's ADM0 boundary from fieldmaps.io."""
    code = iso3.lower()
    url = GEOBOUNDARIES_URL.format(code=code)
    logger.info(f"Downloading ADM0 boundary for {iso3.upper()} from {url}")
    archive = zipfile.ZipFile(io.BytesIO(download(url)))
    name = next(n for n in archive.namelist() if n.endswith(".gpkg"))

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, os.path.basename(name))
        with open(local, "wb") as f:
            f.write(archive.read(name))
        boundary = gpd.read_file(local, layer=f"{code}_adm0")

    logger.info(f"Boundary has {len(boundary)} feature(s)")
    return boundary


def mercator_meridian_scale(latitude_deg: float) -> float:
    """Largest ratio of EPSG:3857 units to ground meters at a given latitude.

    EPSG:3857 is not conformal with respect to the WGS84 ellipsoid, so one
    projected unit corresponds to a different ground distance north-south than
    east-west::

        parallel scale  k_p(phi) = sec(phi) * sqrt(1 - e^2 sin^2 phi)
        meridian scale  k_m(phi) = sec(phi) * (1 - e^2 sin^2 phi)^1.5 / (1 - e^2)

    The meridian scale is the larger of the two, so scaling a buffer by it
    guarantees at least the requested ground distance in every direction.
    """
    ellipsoid = CRS.from_epsg(4326).ellipsoid
    flattening = 1.0 / ellipsoid.inverse_flattening
    e2 = 2.0 * flattening - flattening**2
    phi = math.radians(latitude_deg)
    w = 1.0 - e2 * math.sin(phi) ** 2
    return w**1.5 / ((1.0 - e2) * math.cos(phi))


def measure_margin(original: gpd.GeoDataFrame, buffered: gpd.GeoDataFrame, samples: int = 500):
    """Geodesic distances from the original boundary out to the buffered boundary."""
    geod = Geod(ellps="WGS84")
    border = original.to_crs(4326).geometry.union_all().boundary
    outline = buffered.to_crs(4326).geometry.union_all().boundary

    rings = [border] if isinstance(border, LineString) else list(border.geoms)
    longest = max(rings, key=lambda ring: ring.length)

    distances = []
    for fraction in np.linspace(0, 1, samples, endpoint=False):
        point = longest.interpolate(fraction, normalized=True)
        nearest = outline.interpolate(outline.project(point))
        distances.append(geod.inv(point.x, point.y, nearest.x, nearest.y)[2])
    return np.array(distances)


def buffer_boundary(boundary: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Grow a boundary by a ground distance, working in EPSG:3857."""
    projected = boundary.to_crs(WORKING_CRS)
    projected["geometry"] = projected.geometry.make_valid()
    if buffer_m <= 0:
        return projected

    bounds = boundary.to_crs(4326).total_bounds
    latitude = max(abs(bounds[1]), abs(bounds[3]))
    scale = mercator_meridian_scale(latitude)
    logger.info(
        f"Buffering by {buffer_m:,.0f} m ({buffer_m * scale:,.1f} projected units, "
        f"scale {scale:.5f} at {latitude:.2f} degrees)"
    )

    projected["geometry"] = projected.geometry.buffer(
        buffer_m * scale, join_style="round", resolution=32
    )
    projected["geometry"] = projected.geometry.make_valid()

    margins = measure_margin(boundary, projected)
    logger.info(
        f"Margin around boundary: min {margins.min():,.1f} m, "
        f"median {np.median(margins):,.1f} m"
    )
    return projected


def select_tiles(index_path: str, layer: str, boundary: gpd.GeoDataFrame, quarter: str) -> list:
    """Return the tile URLs for the quads intersecting the boundary."""
    column = f"{QUARTER_COLUMN_PREFIX}{quarter}"
    bounds = tuple(boundary.to_crs(WORKING_CRS).total_bounds)

    logger.info("Querying the tile index")
    tiles = pyogrio.read_dataframe(index_path, layer=layer, bbox=bounds)
    if column not in tiles.columns:
        raise SystemExit(f"Tile index has no column {column!r}")

    geometry = boundary.to_crs(tiles.crs).geometry.union_all()
    tiles = tiles[tiles.geometry.intersects(geometry)]
    urls = [url for url in tiles[column].tolist() if url]

    logger.info(f"{len(urls)} tiles intersect the area of interest")
    if not urls:
        raise SystemExit("No tiles overlap the area of interest")
    return urls


def snap_extent(dataset: gdal.Dataset, boundary: gpd.GeoDataFrame):
    """Boundary extent clipped to the mosaic and snapped onto its pixel grid."""
    origin_x, resolution, _, origin_y, _, negative_res = dataset.GetGeoTransform()
    mosaic_max_x = origin_x + resolution * dataset.RasterXSize
    mosaic_min_y = origin_y + negative_res * dataset.RasterYSize

    min_x, min_y, max_x, max_y = boundary.to_crs(WORKING_CRS).total_bounds
    min_x, max_x = max(min_x, origin_x), min(max_x, mosaic_max_x)
    min_y, max_y = max(min_y, mosaic_min_y), min(max_y, origin_y)

    min_x = origin_x + math.floor((min_x - origin_x) / resolution) * resolution
    max_x = origin_x + math.ceil((max_x - origin_x) / resolution) * resolution
    max_y = origin_y - math.floor((origin_y - max_y) / resolution) * resolution
    min_y = origin_y - math.ceil((origin_y - min_y) / resolution) * resolution
    return (min_x, min_y, max_x, max_y), resolution


def run(command: list) -> None:
    """Run a GDAL command line tool, raising if it fails."""
    subprocess.run(command, check=True)


def clip(urls: list, boundary: gpd.GeoDataFrame, quarter: str, output: str,
         buffer_m: float, boundary_source: str) -> None:
    """Mosaic the tiles, clip them to the boundary and write a COG."""
    with tempfile.TemporaryDirectory() as tmp:
        listing = os.path.join(tmp, "tiles.txt")
        with open(listing, "w") as f:
            f.write("\n".join(f"/vsicurl/{url}" for url in urls) + "\n")

        cutline = os.path.join(tmp, "cutline.gpkg")
        boundary.to_crs(WORKING_CRS)[["geometry"]].to_file(
            cutline, layer="cutline", driver="GPKG"
        )

        mosaic = os.path.join(tmp, "mosaic.vrt")
        logger.info("Building a virtual mosaic of the tiles")
        run([
            "gdalbuildvrt", "-q", "-input_file_list", listing,
            "-srcnodata", str(NODATA), "-vrtnodata", str(NODATA), mosaic,
        ])

        dataset = gdal.Open(mosaic, gdal.GA_Update)
        if dataset.RasterCount != len(BAND_NAMES):
            raise SystemExit(
                f"Expected {len(BAND_NAMES)} bands, found {dataset.RasterCount}"
            )
        dataset.SetMetadata({
            "quarter": quarter,
            "band_1": "building density: fraction of pixel covered by buildings (0-1)",
            "band_2": "building height: normalised (0-1), multiply by 100 for meters",
            "nodata_value": str(NODATA),
            "boundary_buffer_m": str(buffer_m),
            "boundary_source": boundary_source,
            "source_tiles": str(len(urls)),
            "source": "Microsoft Building Density & Height Dataset",
        })
        for position, name in enumerate(BAND_NAMES, start=1):
            dataset.GetRasterBand(position).SetDescription(name)
        dataset.FlushCache()

        (min_x, min_y, max_x, max_y), resolution = snap_extent(dataset, boundary)
        del dataset

        width = int(round((max_x - min_x) / resolution))
        height = int(round((max_y - min_y) / resolution))
        logger.info(f"Output grid is {width} x {height} pixels at {resolution:.6f} m")

        clipped = os.path.join(tmp, "clipped.vrt")
        run([
            "gdalwarp", "-q", "-overwrite", "-of", "VRT",
            "-t_srs", f"EPSG:{WORKING_CRS}",
            "-te", str(min_x), str(min_y), str(max_x), str(max_y),
            "-tr", str(resolution), str(resolution),
            "-r", "near",
            "-cutline", cutline, "-cl", "cutline",
            "-srcnodata", str(NODATA), "-dstnodata", str(NODATA),
            mosaic, clipped,
        ])

        logger.info(f"Writing {output}")
        parent = os.path.dirname(os.path.abspath(output))
        os.makedirs(parent, exist_ok=True)
        run([
            "gdal_translate", "-q", "-of", "COG",
            "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=3",
            "-co", "BIGTIFF=IF_SAFER", "-co", "NUM_THREADS=ALL_CPUS",
            "-co", "RESAMPLING=AVERAGE",
            clipped, output,
        ])

    size_mb = os.path.getsize(output) / 1e6
    logger.info(f"Wrote {output} ({size_mb:,.1f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clip the global building density & height layer to an area of interest "
            "and write a single two-band Cloud-Optimized GeoTIFF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--boundary",
        help="Vector file describing the area of interest (any OGR-readable format)",
    )
    source.add_argument(
        "--iso3",
        help="ISO3 country code; the ADM0 boundary is downloaded from fieldmaps.io",
    )
    parser.add_argument(
        "--layer", help="Layer to read from --boundary (defaults to the first layer)"
    )
    parser.add_argument("--quarter", help="Quarter to clip, for example 2023q4")
    parser.add_argument("--output", help="Path of the GeoTIFF to write")
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help="Local path of the tile index; it is downloaded here if missing",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=0.0,
        help="Optional margin in meters to keep around the boundary",
    )
    parser.add_argument(
        "--list-quarters",
        action="store_true",
        help="Print the quarters available in the tile index and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Streaming many small COGs over HTTP is much faster when GDAL does not probe
    # sibling files and caches the ranges it reads.
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "3")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "1")
    for key in (
        "GDAL_DISABLE_READDIR_ON_OPEN",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS",
        "GDAL_HTTP_MAX_RETRY",
        "GDAL_HTTP_RETRY_DELAY",
    ):
        os.environ.setdefault(key, gdal.GetConfigOption(key))

    index_path = ensure_tile_index(args.index)
    layer = index_layer(index_path)
    quarters = available_quarters(index_path, layer)

    if args.list_quarters:
        print("Quarters available in the tile index:")
        for quarter in quarters:
            print(f"  {quarter}")
        return

    if not args.boundary and not args.iso3:
        raise SystemExit("Provide either --boundary or --iso3")
    if not args.quarter:
        raise SystemExit(f"Provide --quarter (one of: {', '.join(quarters)})")
    if not args.output:
        raise SystemExit("Provide --output")
    if args.quarter not in quarters:
        raise SystemExit(
            f"Quarter {args.quarter!r} is not in the tile index "
            f"(available: {', '.join(quarters)})"
        )
    if args.layer and not args.boundary:
        raise SystemExit("--layer only applies to --boundary")

    if args.iso3:
        boundary = load_geoboundaries_adm0(args.iso3)
        boundary_source = f"geoBoundaries ADM0 ({args.iso3.upper()}) via fieldmaps.io"
    else:
        boundary = gpd.read_file(args.boundary, layer=args.layer)
        boundary_source = os.path.basename(args.boundary)
        logger.info(f"Read {len(boundary)} feature(s) from {args.boundary}")

    if boundary.empty:
        raise SystemExit("The area of interest is empty")
    if boundary.crs is None:
        raise SystemExit("The area of interest has no CRS; please set one")

    boundary = buffer_boundary(boundary, args.buffer_m)
    urls = select_tiles(index_path, layer, boundary, args.quarter)
    clip(urls, boundary, args.quarter, args.output, args.buffer_m, boundary_source)


if __name__ == "__main__":
    sys.exit(main())
