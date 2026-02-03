# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from shapely import unary_union
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from rasterio.crs import CRS


def get_overture_buildings(quad, overture_feathers_path):
    """Load building data from local feather file."""
    feather_path = Path(overture_feathers_path) / f"{quad}.feather"
    if feather_path.exists():
        buildings = gpd.read_feather(feather_path)
        if not buildings.empty:
            buildings.set_crs(epsg=4326, inplace=True)
        return buildings
    return gpd.GeoDataFrame()


def write_raster(density_array, transform, quad, output_path):
    """
    Write raster data to local file.

    Args:
        density_array (np.ndarray): Array containing building density values
        transform (affine.Affine): The raster transform
        quad (str): The quadkey identifier
        output_path (str): Directory path to save the output file
    """
    raster_profile = {
        "driver": "GTiff",
        "height": density_array.shape[0],
        "width": density_array.shape[1],
        "count": 1,  # Single band for density
        "dtype": "float32",
        "crs": CRS.from_epsg(3857),
        "transform": transform,
        "compress": "lzw",
        "predictor": 2,
        "tiled": True,
        "bigtiff": "IF_SAFER",
        "num_threads": "ALL_CPUS",
    }

    try:
        output_file = Path(output_path) / f"{quad}.tif"
        with rasterio.open(output_file, "w", **raster_profile) as dataset:
            dataset.write(density_array.astype("float32"), 1)  # Band 1: density

    except Exception as e:
        print(f"Error writing raster for {quad}: {str(e)}")
        raise


def create_density_map(quad, geom, overture_feathers_path, output_path):

    # Reproject the quad geometry to EPSG:3857
    quad_geom_3857 = (
        gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(epsg=3857).geometry[0]
    )
    minx, miny, maxx, maxy = quad_geom_3857.bounds

    target_width = 512
    target_height = 512

    try:
        # Get the Overture buildings
        buildings_gdf = get_overture_buildings(quad, overture_feathers_path)

        if not buildings_gdf.empty:

            # Project to EPSG:3857
            buildings_gdf = buildings_gdf.to_crs(epsg=3857)

            # Create grid cells over the quad area
            x_coords = np.linspace(minx, maxx, num=target_width + 1)
            y_coords = np.linspace(maxy, miny, num=target_height + 1)  # Top to Bottom
            grid_cells = [
                box(x0, y0, x1, y1)
                for y0, y1 in zip(y_coords[:-1], y_coords[1:])
                for x0, x1 in zip(x_coords[:-1], x_coords[1:])
            ]

            # Create GeoDataFrame with unique cell identifiers
            grid = gpd.GeoDataFrame(
                {"cell_id": np.arange(len(grid_cells)), "geometry": grid_cells},
                crs="EPSG:3857",
            )

            # Calculate area of each grid cell (assuming rectangular cells)
            grid["cell_area"] = grid.geometry.area

            # Perform spatial overlay to get intersections
            intersected = gpd.overlay(
                grid, buildings_gdf, how="intersection", keep_geom_type=False
            )

            if not intersected.empty:
                # Retain only Polygon and MultiPolygon geometries
                intersected = intersected[
                    intersected.geometry.type.isin(["Polygon", "MultiPolygon"])
                ]

                # Union geometries by cell_id and calculate areas
                overlap_areas = (
                    intersected.groupby("cell_id")
                    .agg({"geometry": unary_union})
                    .reset_index()
                )
                overlap_areas["overlap_area"] = overlap_areas["geometry"].area

                # Merge and calculate fractions for density
                grid = grid.merge(
                    overlap_areas[["cell_id", "overlap_area"]], on="cell_id", how="left"
                )
                grid["overlap_area"] = grid["overlap_area"].fillna(0.0)
                grid["fraction"] = grid["overlap_area"] / grid["cell_area"]
            else:
                # No overlaps found, set fraction to 0
                grid["fraction"] = 0.0

            # Reshape the fractions into 2D array
            fractions_array = grid["fraction"].values.reshape(
                (target_height, target_width)
            )
        else:
            # Handle cases with no buildings (write zeros for density)
            fractions_array = np.zeros((target_height, target_width), dtype="float32")

        # Create transform
        pixel_size_x = (maxx - minx) / target_width
        pixel_size_y = (maxy - miny) / target_height
        transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

        # Write density band to local file
        write_raster(fractions_array, transform, quad, output_path)

    except Exception as e:
        print(f"Error processing {quad}: {str(e)}")
        raise


def main(quads, geoms, overture_feathers_path, output_path):
    """Process all quads sequentially."""
    for quad, geom in tqdm(zip(quads, geoms), total=len(quads)):
        try:
            create_density_map(quad, geom, overture_feathers_path, output_path)
        except Exception as exc:
            print(f"{quad} generated an exception: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Overture building density maps from local feather files."
    )
    parser.add_argument(
        "--overture_feathers_path",
        type=str,
        required=True,
        help="Path to directory containing quad-level feather files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to directory where density map GeoTIFFs will be saved.",
    )
    args = parser.parse_args()

    # Validate paths
    overture_feathers_path = Path(args.overture_feathers_path)
    output_path = Path(args.output_path)

    if not overture_feathers_path.exists():
        raise FileNotFoundError(
            f"Overture feathers path does not exist: {overture_feathers_path}"
        )

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # List all feather files in the input directory
    feather_files = list(overture_feathers_path.glob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No feather files found in {overture_feathers_path}")

    print(f"Found {len(feather_files)} feather files to process.")

    # Extract quad names from filenames (without extension)
    quad_names = [f.stem for f in feather_files]

    # For each quad, we need to get its geometry
    # We'll load each feather file to get the geometry
    quad_geoms = []
    valid_quad_names = []

    for quad_name, feather_file in tqdm(
        zip(quad_names, feather_files), total=len(quad_names)
    ):
        try:
            gdf = gpd.read_feather(feather_file)
            if not gdf.empty:
                geom = gdf.geometry.union_all()
                quad_geoms.append(geom)
                valid_quad_names.append(quad_name)
        except Exception as e:
            print(f"Error loading {feather_file}: {e}")

    print(f"Successfully loaded {len(valid_quad_names)} valid quads.")

    # Launch the main function
    main(valid_quad_names, quad_geoms, overture_feathers_path, output_path)
