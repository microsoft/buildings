# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Overture buildings per Planet quad using DuckDB over Azure."
    )
    parser.add_argument(
        "--planet_index",
        required=True,
        help="Path to GPKG file containing quad geometries.",
    )
    parser.add_argument(
        "--available_imagery",
        required=True,
        help="Path to CSV with available quad names (column: 'file').",
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Overture release code (e.g. 2025-10-22.0).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where quad-level feather files will be written.",
    )
    return parser.parse_args()


def init_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    num_cores = os.cpu_count() or 1
    con.execute(f"PRAGMA threads={num_cores};")

    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL azure; LOAD azure;")
    con.execute("SET azure_transport_option_type = 'curl';")
    con.execute(
        """
        CREATE SECRET IF NOT EXISTS overture_anon (
            TYPE azure,
            PROVIDER config,
            ACCOUNT_NAME 'overturemapswestus2'
        );
    """
    )
    return con


def load_quads_of_interest(
    planet_index_path: str, available_imagery_path: str
) -> gpd.GeoDataFrame:
    gdf_index = gpd.read_file(planet_index_path)

    # Create 'quad' column from 'data' column (extract filename without extension)
    if "data" in gdf_index.columns:
        gdf_index["quad"] = gdf_index["data"].apply(
            lambda x: Path(x).stem if isinstance(x, str) else None
        )
    elif "quad" not in gdf_index.columns:
        raise ValueError("Planet index must have either 'data' or 'quad' column.")

    # Ensure CRS is WGS84 for compatibility with Overture (GeoParquet is in EPSG:4326)
    if gdf_index.crs is not None and gdf_index.crs.to_epsg() != 4326:
        gdf_index = gdf_index.to_crs(epsg=4326)

    available = pd.read_csv(available_imagery_path)
    if "file" not in available.columns:
        raise ValueError(
            "available_imagery CSV must have a 'file' column with quad names."
        )

    available_quads = available["file"].astype(str).str.strip().unique()

    if "quad" not in gdf_index.columns:
        raise ValueError("Planet index must have a 'quad' column.")

    gdf_index["quad"] = gdf_index["quad"].astype(str)

    gdf_quads = gdf_index[gdf_index["quad"].isin(available_quads)].copy()

    if gdf_quads.empty:
        raise ValueError(
            "No overlapping quads between planet_index and available_imagery."
        )

    return gdf_quads


def duckdb_query_for_quad(
    con: duckdb.DuckDBPyConnection,
    parquet_path_pattern: str,
    quad_geom: BaseGeometry,
) -> pd.DataFrame:
    # Bounding box filter for coarse pruning
    xmin, ymin, xmax, ymax = quad_geom.bounds
    quad_wkt = quad_geom.wkt

    query = f"""
        SELECT ST_AsText(geometry) AS geometry
        FROM read_parquet('{parquet_path_pattern}')
        WHERE
            bbox.xmin < ? AND bbox.xmax > ?
            AND bbox.ymin < ? AND bbox.ymax > ?
            AND ST_INTERSECTS(
                ST_GeomFromText(?),
                geometry
            )
    """

    params = [xmax, xmin, ymax, ymin, quad_wkt]

    result_df = con.execute(query, params).df()
    return result_df


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load quads
    quads_gdf = load_quads_of_interest(args.planet_index, args.available_imagery)

    # DuckDB connection
    con = init_duckdb_connection()

    # Overture buildings on Azure via DuckDB azure filesystem
    parquet_pattern = (
        f"az://overturemapswestus2.blob.core.windows.net/"
        f"release/{args.release}/theme=buildings/type=building/*.parquet"
    )

    total = len(quads_gdf)
    for i, row in enumerate(quads_gdf.itertuples(index=False), start=1):
        quad_name = getattr(row, "quad")
        quad_geom = getattr(row, "geometry")

        if not isinstance(quad_geom, BaseGeometry):
            print(f"[{i}/{total}] Skipping quad {quad_name}: invalid geometry type.")
            continue

        out_path = output_dir / f"{quad_name}.feather"
        if out_path.exists():
            print(f"[{i}/{total}] Quad {quad_name}: output exists, skipping.")
            continue

        print(f"[{i}/{total}] Processing quad {quad_name}...")

        try:
            df = duckdb_query_for_quad(con, parquet_pattern, quad_geom)
            geom = gpd.GeoSeries.from_wkt(df["geometry"], crs="EPSG:4326")
            attrs = df.drop(columns=["geometry"])
            gdf = gpd.GeoDataFrame(attrs, geometry=geom, crs="EPSG:4326")
        except Exception as e:
            print(f"  Error querying Overture for quad {quad_name}: {e}")
            continue

        try:
            gdf.to_feather(out_path)
            print(f"  Wrote {len(df)} buildings to {out_path}")
        except Exception as e:
            print(f"  Error writing feather for quad {quad_name}: {e}")


if __name__ == "__main__":
    main()
