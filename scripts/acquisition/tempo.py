"""
Download TEMPO building density tiles (COGs) from AI for Good Open Data.

Progress bars
-------------
- For multi-file downloads: shows a tile-level progress bar (N tiles completed).
- For a single file: shows a byte-level progress bar (MB downloaded), when possible.

Examples
--------
# Download everything (all locations, all quarters) into ./data/tempo_tiles
python scripts/acquisition/tempo.py download-all

# Download one tile (shows per-file byte progress bar)
python scripts/acquisition/tempo.py download-one --location nakuru --quarter 2024q1

# Download one quarter for all locations
python scripts/acquisition/tempo.py download-quarter --quarter 2024q1

# Download all quarters for one location
python scripts/acquisition/tempo.py download-location --location nakuru

# Download a custom subset
python scripts/acquisition/tempo.py download-subset --locations nakuru bamako --quarters 2024q1 2024q2

# Verify what you have on disk
python scripts/acquisition/tempo.py verify
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


# ----------------------------
# Configuration (edit if needed)
# ----------------------------

LOCATIONS: List[str] = [
    "bamako",
    "guangdong_province",
    "guatemala_department",
    "lusaka_district",
    "nakuru",
]

# from 2020q2 up to 2025q2 (inclusive)
QUARTERS: List[str] = [
    f"{year}q{q}"
    for year in range(2020, 2026)
    for q in range(1, 5)
    if not (year == 2025 and q > 2)
][1:]

BASE_URL = "https://opendata.aiforgood.ai/building-density/locations/{location}/{yq}_cog.tif"
DEFAULT_DOWNLOAD_DIR = Path("data/tempo_tiles")


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class DownloadTask:
    """A single (location, quarter) download target."""
    location: str
    quarter: str


@dataclass
class DownloadReport:
    """Summary stats for a batch download."""
    total: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    failed_items: List[Tuple[str, str, str]] = None

    def __post_init__(self) -> None:
        if self.failed_items is None:
            self.failed_items = []


# ----------------------------
# Core utilities
# ----------------------------

def build_url(location: str, quarter: str) -> str:
    """Construct the remote URL for a given location and quarter."""
    return BASE_URL.format(location=location, yq=quarter)


def output_path(output_dir: Path, location: str, quarter: str) -> Path:
    """Compute the on-disk path for a tile."""
    return output_dir / location / f"{quarter}_cog.tif"


def iter_tasks(locations: Sequence[str], quarters: Sequence[str]) -> List[DownloadTask]:
    """Create a cartesian product of (locations x quarters) as download tasks."""
    return [DownloadTask(loc, q) for loc in locations for q in quarters]


def download_one(
    task: DownloadTask,
    output_dir: Path,
    overwrite: bool = False,
    timeout_s: int = 60,
    chunk_size: int = 1024 * 1024,
    session: Optional[requests.Session] = None,
    show_progress: bool = False,
) -> Tuple[str, str, str]:
    """
    Download a single tile.

    If show_progress=True and Content-Length is available, shows a byte-level tqdm bar.

    Returns
    -------
    (status, location, quarter)
        status is one of: "downloaded", "skipped", "failed: <msg>"
    """
    out_path = output_path(output_dir, task.location, task.quarter)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return ("skipped", task.location, task.quarter)

    url = build_url(task.location, task.quarter)
    sess = session or requests.Session()

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    pbar = None

    try:
        with sess.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()

            total_bytes = int(r.headers.get("Content-Length", 0) or 0)

            # Only show per-file progress when explicitly enabled (e.g., download-one).
            if show_progress and total_bytes > 0:
                pbar = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"{task.location}/{task.quarter}",
                    leave=True,
                )

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))

        # Close progress bar before replacing file on disk
        if pbar is not None:
            pbar.close()

        tmp_path.replace(out_path)
        return ("downloaded", task.location, task.quarter)

    except Exception as e:
        if pbar is not None:
            pbar.close()

        # Clean up partial file if present
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        return (f"failed: {e}", task.location, task.quarter)


def download_many(
    tasks: Sequence[DownloadTask],
    output_dir: Path,
    overwrite: bool = False,
    max_workers: int = 8,
    timeout_s: int = 60,
) -> DownloadReport:
    """
    Download many tiles in parallel with a tile-level progress bar.

    Note: We intentionally do NOT show per-file byte progress here to avoid
    messy nested progress bars during parallel execution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport(total=len(tasks))

    def _worker(t: DownloadTask) -> Tuple[str, str, str]:
        # Per-thread Session for better performance vs creating a new TCP connection per request.
        with requests.Session() as s:
            return download_one(
                t,
                output_dir=output_dir,
                overwrite=overwrite,
                timeout_s=timeout_s,
                session=s,
                show_progress=False,  # keep multi-download output clean
            )

    with tqdm(total=len(tasks), desc="Downloading tiles", unit="tile") as pbar:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]

            for fut in cf.as_completed(futures):
                status, loc, q = fut.result()

                if status == "downloaded":
                    report.downloaded += 1
                elif status == "skipped":
                    report.skipped_existing += 1
                else:
                    report.failed += 1
                    report.failed_items.append((loc, q, status))

                pbar.set_postfix(
                    downloaded=report.downloaded,
                    skipped=report.skipped_existing,
                    failed=report.failed,
                )
                pbar.update(1)

    return report


def print_report(report: DownloadReport) -> None:
    """Pretty-print a DownloadReport."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total tiles:       {report.total}")
    print(f"Downloaded:        {report.downloaded}")
    print(f"Skipped existing:  {report.skipped_existing}")
    print(f"Failed:            {report.failed}")
    print("=" * 60)

    if report.failed_items:
        print("\nFailed downloads:")
        for loc, q, msg in report.failed_items:
            print(f"  - {loc}/{q}: {msg}")


def verify_downloads(output_dir: Path, locations: Sequence[str] = LOCATIONS) -> dict:
    """Verify downloaded files and summarize counts/sizes by location."""
    if not output_dir.exists():
        return {"exists": False, "output_dir": str(output_dir)}

    summary = {
        "exists": True,
        "output_dir": str(output_dir),
        "total_files": 0,
        "total_size_mb": 0.0,
        "locations": {},
    }

    for loc in locations:
        loc_dir = output_dir / loc
        if not loc_dir.exists():
            continue

        files = sorted(loc_dir.glob("*.tif"))
        size_bytes = sum(p.stat().st_size for p in files)
        quarters = sorted(p.stem.replace("_cog", "") for p in files)

        summary["locations"][loc] = {
            "count": len(files),
            "size_mb": size_bytes / (1024 * 1024),
            "quarters": quarters,
        }
        summary["total_files"] += len(files)
        summary["total_size_mb"] += size_bytes / (1024 * 1024)

    return summary


def print_verification(stats: dict) -> None:
    """Pretty-print verification stats returned by verify_downloads()."""
    if not stats.get("exists", False):
        print(f"Directory does not exist: {stats.get('output_dir')}")
        return

    print("\n" + "=" * 60)
    print("DOWNLOAD VERIFICATION")
    print("=" * 60)
    print(f"Directory:  {stats['output_dir']}")
    print(f"Total files:{stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print("=" * 60)

    for loc, loc_stats in stats["locations"].items():
        qs = loc_stats["quarters"]
        preview = ", ".join(qs[:5])
        more = f" ... ({len(qs)} total)" if len(qs) > 5 else ""
        print(f"\n{loc}:")
        print(f"  Files: {loc_stats['count']}")
        print(f"  Size:  {loc_stats['size_mb']:.2f} MB")
        print(f"  Quarters: {preview}{more}")


# ----------------------------
# CLI helpers
# ----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download TEMPO building density tiles (COGs).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Directory to store tiles (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if a file already exists.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel downloads for multi-file commands (default: 8).",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=60,
        help="HTTP timeout in seconds (default: 60).",
    )

    sub = p.add_subparsers(dest="command", required=True)

    one = sub.add_parser("download-one", help="Download a single tile (shows byte progress).")
    one.add_argument("--location", required=True, choices=LOCATIONS)
    one.add_argument("--quarter", required=True, choices=QUARTERS)

    qtr = sub.add_parser("download-quarter", help="Download all locations for one quarter.")
    qtr.add_argument("--quarter", required=True, choices=QUARTERS)

    loc = sub.add_parser("download-location", help="Download all quarters for one location.")
    loc.add_argument("--location", required=True, choices=LOCATIONS)

    sub.add_parser("download-all", help="Download all locations and all quarters.")

    subp = sub.add_parser("download-subset", help="Download a custom subset.")
    subp.add_argument("--locations", nargs="+", required=True, choices=LOCATIONS)
    subp.add_argument("--quarters", nargs="+", required=True, choices=QUARTERS)

    sub.add_parser("verify", help="Verify downloaded files on disk.")

    return p.parse_args()


def main() -> int:
    """Entry point for CLI usage."""
    args = parse_args()
    out_dir: Path = args.output_dir

    if args.command == "verify":
        stats = verify_downloads(out_dir, locations=LOCATIONS)
        print_verification(stats)
        return 0

    # Single-file path: show a byte-level progress bar.
    if args.command == "download-one":
        task = DownloadTask(args.location, args.quarter)
        with requests.Session() as s:
            status, loc, q = download_one(
                task,
                output_dir=out_dir,
                overwrite=args.overwrite,
                timeout_s=args.timeout_s,
                session=s,
                show_progress=True,  # <-- ensures per-file progress bar
            )
        print(f"{loc}/{q}: {status}")
        return 0 if status == "downloaded" or status == "skipped" else 2

    # Multi-file paths: show tile-level progress bar via download_many.
    if args.command == "download-quarter":
        tasks = iter_tasks(LOCATIONS, [args.quarter])
    elif args.command == "download-location":
        tasks = iter_tasks([args.location], QUARTERS)
    elif args.command == "download-all":
        tasks = iter_tasks(LOCATIONS, QUARTERS)
        print(
            f"Downloading ALL tiles: {len(LOCATIONS)} locations x {len(QUARTERS)} quarters = {len(tasks)} files\n"
            f"Output dir: {out_dir}\n"
        )
    elif args.command == "download-subset":
        tasks = iter_tasks(args.locations, args.quarters)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    report = download_many(
        tasks=tasks,
        output_dir=out_dir,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        timeout_s=args.timeout_s,
    )
    print_report(report)
    return 0 if report.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
