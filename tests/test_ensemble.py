# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
from pathlib import Path

# Make the repo root importable so `tempo` and `scripts` resolve correctly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import rasterio
import rasterio.transform
from unittest.mock import patch, MagicMock

from tempo.postprocess import create_ranked_mask, downsample_ranked_mask, ensemble_pixels
from scripts.ensemble import set_up_parser, process_scene


# ---------------------------------------------------------------------------
# create_ranked_mask — UDM v2 (2022q1+)
# ---------------------------------------------------------------------------

def test_create_ranked_mask_shape():
    udm = np.zeros((8, 16, 16), dtype=np.float32)
    result = create_ranked_mask(udm, timestamp="2023q1")
    assert result.shape == (16, 16)


def test_create_ranked_mask_values_v2():
    """2×2 patch, each pixel exercises a different rank (UDM v2, third_set)."""
    udm = np.zeros((8, 2, 2), dtype=np.float32)

    # (0,0): high confidence + clear → rank 4
    udm[0, 0, 0] = 1    # Band 1 clear
    udm[6, 0, 0] = 100  # Band 7 confidence >= 95

    # (0,1): low confidence + clear → rank 3
    udm[0, 0, 1] = 1    # clear
    udm[6, 0, 1] = 50   # confidence < 95

    # (1,0): high confidence + not clear → rank 1
    udm[0, 1, 0] = 0    # not clear
    udm[6, 1, 0] = 100  # confidence >= 95

    # (1,1): low confidence + not clear → rank 2
    udm[0, 1, 1] = 0    # not clear
    udm[6, 1, 1] = 50   # confidence < 95

    result = create_ranked_mask(udm, confidence_threshold=95, timestamp="2023q1")
    assert result[0, 0] == 4
    assert result[0, 1] == 3
    assert result[1, 0] == 1
    assert result[1, 1] == 2


def test_create_ranked_mask_values_v1():
    """UDM v1 (second_set): band 0 == 0 means clear (inverted), all pixels high-confidence."""
    # Single-band UDM v1: pixel (0,0) = 0 → clear; pixel (1,1) = 1 → not clear.
    udm = np.ones((1, 2, 2), dtype=np.float32)
    udm[0, 0, 0] = 0  # clear pixel

    result = create_ranked_mask(udm, confidence_threshold=95, timestamp="2021q4")
    # (0,0): clear + high confidence (100) → rank 4
    assert result[0, 0] == 4
    # (0,1): not clear + high confidence → rank 1
    assert result[0, 1] == 1
    # (1,0): not clear + high confidence → rank 1
    assert result[1, 0] == 1


def test_create_ranked_mask_first_set_raises():
    """Timestamps before 2019q3 have no UDM — should raise ValueError."""
    udm = np.zeros((8, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2018q4"):
        create_ranked_mask(udm, timestamp="2018q4")


# ---------------------------------------------------------------------------
# downsample_ranked_mask
# ---------------------------------------------------------------------------

def test_downsample_ranked_mask_shape():
    ranked_mask = np.ones((4096, 4096), dtype=np.float32)
    for algorithm in ["mean", "max", "min", "mode"]:
        result = downsample_ranked_mask(ranked_mask, algorithm=algorithm)
        assert result.shape == (512, 512), f"Wrong shape for algorithm={algorithm}"


# ---------------------------------------------------------------------------
# ensemble_pixels
# ---------------------------------------------------------------------------

THRESHOLD = 2 / 255


def test_ensemble_pixels_all_above():
    """All T timestamps are building — result should be median of building values."""
    T, H, W = 4, 4, 4
    model_outputs = np.full((T, H, W), 0.5, dtype=np.float32)
    ranked_masks = np.full((T, H, W), 4.0, dtype=np.float32)
    result = ensemble_pixels(model_outputs, ranked_masks, THRESHOLD, 3.5, "median")
    assert result.shape == (H, W)
    np.testing.assert_allclose(result, 0.5, rtol=1e-5)


def test_ensemble_pixels_all_below():
    """All T timestamps are non-building — result should be all zeros."""
    T, H, W = 4, 4, 4
    model_outputs = np.zeros((T, H, W), dtype=np.float32)
    ranked_masks = np.full((T, H, W), 4.0, dtype=np.float32)
    result = ensemble_pixels(model_outputs, ranked_masks, THRESHOLD, 3.5, "median")
    np.testing.assert_array_equal(result, 0.0)


def test_ensemble_pixels_exactly_2_both_clear():
    """Exactly 2 building timestamps, both UDM-clear → max of building values."""
    T, H, W = 4, 1, 1
    model_outputs = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32).reshape(T, H, W)
    ranked_masks = np.array([4.0, 4.0, 1.0, 1.0], dtype=np.float32).reshape(T, H, W)
    result = ensemble_pixels(model_outputs, ranked_masks, THRESHOLD, 3.5, "median")
    np.testing.assert_allclose(result[0, 0], 0.8, rtol=1e-5)


def test_ensemble_pixels_exactly_2_neither_clear():
    """Exactly 2 building timestamps, neither UDM-clear → zero."""
    T, H, W = 4, 1, 1
    model_outputs = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32).reshape(T, H, W)
    ranked_masks = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(T, H, W)
    result = ensemble_pixels(model_outputs, ranked_masks, THRESHOLD, 3.5, "median")
    np.testing.assert_array_equal(result, 0.0)


@pytest.mark.parametrize("behavior,expected", [
    ("median", np.median([0.8, 0.6, 0.0, 0.0])),
    ("mean",   np.mean([0.8, 0.6, 0.0, 0.0])),
    ("max",    np.max([0.8, 0.6, 0.0, 0.0])),
    ("min",    np.min([0.8, 0.6, 0.0, 0.0])),
])
def test_ensemble_pixels_exactly_2_mixed(behavior, expected):
    """Exactly 2 building timestamps, mixed clarity → default_behavior over all T."""
    T, H, W = 4, 1, 1
    model_outputs = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32).reshape(T, H, W)
    # One clear (4.0 > 3.5), one not clear (1.0 <= 3.5)
    ranked_masks = np.array([4.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(T, H, W)
    result = ensemble_pixels(model_outputs, ranked_masks, THRESHOLD, 3.5, behavior)
    np.testing.assert_allclose(result[0, 0], expected, rtol=1e-5)


def test_ensemble_pixels_dtype():
    """Output should always be float32 regardless of input dtype."""
    T, H, W = 4, 8, 8
    model_outputs = np.random.rand(T, H, W).astype(np.float64)
    ranked_masks = np.random.rand(T, H, W).astype(np.float64)
    result = ensemble_pixels(model_outputs, ranked_masks, THRESHOLD, 3.5, "median")
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# UDM v2 cutoff routing
# ---------------------------------------------------------------------------

def test_udm_v2_cutoff_routing():
    """Timestamps before the cutoff use the primary template; >= cutoff use v2."""
    called_urls = []

    class _FakeSrc:
        profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
                   "width": 512, "height": 512, "crs": "EPSG:32632",
                   "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 512, 512)}
        crs = "EPSG:32632"
        bounds = (0.0, 0.0, 1.0, 1.0)
        count = 1  # single-band label — no height path
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, *a): return np.ones((512, 512), dtype=np.float32)

    class _FakeWriteDst:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def write(self, *a): pass

    def _fake_open(url, *args, **kwargs):
        called_urls.append(str(url))
        if args and args[0] == "w":
            return _FakeWriteDst()
        return _FakeSrc()

    with (
        patch("scripts.ensemble.rasterio.open", side_effect=_fake_open),
        patch("scripts.ensemble.transform_bounds", return_value=(0.0, 0.0, 1.0, 1.0)),
        patch("scripts.ensemble.create_ranked_mask", return_value=np.ones((512, 512))),
        patch("scripts.ensemble.downsample_ranked_mask", return_value=np.ones((512, 512))),
        patch("scripts.ensemble._load_mask", return_value=None),
    ):
        from scripts.ensemble import process_scene
        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            process_scene(
                scene="SC",
                timestamps=["2021q4", "2022q1"],
                model="mymodel",
                save_fp=Path(tmp) / "out.tif",
                output_sas="osas",
                udm_sas="usas",
                udm_url_template="https://old/{timestamp}/{scene}.tif?{sas}",
                udm_url_template_v2="https://new/{timestamp}/{scene}.tif?{sas}",
                udm_v2_cutoff="2022q1",
                water=False,
                elevation=False,
            )

    udm_urls = [u for u in called_urls if "old/" in u or "new/" in u]
    assert any("old/" in u and "2021q4" in u for u in udm_urls), "2021q4 should use primary template"
    assert any("new/" in u and "2022q1" in u for u in udm_urls), "2022q1 should use v2 template"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_defaults():
    parser = set_up_parser()
    args = parser.parse_args([
        "--input-csv", "data.csv",
        "--model", "9-37-best_practices_p3",
        "--save-dir", "outputs/",
        "--save-folder", "v7",
        "--timestamps", "2023q1",
    ])
    assert args.workers == 1
    assert args.confidence_threshold == 95
    assert args.averaging_algorithm == "mean"
    assert args.small_value_threshold == pytest.approx(2 / 255)
    assert args.small_height_threshold == pytest.approx(0.024)
    assert args.clarity_threshold == 3.5
    assert args.default_behavior == "median"
    assert args.elevation_threshold == 5100
    assert args.no_water is False
    assert args.no_elevation is False
    assert args.skip_existing is False
    assert args.water_url_template is None
    assert args.elevation_url_template is None


def test_parser_timestamps():
    parser = set_up_parser()
    args = parser.parse_args([
        "--input-csv", "data.csv",
        "--model", "az_63_ot",
        "--save-dir", "outputs/",
        "--save-folder", "v3",
        "--timestamps", "2023q1", "2023q2",
    ])
    assert args.timestamps == ["2023q1", "2023q2"]


def test_parser_no_water_flag():
    parser = set_up_parser()
    args = parser.parse_args([
        "--input-csv", "data.csv",
        "--model", "az_63_ot",
        "--save-dir", "outputs/",
        "--save-folder", "v3",
        "--timestamps", "2023q1",
        "--no-water",
    ])
    assert args.no_water is True


# ---------------------------------------------------------------------------
# Integration tests (mocked I/O)
# ---------------------------------------------------------------------------

def test_process_scene_skips_existing(tmp_path):
    """When skip_existing=True and the output exists, rasterio.open is never called."""
    save_fp = tmp_path / "output.tif"
    save_fp.touch()  # create the file so .exists() returns True

    with patch("scripts.ensemble.rasterio.open") as mock_open:
        process_scene(
            scene="test_scene",
            timestamps=["2023q1"],
            model="az_63_ot",
            save_fp=save_fp,
            output_sas="fake_sas",
            udm_sas="fake_udm_sas",
            skip_existing=True,
        )
    mock_open.assert_not_called()


def test_process_scene_exception_handling(tmp_path):
    """A rasterio error should be caught and logged — process_scene must not re-raise."""
    save_fp = tmp_path / "output.tif"

    with patch("scripts.ensemble.rasterio.open", side_effect=RuntimeError("network error")):
        # Should complete without raising.
        process_scene(
            scene="test_scene",
            timestamps=["2023q1"],
            model="az_63_ot",
            save_fp=save_fp,
            output_sas="fake_sas",
            udm_sas="fake_udm_sas",
        )


def test_process_scene_writes_output(tmp_path):
    """With mocked reads, process_scene writes both bands; water pixels are -1."""
    fake_density = np.ones((512, 512), dtype=np.float32) * 0.5
    fake_height = np.ones((512, 512), dtype=np.float32) * 5.0
    fake_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 2,
        "width": 512,
        "height": 512,
        "crs": "EPSG:32632",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 512, 512),
    }

    class _FakeReadSrc:
        def __init__(self):
            self.profile = fake_profile.copy()
            self.crs = "EPSG:32632"
            self.bounds = (0.0, 0.0, 1.0, 1.0)
            self.count = 2  # 2-band label TIFF (density + height)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, band=None):
            if band == 1:
                return fake_density
            return fake_height

    written = {}

    class _FakeWriteDst:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, data, band):
            written[band] = data.copy()

    def _fake_open(url, *args, **kwargs):
        if args and args[0] == "w":
            return _FakeWriteDst()
        return _FakeReadSrc()

    save_fp = tmp_path / "output.tif"

    # Simple 2×2 water mask (top-left corner)
    fake_water = np.zeros((512, 512), dtype=bool)
    fake_water[0, 0] = True

    with (
        patch("scripts.ensemble.rasterio.open", side_effect=_fake_open),
        patch("scripts.ensemble.transform_bounds", return_value=(0.0, 0.0, 1.0, 1.0)),
        patch("scripts.ensemble.create_ranked_mask", return_value=np.ones((512, 512))),
        patch("scripts.ensemble.downsample_ranked_mask", return_value=np.ones((512, 512))),
        patch("scripts.ensemble._load_mask", side_effect=[fake_water, None]),
    ):
        process_scene(
            scene="test_scene",
            timestamps=["2023q1", "2023q2", "2023q3", "2023q4"],
            model="az_63_ot",
            save_fp=save_fp,
            output_sas="fake_sas",
            udm_sas="fake_udm_sas",
        )

    assert 1 in written, "dst.write(data, 1) was never called — density band not written"
    assert 2 in written, "dst.write(data, 2) was never called — height band not written"
    assert written[1].dtype == np.float32
    assert written[2].dtype == np.float32
    # Water pixels should be burned as -1.
    assert written[1][0, 0] == pytest.approx(-1.0), "Water pixel in density should be -1"
    assert written[2][0, 0] == pytest.approx(-1.0), "Water pixel in height should be -1"
    # Non-water pixels should be positive.
    assert written[1][1, 1] > 0


def test_process_scene_fewer_than_4_timestamps(tmp_path):
    """With < 4 timestamps, the most recent prediction is used directly."""
    fake_density = np.ones((512, 512), dtype=np.float32) * 0.7
    fake_profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": 512, "height": 512, "crs": "EPSG:32632",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 512, 512),
    }

    class _FakeSrc:
        profile = fake_profile.copy()
        crs = "EPSG:32632"
        bounds = (0.0, 0.0, 1.0, 1.0)
        count = 1
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, band=None):
            # Band arg → 2D (label read); no arg → 3D (UDM read).
            return fake_density if band is not None else fake_density[np.newaxis]

    written = {}

    class _FakeWriteDst:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def write(self, data, band): written[band] = data.copy()

    def _fake_open(url, *args, **kwargs):
        if args and args[0] == "w":
            return _FakeWriteDst()
        return _FakeSrc()

    save_fp = tmp_path / "output.tif"

    with (
        patch("scripts.ensemble.rasterio.open", side_effect=_fake_open),
        patch("scripts.ensemble.transform_bounds", return_value=(0.0, 0.0, 1.0, 1.0)),
        patch("scripts.ensemble.create_ranked_mask", return_value=np.ones((512, 512))),
        patch("scripts.ensemble.downsample_ranked_mask", return_value=np.ones((512, 512))),
        patch("scripts.ensemble._load_mask", return_value=None),
    ):
        process_scene(
            scene="test_scene",
            timestamps=["2023q1", "2023q2"],  # only 2 timestamps
            model="az_63_ot",
            save_fp=save_fp,
            output_sas="fake_sas",
            udm_sas="fake_udm_sas",
            water=False,
            elevation=False,
        )

    assert 1 in written
    np.testing.assert_allclose(written[1], 0.7, rtol=1e-5)
