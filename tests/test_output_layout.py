"""Tests for the selectable output layout.

Mirroring the rawData tree used to be forced. These cover the three modes the
Preprocessing tab now offers, and the backward compatibility of the older
``mirror_raw_hierarchy`` boolean.

Paths are built with ``pathlib`` rather than literal separators so the assertions
hold on both Windows and Linux.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropyguin.preprocessing import (
    OUTPUT_LAYOUT_EXACT,
    OUTPUT_LAYOUT_MIRROR,
    OUTPUT_LAYOUT_RUN_FOLDER,
    OUTPUT_LAYOUTS,
    default_pipeline_ks_output_dir,
    default_pipeline_output_dir,
    default_pipeline_raw_output_layout,
    describe_output_layout,
    mirrored_concat_base_dir,
    normalize_output_layout,
)


ROOT = Path("/data/processedData")
RAW_BIN = Path(
    "/data/rawData/mPFC-NAc/51543/mPFC_NAc_week1/object/run_g0/run_g0_imec0/run_g0_t0.imec0.ap.bin"
)


def test_normalize_falls_back_to_the_legacy_boolean() -> None:
    assert normalize_output_layout(None, mirror_raw_hierarchy=True) == OUTPUT_LAYOUT_MIRROR
    assert normalize_output_layout("", mirror_raw_hierarchy=False) == OUTPUT_LAYOUT_RUN_FOLDER
    assert normalize_output_layout("nonsense", mirror_raw_hierarchy=True) == OUTPUT_LAYOUT_MIRROR


@pytest.mark.parametrize("mode", OUTPUT_LAYOUTS)
def test_normalize_is_idempotent_and_case_insensitive(mode: str) -> None:
    assert normalize_output_layout(mode) == mode
    assert normalize_output_layout(mode.upper()) == mode
    assert describe_output_layout(mode)


def test_mirror_layout_reproduces_the_raw_session_tree() -> None:
    got = default_pipeline_output_dir(
        str(RAW_BIN), ROOT, run_name="run", layout=OUTPUT_LAYOUT_MIRROR
    )
    assert got == ROOT / "mPFC-NAc" / "51543" / "mPFC_NAc_week1" / "object" / "spike_sorting"


def test_run_folder_layout_uses_the_output_root_and_run_name() -> None:
    got = default_pipeline_output_dir(
        str(RAW_BIN), ROOT, run_name="run_g0", layout=OUTPUT_LAYOUT_RUN_FOLDER
    )
    assert got == ROOT / "run_g0"


def test_exact_layout_writes_into_the_chosen_folder_itself() -> None:
    got = default_pipeline_output_dir(
        str(RAW_BIN), ROOT, run_name="run_g0", layout=OUTPUT_LAYOUT_EXACT
    )
    assert got == ROOT


def test_exact_layout_ignores_the_run_name_for_every_run() -> None:
    """Two different runs resolve to the same folder: that is the point, and the risk."""
    first = default_pipeline_output_dir(str(RAW_BIN), ROOT, run_name="a", layout=OUTPUT_LAYOUT_EXACT)
    second = default_pipeline_output_dir(str(RAW_BIN), ROOT, run_name="b", layout=OUTPUT_LAYOUT_EXACT)
    assert first == second == ROOT


def test_mirror_falls_back_to_the_run_folder_without_a_raw_hierarchy() -> None:
    stray = Path("/somewhere/else/run_g0_t0.imec0.ap.bin")
    got = default_pipeline_output_dir(str(stray), ROOT, run_name="run", layout=OUTPUT_LAYOUT_MIRROR)
    assert got == ROOT / "run"


def test_layout_flows_through_to_the_kilosort_folder() -> None:
    extracted, ks_folder = default_pipeline_raw_output_layout(
        str(RAW_BIN), ROOT, "ks4", "0", run_name="run_g0", layout=OUTPUT_LAYOUT_RUN_FOLDER
    )
    assert extracted == ROOT / "run_g0"
    assert ks_folder == ROOT / "run_g0" / "imec0_ks4"

    exact_ks = default_pipeline_ks_output_dir(
        str(RAW_BIN), "ks4", "0", output_root=ROOT, run_name="run_g0", layout=OUTPUT_LAYOUT_EXACT
    )
    assert exact_ks == ROOT / "imec0_ks4"


def test_layout_argument_wins_over_the_legacy_boolean() -> None:
    got = default_pipeline_output_dir(
        str(RAW_BIN),
        ROOT,
        run_name="run_g0",
        mirror_raw_hierarchy=True,
        layout=OUTPUT_LAYOUT_RUN_FOLDER,
    )
    assert got == ROOT / "run_g0"


def test_legacy_boolean_calls_are_unchanged() -> None:
    """Callers that never pass a layout must behave exactly as before."""
    mirrored = default_pipeline_output_dir(str(RAW_BIN), ROOT, run_name="run", mirror_raw_hierarchy=True)
    flat = default_pipeline_output_dir(str(RAW_BIN), ROOT, run_name="run", mirror_raw_hierarchy=False)
    assert mirrored == ROOT / "mPFC-NAc" / "51543" / "mPFC_NAc_week1" / "object" / "spike_sorting"
    assert flat == ROOT / "run"


def test_concat_destination_follows_the_selected_layout() -> None:
    mirrored = mirrored_concat_base_dir(RAW_BIN, ROOT, "combined", layout=OUTPUT_LAYOUT_MIRROR)
    assert mirrored == ROOT / "mPFC-NAc" / "51543" / "mPFC_NAc_week1" / "combined"

    per_run = mirrored_concat_base_dir(RAW_BIN, ROOT, "combined", layout=OUTPUT_LAYOUT_RUN_FOLDER)
    assert per_run == ROOT / "combined"

    exact = mirrored_concat_base_dir(RAW_BIN, ROOT, "combined", layout=OUTPUT_LAYOUT_EXACT)
    assert exact == ROOT


def test_concat_without_a_layout_keeps_the_legacy_fallback() -> None:
    got = mirrored_concat_base_dir(RAW_BIN, ROOT, "combined", mirror_raw_hierarchy=False)
    assert got == RAW_BIN.parents[2]


def test_concat_non_mirror_layouts_need_an_output_root() -> None:
    got = mirrored_concat_base_dir(RAW_BIN, "", "combined", layout=OUTPUT_LAYOUT_RUN_FOLDER)
    assert got == RAW_BIN.parents[2]
