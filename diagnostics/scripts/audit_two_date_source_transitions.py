#!/usr/bin/env python3
"""Audit source-path and fused-probability transitions at fixed pixels.

The primary scope is water in 2025 C4 but not in 2024 C4.  The script compares
the saved source flags and probabilities on the same grid without rerunning or
changing any model module.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/two_date_source_transitions"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
SOURCE_NAMES = {
    0: "outside_or_unassigned",
    1: "sar_only",
    2: "optical_only",
    3: "dual_source",
    4: "prior_or_no_observation",
}
THRESHOLDS = (0.50, 0.62, 0.68, 0.78, 0.93)


def aligned(reference, other) -> bool:
    return (
        reference.shape == other.shape
        and reference.crs == other.crs
        and reference.transform.almost_equals(other.transform)
    )


def new_accumulator() -> dict:
    return {
        "pixels": 0,
        "prob24_sum": 0.0,
        "prob25_sum": 0.0,
        "prob24_sumsq": 0.0,
        "prob25_sumsq": 0.0,
        "prob24_ge": {str(value): 0 for value in THRESHOLDS},
        "prob25_ge": {str(value): 0 for value in THRESHOLDS},
    }


def update(acc: dict, selection: np.ndarray, p24: np.ndarray, p25: np.ndarray) -> None:
    count = int(np.count_nonzero(selection))
    if not count:
        return
    values24 = p24[selection].astype("float64")
    values25 = p25[selection].astype("float64")
    finite = np.isfinite(values24) & np.isfinite(values25)
    values24 = values24[finite]
    values25 = values25[finite]
    count = int(values24.size)
    if not count:
        return
    acc["pixels"] += count
    acc["prob24_sum"] += float(values24.sum())
    acc["prob25_sum"] += float(values25.sum())
    acc["prob24_sumsq"] += float(np.square(values24).sum())
    acc["prob25_sumsq"] += float(np.square(values25).sum())
    for threshold in THRESHOLDS:
        key = str(threshold)
        acc["prob24_ge"][key] += int(np.count_nonzero(values24 >= threshold))
        acc["prob25_ge"][key] += int(np.count_nonzero(values25 >= threshold))


def finish(acc: dict, pixel_area: float) -> dict:
    count = acc["pixels"]
    mean24 = acc["prob24_sum"] / count if count else None
    mean25 = acc["prob25_sum"] / count if count else None
    variance24 = max(acc["prob24_sumsq"] / count - mean24 * mean24, 0.0) if count else None
    variance25 = max(acc["prob25_sumsq"] / count - mean25 * mean25, 0.0) if count else None
    return {
        "pixels": count,
        "area_km2": count * pixel_area / 1e6,
        "probability_2024_mean": mean24,
        "probability_2025_mean": mean25,
        "probability_mean_change": mean25 - mean24 if count else None,
        "probability_2024_sd": variance24 ** 0.5 if variance24 is not None else None,
        "probability_2025_sd": variance25 ** 0.5 if variance25 is not None else None,
        **{
            f"probability_2024_ge_{threshold}_fraction": acc["prob24_ge"][str(threshold)] / count
            if count else None
            for threshold in THRESHOLDS
        },
        **{
            f"probability_2025_ge_{threshold}_fraction": acc["prob25_ge"][str(threshold)] / count
            if count else None
            for threshold in THRESHOLDS
        },
    }


def main() -> int:
    paths = {
        "water24": UPSTREAM / DATE_24 / "c4_water_mask.tif",
        "water25": UPSTREAM / DATE_25 / "c4_water_mask.tif",
        "source24": UPSTREAM / DATE_24 / "c2_source_flag.tif",
        "source25": UPSTREAM / DATE_25 / "c2_source_flag.tif",
        "prob24": UPSTREAM / DATE_24 / "c2_fused_probability.tif",
        "prob25": UPSTREAM / DATE_25 / "c2_fused_probability.tif",
    }
    datasets = {name: rasterio.open(path) for name, path in paths.items()}
    try:
        reference = datasets["water24"]
        for name, dataset in datasets.items():
            if not aligned(reference, dataset):
                raise RuntimeError(f"Grid mismatch: {name}")
        pixel_area = abs(float(reference.transform.a * reference.transform.e))

        scopes = {
            "common_water": new_accumulator(),
            "water_2025_only": new_accumulator(),
            "water_2024_only": new_accumulator(),
            "nonwater_both": new_accumulator(),
        }
        transition_acc: dict[tuple[str, int, int], dict] = defaultdict(new_accumulator)
        source_totals: dict[tuple[str, int], int] = defaultdict(int)
        invalid_source_codes = 0

        for _, window in reference.block_windows(1):
            water24 = datasets["water24"].read(1, window=window) > 0
            water25 = datasets["water25"].read(1, window=window) > 0
            source24 = datasets["source24"].read(1, window=window)
            source25 = datasets["source25"].read(1, window=window)
            prob24 = datasets["prob24"].read(1, window=window).astype("float32")
            prob25 = datasets["prob25"].read(1, window=window).astype("float32")
            valid_source = (
                (source24 >= 0) & (source24 <= 4)
                & (source25 >= 0) & (source25 <= 4)
            )
            invalid_source_codes += int(np.count_nonzero(~valid_source))
            scope_masks = {
                "common_water": water24 & water25,
                "water_2025_only": ~water24 & water25,
                "water_2024_only": water24 & ~water25,
                "nonwater_both": ~water24 & ~water25 & (source24 > 0) & (source25 > 0),
            }
            for scope_name, scope_mask in scope_masks.items():
                selection = scope_mask & valid_source
                update(scopes[scope_name], selection, prob24, prob25)
                for code24 in range(5):
                    for code25 in range(5):
                        cell = selection & (source24 == code24) & (source25 == code25)
                        update(transition_acc[(scope_name, code24, code25)], cell, prob24, prob25)
                for code in range(5):
                    source_totals[(f"{scope_name}|2024", code)] += int(
                        np.count_nonzero(selection & (source24 == code))
                    )
                    source_totals[(f"{scope_name}|2025", code)] += int(
                        np.count_nonzero(selection & (source25 == code))
                    )
    finally:
        for dataset in datasets.values():
            dataset.close()

    scope_rows = [
        {"scope": scope_name, **finish(acc, pixel_area)}
        for scope_name, acc in scopes.items()
    ]
    transition_rows = []
    for scope_name in scopes:
        scope_pixels = scopes[scope_name]["pixels"]
        for code24 in range(5):
            for code25 in range(5):
                row = finish(transition_acc[(scope_name, code24, code25)], pixel_area)
                transition_rows.append({
                    "scope": scope_name,
                    "source_2024_code": code24,
                    "source_2024_name": SOURCE_NAMES[code24],
                    "source_2025_code": code25,
                    "source_2025_name": SOURCE_NAMES[code25],
                    **row,
                    "fraction_of_scope": row["pixels"] / scope_pixels if scope_pixels else None,
                })

    source_rows = []
    for scope_name in scopes:
        scope_pixels = scopes[scope_name]["pixels"]
        for date in ("2024", "2025"):
            for code in range(5):
                count = source_totals[(f"{scope_name}|{date}", code)]
                source_rows.append({
                    "scope": scope_name,
                    "date": date,
                    "source_code": code,
                    "source_name": SOURCE_NAMES[code],
                    "pixels": count,
                    "area_km2": count * pixel_area / 1e6,
                    "fraction_of_scope": count / scope_pixels if scope_pixels else None,
                })

    new_scope_pixels = scopes["water_2025_only"]["pixels"]
    new_transition = [
        row for row in transition_rows
        if row["scope"] == "water_2025_only" and row["pixels"] > 0
    ]
    new_transition.sort(key=lambda row: row["pixels"], reverse=True)
    largest = new_transition[:10]
    gained_sar_support = sum(
        row["pixels"]
        for row in new_transition
        if row["source_2024_code"] in (2, 4)
        and row["source_2025_code"] in (1, 3)
    )
    remained_dual = sum(
        row["pixels"]
        for row in new_transition
        if row["source_2024_code"] == 3 and row["source_2025_code"] == 3
    )

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("scope_probability_summary.csv", scope_rows),
        ("source_transition_matrix.csv", transition_rows),
        ("source_marginals.csv", source_rows),
    ):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    expected_new_pixels = 9_455_724
    validations = {
        "invalid_source_codes_zero": invalid_source_codes == 0,
        "water_2025_only_matches_object_audit": new_scope_pixels == expected_new_pixels,
        "transition_pixels_close": sum(row["pixels"] for row in new_transition) == new_scope_pixels,
    }
    status = "complete" if all(validations.values()) else "validation_failed"
    summary = {
        "status": status,
        "experiment": "Two-date source-path transition audit",
        "source_code_meaning": SOURCE_NAMES,
        "scopes": {row["scope"]: row for row in scope_rows},
        "water_2025_only": {
            "pixels": new_scope_pixels,
            "area_km2": new_scope_pixels * pixel_area / 1e6,
            "gained_sar_support_pixels": gained_sar_support,
            "gained_sar_support_area_km2": gained_sar_support * pixel_area / 1e6,
            "gained_sar_support_fraction": gained_sar_support / new_scope_pixels,
            "remained_dual_source_pixels": remained_dual,
            "remained_dual_source_area_km2": remained_dual * pixel_area / 1e6,
            "remained_dual_source_fraction": remained_dual / new_scope_pixels,
            "largest_source_transitions": largest,
        },
        "validations": validations,
        "limits": [
            "Source flags identify which evidence paths were available, not which branch caused a fused decision.",
            "A transition to SAR or dual-source support does not itself prove SAR error.",
            "Saved fused probability does not separate SAR and optical branch probabilities.",
            "This fixed-pixel audit does not identify physical pond objects or survey membership.",
            "No model training, input, threshold, mask, DEM volume, or production result is modified.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {status}", flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
