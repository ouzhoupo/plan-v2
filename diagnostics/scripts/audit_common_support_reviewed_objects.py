#!/usr/bin/env python3
"""Measure common-support effects on the 16 previously reviewed objects.

This reuses fixed proxy labels from the two-date upstream object review.  It
does not create new truth labels, tune thresholds, or modify model outputs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DATE = "2025-05-21"
FROZEN = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712"
    / DATE
    / "c4_water_mask.tif"
)
REVIEW = (
    ROOT
    / "diagnostics/results/two_date_upstream_object_driver_review"
    / "20261009_172614_247553"
    / "review_manifest.csv"
)
SHARED_SAR = (
    ROOT
    / "diagnostics/results/shared_sar_calibration_upstream"
    / "20261009_193736_367365"
    / DATE
)
COMMON = (
    ROOT
    / "diagnostics/results/common_sar_support_upstream"
    / "20261009_232419_505438"
    / DATE
)
OUTPUT_ROOT = ROOT / "diagnostics/results/common_support_reviewed_objects"
VARIANTS = (
    "shared_sar_current_alpha",
    "shared_sar_shared_alpha",
    "shared_sar_alpha_035",
    "shared_sar_base_only_bound",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_name(row: dict) -> str:
    label = row["proxy_label"]
    if label.startswith("stable_true_water"):
        return "stable_true_water"
    if label.startswith("nonwater") or label.startswith("cloud_contaminated"):
        return "reviewed_nonwater"
    if label == "true_large_water_non_target":
        return "true_water_non_target"
    return "mixed_or_uncertain"


def main() -> int:
    rows = read_rows(REVIEW)
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 reviewed objects, found {len(rows)}")
    if any(row["proxy_label"] == "unreviewed" for row in rows):
        raise RuntimeError("Review manifest still contains unreviewed labels")

    with rasterio.open(FROZEN) as source:
        current = source.read(1) > 0
        pixel_area = abs(float(source.transform.a * source.transform.e))
        reference_grid = (source.shape, source.crs, source.transform)
    labels, count = ndimage.label(
        current,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    slices = ndimage.find_objects(labels)

    output_rows = []
    with ExitStack() as stack:
        shared_readers = {}
        common_readers = {}
        for name in VARIANTS:
            for folder, target in ((SHARED_SAR, shared_readers), (COMMON, common_readers)):
                path = folder / f"{name}_c4_water_mask.tif"
                source = stack.enter_context(rasterio.open(path))
                if (
                    source.shape != reference_grid[0]
                    or source.crs != reference_grid[1]
                    or not source.transform.almost_equals(reference_grid[2])
                ):
                    raise RuntimeError(f"Grid mismatch: {path}")
                target[name] = source

        for row in rows:
            object_id = int(row["object_id"])
            if object_id < 1 or object_id > count:
                raise RuntimeError(f"Invalid object id: {object_id}")
            component_slice = slices[object_id - 1]
            object_mask = labels[component_slice] == object_id
            pixels = int(np.count_nonzero(object_mask))
            if pixels != int(row["pixels"]):
                raise RuntimeError(f"Object {object_id} pixel count mismatch")
            window = Window.from_slices(*component_slice)
            result = {
                "object_id": object_id,
                "review_group": row["review_group"],
                "proxy_label": row["proxy_label"],
                "evidence_group": group_name(row),
                "original_pixels": pixels,
                "original_area_m2": pixels * pixel_area,
                "notes": row.get("notes", ""),
            }
            for name in VARIANTS:
                shared = shared_readers[name].read(1, window=window) > 0
                common = common_readers[name].read(1, window=window) > 0
                shared_retained = int(np.count_nonzero(object_mask & shared))
                common_retained = int(np.count_nonzero(object_mask & common))
                result[f"{name}_shared_sar_retained_fraction"] = shared_retained / pixels
                result[f"{name}_common_support_retained_fraction"] = common_retained / pixels
                result[f"{name}_support_incremental_removed_fraction"] = (
                    shared_retained - common_retained
                ) / pixels
            output_rows.append(result)

    grouped = defaultdict(lambda: defaultdict(float))
    for row in output_rows:
        group = row["evidence_group"]
        weight = int(row["original_pixels"])
        grouped[group]["objects"] += 1
        grouped[group]["pixels"] += weight
        for name in VARIANTS:
            for suffix in (
                "shared_sar_retained_fraction",
                "common_support_retained_fraction",
                "support_incremental_removed_fraction",
            ):
                grouped[group][f"{name}_{suffix}_weighted_sum"] += (
                    float(row[f"{name}_{suffix}"]) * weight
                )

    group_summary = {}
    for group, values in grouped.items():
        pixels = int(values["pixels"])
        entry = {"objects": int(values["objects"]), "pixels": pixels}
        for name in VARIANTS:
            for suffix in (
                "shared_sar_retained_fraction",
                "common_support_retained_fraction",
                "support_incremental_removed_fraction",
            ):
                entry[f"{name}_{suffix}"] = (
                    values[f"{name}_{suffix}_weighted_sum"] / pixels
                )
        group_summary[group] = entry

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "reviewed_object_retention.csv", output_rows)
    report = {
        "status": "complete",
        "experiment": "Common-support retention on 16 fixed reviewed objects",
        "current_component_count": int(count),
        "reviewed_objects": len(output_rows),
        "groups": group_summary,
        "output_table": str(output / "reviewed_object_retention.csv"),
        "limits": [
            "The 16 objects are fixed mechanism samples, not a random accuracy sample.",
            "Proxy labels come from prior GIS interpretation and are not field truth.",
            "Stable linear river/canal is a true-water control but not a target pond object.",
            "Retention describes upstream masks only and does not estimate final volume accuracy.",
            "No threshold, mask, object, production code or production output is modified.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print("STATUS: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
