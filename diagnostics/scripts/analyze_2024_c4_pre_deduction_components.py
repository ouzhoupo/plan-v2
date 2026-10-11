#!/usr/bin/env python3
"""Measure complete C4 water components before 2024-06 vector deduction."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DATE = "2024-06-01"
SPLIT_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_morphology_split"
    / "20260926_203928_115516"
)
GROUP_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_groups"
    / "20260926_204740_877320"
)
CODE_LABELS = {
    0: "not_deducted",
    1: "reservoir_only",
    2: "lake_only",
    3: "reservoir_and_lake",
    4: "linear_only",
    5: "reservoir_and_linear",
    6: "lake_and_linear",
    7: "reservoir_lake_and_linear",
}


def area_band(area_m2):
    if area_m2 < 500:
        return "below_area_gate"
    if area_m2 <= 20_000:
        return "pond_scale_500_20000"
    if area_m2 <= 500_000:
        return "target_range_20000_500000"
    return "above_max_area_gate"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    split = json.loads((SPLIT_RESULT / "summary.json").read_text(encoding="utf-8"))
    groups = json.loads((GROUP_RESULT / "summary.json").read_text(encoding="utf-8"))
    if split.get("status") != "complete" or groups.get("status") != "complete":
        raise RuntimeError("Required second-round result is incomplete")

    pre_path = SPLIT_RESULT / "c4_pre_deduction_no_morph.tif"
    membership_path = GROUP_RESULT / "deduction_group_membership.tif"
    with rasterio.open(pre_path) as pre_source, rasterio.open(membership_path) as member_source:
        if (
            pre_source.width != member_source.width
            or pre_source.height != member_source.height
            or pre_source.crs != member_source.crs
            or not pre_source.transform.almost_equals(member_source.transform)
        ):
            raise RuntimeError("Pre-deduction and membership grids do not match")
        pre = pre_source.read(1) > 0
        membership = member_source.read(1).astype("uint8")
        transform = pre_source.transform
        pixel_area = abs(transform.a * transform.e - transform.b * transform.d)

    if np.any((membership > 0) & ~pre):
        raise RuntimeError("Deduction membership exists outside pre-deduction water")

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8")
    labels, component_count = ndimage.label(pre, structure=structure)
    counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    removed_counts = np.bincount(
        labels[membership > 0].ravel(), minlength=component_count + 1
    )
    code_counts = {
        code: np.bincount(labels[membership == code].ravel(), minlength=component_count + 1)
        for code in range(1, 8)
    }
    slices = ndimage.find_objects(labels)

    rows = []
    for component_id in range(1, component_count + 1):
        pixels = int(counts[component_id])
        if pixels == 0:
            continue
        removed = int(removed_counts[component_id])
        bbox = slices[component_id - 1]
        if bbox is None:
            continue
        height = int(bbox[0].stop - bbox[0].start)
        width = int(bbox[1].stop - bbox[1].start)
        component_codes = {code: int(values[component_id]) for code, values in code_counts.items()}
        dominant_code = max(component_codes, key=component_codes.get) if removed else 0
        area_m2 = pixels * pixel_area
        rows.append(
            {
                "component_id": component_id,
                "pixels": pixels,
                "area_m2": area_m2,
                "area_band": area_band(area_m2),
                "removed_pixels": removed,
                "removed_area_m2": removed * pixel_area,
                "removed_fraction": removed / pixels,
                "surviving_pixels": pixels - removed,
                "fully_removed": removed == pixels,
                "partially_removed": 0 < removed < pixels,
                "dominant_deduction_code": dominant_code,
                "dominant_deduction_group": CODE_LABELS[dominant_code],
                "reservoir_involved_pixels": sum(
                    component_codes[code] for code in (1, 3, 5, 7)
                ),
                "lake_involved_pixels": sum(component_codes[code] for code in (2, 3, 6, 7)),
                "linear_involved_pixels": sum(component_codes[code] for code in (4, 5, 6, 7)),
                "bbox_width_px": width,
                "bbox_height_px": height,
                "bbox_fill_fraction": pixels / max(width * height, 1),
                "bbox_elongation": max(width, height) / max(min(width, height), 1),
                "centroid_x": transform.c + (bbox[1].start + width / 2) * transform.a,
                "centroid_y": transform.f + (bbox[0].start + height / 2) * transform.e,
            }
        )

    total_pre = sum(row["pixels"] for row in rows)
    total_removed = sum(row["removed_pixels"] for row in rows)
    validations = {
        "component_pixels_match_pre_deduction": total_pre == int(np.count_nonzero(pre)),
        "component_removed_pixels_match_group_union": total_removed
        == int(np.count_nonzero(membership > 0)),
    }
    status = "complete" if all(validations.values()) else "validation_failed"

    candidate_rows = [
        row
        for row in rows
        if row["fully_removed"]
        and 500 <= row["area_m2"] <= 500_000
        and row["reservoir_involved_pixels"] > 0
    ]
    summary_groups = defaultdict(lambda: {"objects": 0, "area_m2": 0.0, "removed_area_m2": 0.0})
    for row in rows:
        state = "fully_removed" if row["fully_removed"] else (
            "partially_removed" if row["partially_removed"] else "not_removed"
        )
        key = f"{row['area_band']}|{state}|{row['dominant_deduction_group']}"
        summary_groups[key]["objects"] += 1
        summary_groups[key]["area_m2"] += row["area_m2"]
        summary_groups[key]["removed_area_m2"] += row["removed_area_m2"]

    output = (
        ROOT
        / "diagnostics/results/second_round_pre_deduction_components"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "all_pre_deduction_components.csv", rows)
    write_csv(output / "fully_removed_reservoir_target_range.csv", candidate_rows)
    grouped_rows = [
        {
            "group_key": key,
            "objects": value["objects"],
            "area_km2": value["area_m2"] / 1_000_000.0,
            "removed_area_km2": value["removed_area_m2"] / 1_000_000.0,
        }
        for key, value in sorted(summary_groups.items())
    ]
    write_csv(output / "component_group_summary.csv", grouped_rows)

    report = {
        "status": status,
        "experiment": "Second-round 2024-06 complete pre-deduction component audit",
        "date": DATE,
        "connectivity": 4,
        "pixel_area_m2": pixel_area,
        "validations": validations,
        "all_components": {
            "objects": len(rows),
            "area_km2": total_pre * pixel_area / 1_000_000.0,
        },
        "components_touched_by_deduction": {
            "objects": sum(row["removed_pixels"] > 0 for row in rows),
            "fully_removed_objects": sum(row["fully_removed"] for row in rows),
            "partially_removed_objects": sum(row["partially_removed"] for row in rows),
            "removed_area_km2": total_removed * pixel_area / 1_000_000.0,
        },
        "fully_removed_reservoir_target_range": {
            "objects": len(candidate_rows),
            "original_component_area_km2": sum(row["area_m2"] for row in candidate_rows)
            / 1_000_000.0,
        },
        "outputs": {
            "all_components_csv": str(output / "all_pre_deduction_components.csv"),
            "candidate_scope_csv": str(output / "fully_removed_reservoir_target_range.csv"),
            "group_summary_csv": str(output / "component_group_summary.csv"),
        },
        "limits": [
            "These are complete connected C4 water components before deduction, not verified ponds.",
            "Area-range membership does not establish target class or justify rescue.",
            "Bounding-box shape measures are screening descriptors, not final object geometry metrics.",
            "No mask is changed and no DEM volume is calculated.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {status}", flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
