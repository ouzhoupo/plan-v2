#!/usr/bin/env python3
"""Check selective new-SAR gates on the 16 fixed reviewed objects."""

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
CURRENT = (
    ROOT / "diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712"
    / DATE / "c4_water_mask.tif"
)
SHARED = (
    ROOT / "diagnostics/results/shared_sar_calibration_upstream/20261009_193736_367365"
    / DATE / "shared_sar_shared_alpha_c4_water_mask.tif"
)
EXPERIMENT = (
    ROOT / "diagnostics/results/new_sar_evidence_gate_upstream/20261010_134344_780095"
)
REVIEW = (
    ROOT / "diagnostics/results/two_date_upstream_object_driver_review/20261009_172614_247553"
    / "review_manifest.csv"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/new_sar_gate_reviewed_objects"
VARIANTS = (
    "clear_low_no_aux_gate",
    "unclear_or_low_no_aux_gate",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def group_name(label: str) -> str:
    if label.startswith("stable_true_water"):
        return "stable_true_water"
    if label.startswith("nonwater") or label.startswith("cloud_contaminated"):
        return "reviewed_nonwater"
    if label == "true_large_water_non_target":
        return "true_water_non_target"
    return "mixed_or_uncertain"


def main() -> int:
    rows = read_rows(REVIEW)
    if len(rows) != 16 or any(row["proxy_label"] == "unreviewed" for row in rows):
        raise RuntimeError("Expected 16 completed fixed review labels")
    with rasterio.open(CURRENT) as source:
        current = source.read(1) > 0
        grid = (source.shape, source.crs, source.transform)
    labels, count = ndimage.label(
        current,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    slices = ndimage.find_objects(labels)
    output_rows = []
    with ExitStack() as stack:
        shared = stack.enter_context(rasterio.open(SHARED))
        variants = {
            name: stack.enter_context(
                rasterio.open(EXPERIMENT / f"{name}_c4_water_mask.tif")
            )
            for name in VARIANTS
        }
        for source in (shared, *variants.values()):
            if (
                source.shape != grid[0]
                or source.crs != grid[1]
                or not source.transform.almost_equals(grid[2])
            ):
                raise RuntimeError(f"Grid mismatch: {source.name}")
        for row in rows:
            object_id = int(row["object_id"])
            component_slice = slices[object_id - 1]
            object_mask = labels[component_slice] == object_id
            pixels = int(np.count_nonzero(object_mask))
            if pixels != int(row["pixels"]):
                raise RuntimeError(f"Object {object_id} pixel count mismatch")
            window = Window.from_slices(*component_slice)
            shared_mask = shared.read(1, window=window) > 0
            shared_retained = int(np.count_nonzero(object_mask & shared_mask))
            item = {
                "object_id": object_id,
                "proxy_label": row["proxy_label"],
                "evidence_group": group_name(row["proxy_label"]),
                "pixels": pixels,
                "shared_retained_fraction": shared_retained / pixels,
                "notes": row.get("notes", ""),
            }
            for name, source in variants.items():
                mask = source.read(1, window=window) > 0
                retained = int(np.count_nonzero(object_mask & mask))
                item[f"{name}_retained_fraction"] = retained / pixels
                item[f"{name}_incremental_removed_fraction"] = (
                    shared_retained - retained
                ) / pixels
            output_rows.append(item)

    grouped = defaultdict(lambda: defaultdict(float))
    for row in output_rows:
        group = row["evidence_group"]
        pixels = int(row["pixels"])
        grouped[group]["objects"] += 1
        grouped[group]["pixels"] += pixels
        grouped[group]["shared_weighted"] += row["shared_retained_fraction"] * pixels
        for name in VARIANTS:
            grouped[group][f"{name}_retained_weighted"] += (
                row[f"{name}_retained_fraction"] * pixels
            )
            grouped[group][f"{name}_removed_weighted"] += (
                row[f"{name}_incremental_removed_fraction"] * pixels
            )
    summary_groups = {}
    for group, values in grouped.items():
        pixels = int(values["pixels"])
        entry = {
            "objects": int(values["objects"]),
            "pixels": pixels,
            "shared_retained_fraction": values["shared_weighted"] / pixels,
        }
        for name in VARIANTS:
            entry[f"{name}_retained_fraction"] = (
                values[f"{name}_retained_weighted"] / pixels
            )
            entry[f"{name}_incremental_removed_fraction"] = (
                values[f"{name}_removed_weighted"] / pixels
            )
        summary_groups[group] = entry

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "reviewed_object_retention.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "status": "complete",
        "experiment": "Selective new-SAR gate retention on fixed reviewed objects",
        "current_component_count": int(count),
        "groups": summary_groups,
        "output_table": str(table),
        "limits": [
            "These are 16 reused mechanism samples, not an independent or random accuracy sample.",
            "Proxy labels are GIS interpretations rather than field truth.",
            "Results assess upstream mask side effects only and do not establish volume accuracy.",
            "No model output or production code is modified.",
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
