#!/usr/bin/env python3
"""Audit a second pre-target support date for the Shihe rescue candidate.

All current candidates already have >=25% support on 2024-05-21.  This audit
adds the earlier 2024-05-01 model mask as diagnostic evidence and tests whether
two-date historical persistence separates transient wet agriculture from
stable water.  No production mask or candidate object is changed.
"""

from __future__ import annotations

import csv
import json
import math
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = (
    ROOT / "diagnostics/results/temporal_shape_cross_region_bridge"
    / "20261010_194531_518591/shihe"
)
FEATURES = BRIDGE_DIR / "landuse_rejected_temporal_shape_bridge.csv"
AREA_GATE = BRIDGE_DIR / "original_v75_bridge_after_area_gate.tif"
EARLIER = ROOT / "output/20240501/shihe/water_mask_2024-05-01.tif"
VOLUMES = (
    ROOT / "diagnostics/results/shihe_temporal_shape_dem_volume"
    / "20261010_203830_787211/shihe_candidates_with_volume.csv"
)
BASELINE = (
    ROOT / "diagnostics/results/cross_region_baseline_trends"
    / "20261010_202616_329605/summary.json"
)
MANUAL = (
    ROOT / "diagnostics/results/cross_region_manual_totals"
    / "20261010_201437_787088/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/shihe_two_previous_support"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def evaluate(model_24: float, model_25: float, manual_24: float, manual_25: float) -> dict:
    ratio = model_25 / model_24
    manual_ratio = manual_25 / manual_24
    errors = (abs(model_24 - manual_24), abs(model_25 - manual_25))
    return {
        "model_ratio": ratio,
        "manual_ratio": manual_ratio,
        "direction_matches": (ratio < 1) == (manual_ratio < 1),
        "trend_log_error": abs(math.log(ratio / manual_ratio)),
        "total_absolute_error_wan_m3": sum(errors),
        "mape_percent": 50.0 * (errors[0] / manual_24 + errors[1] / manual_25),
    }


def main() -> int:
    feature_rows = {
        int(row["object_id"]): row
        for row in read_csv(FEATURES)
        if truth(row["selected_loose_candidate"])
    }
    volume_rows = {int(row["pond_id"]): row for row in read_csv(VOLUMES)}
    if set(feature_rows) != set(volume_rows):
        raise RuntimeError("Feature-volume IDs do not match")

    with rasterio.open(AREA_GATE) as reference:
        area_gate = reference.read(1) > 0
        labels, count = ndimage.label(
            area_gate,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
        )
        with ExitStack() as stack:
            source = stack.enter_context(rasterio.open(EARLIER))
            aligned = stack.enter_context(WarpedVRT(
                source,
                crs=reference.crs,
                transform=reference.transform,
                width=reference.width,
                height=reference.height,
                resampling=Resampling.nearest,
            ))
            earlier = aligned.read(1) > 0
    earlier_near = ndimage.binary_dilation(earlier, iterations=1)
    pixels = np.bincount(labels.ravel(), minlength=count + 1)
    earlier_counts = np.bincount(labels[earlier_near], minlength=count + 1)

    objects = []
    for object_id, feature in feature_rows.items():
        n = int(pixels[object_id])
        if n <= 0:
            raise RuntimeError(f"Missing area-gate label {object_id}")
        objects.append({
            "object_id": object_id,
            "area_m2": float(feature["area_m2"]),
            "may21_support": float(feature["previous_near_fraction"]),
            "may01_support": int(earlier_counts[object_id]) / n,
            "future_proxy": truth(feature["future_fraction_ge_025_retrospective"]),
            "bbox_fill": float(feature["bbox_fill"]),
            "elongation": float(feature["elongation"]),
            "v75_wan_m3": float(volume_rows[object_id]["v75_m3"] or 0.0) / 10000.0,
            "qa": volume_rows[object_id].get("qa", ""),
        })

    variants = [
        ("may21_only_current", lambda row: True),
        ("may01_ge_010", lambda row: row["may01_support"] >= 0.10),
        ("may01_ge_025", lambda row: row["may01_support"] >= 0.25),
        ("may01_ge_050", lambda row: row["may01_support"] >= 0.50),
        ("may01_ge_075", lambda row: row["may01_support"] >= 0.75),
    ]

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["regional"]["shihe"]["model"]
    manual = json.loads(MANUAL.read_text(encoding="utf-8"))["regional_trends"]["shihe"]
    model_24 = float(baseline[DATE_24]["v75_wan_m3"])
    model_25 = float(baseline[DATE_25]["v75_wan_m3"])
    manual_scenarios = {
        "source_total": (float(manual["source_total"][DATE_24]), float(manual["source_total"][DATE_25])),
        "component_sum": (float(manual["component_sum"][DATE_24]), float(manual["component_sum"][DATE_25])),
    }
    before = {
        name: evaluate(model_24, model_25, *values)
        for name, values in manual_scenarios.items()
    }

    results = []
    for name, predicate in variants:
        selected = [row for row in objects if predicate(row)]
        increment = sum(row["v75_wan_m3"] for row in selected)
        future = [row for row in selected if row["future_proxy"]]
        result = {
            "variant": name,
            "objects": len(selected),
            "area_km2": sum(row["area_m2"] for row in selected) / 1e6,
            "v75_increment_wan_m3": increment,
            "retained_volume_fraction": increment / sum(row["v75_wan_m3"] for row in objects),
            "future_proxy_object_fraction": len(future) / len(selected) if selected else 0.0,
            "future_proxy_volume_fraction": (
                sum(row["v75_wan_m3"] for row in future) / increment if increment else 0.0
            ),
            "evaluation": {},
        }
        for scenario, manual_values in manual_scenarios.items():
            after = evaluate(model_24 + increment, model_25, *manual_values)
            after["trend_log_error_reduction"] = before[scenario]["trend_log_error"] - after["trend_log_error"]
            after["total_absolute_error_reduction_wan_m3"] = (
                before[scenario]["total_absolute_error_wan_m3"] - after["total_absolute_error_wan_m3"]
            )
            result["evaluation"][scenario] = after
        results.append(result)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    object_table = output / "shihe_candidates_two_previous_support.csv"
    with object_table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    flat_rows = []
    for result in results:
        flat = {key: value for key, value in result.items() if key != "evaluation"}
        for scenario, values in result["evaluation"].items():
            for key, value in values.items():
                flat[f"{scenario}_{key}"] = value
        flat_rows.append(flat)
    result_table = output / "two_previous_support_variants.csv"
    with result_table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    report = {
        "status": "complete",
        "experiment": "Shihe two-previous-date support audit",
        "current_previous_date": "2024-05-21",
        "added_earlier_date": "2024-05-01",
        "baseline_evaluation": before,
        "variants": results,
        "outputs": {"objects": str(object_table), "variants": str(result_table)},
        "interpretation": [
            "Every tested object already satisfies the current 2024-05-21 support rule.",
            "The earlier date tests persistence using information available before the target date.",
            "Manual totals evaluate outcomes but never select objects or set the thresholds.",
        ],
        "limits": [
            "Two model dates can repeat stable false positives and are not field truth.",
            "A missing earlier detection can be caused by water-level or observation changes.",
            "This audit does not modify the model or recompute DEM volume.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
