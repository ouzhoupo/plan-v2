#!/usr/bin/env python3
"""Audit local threshold sensitivity of the operational temporal-shape candidate.

All tested variants are subsets of the already evaluated loose candidate, so this
script reuses exact V75 values and performs no GIS or DEM recomputation. Future
persistence is reported only as a retrospective risk proxy and never selects an
object.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RESULT = (
    ROOT / "diagnostics/results/operational_temporal_shape_candidate/20261010_184856_285062"
)
VOLUME_RESULT = (
    ROOT / "diagnostics/results/operational_temporal_shape_dem_volume/20261010_192503_120178"
)
RETROSPECTIVE_FEATURES = (
    ROOT / "diagnostics/results/retrospective_rescue_volume_concentration"
    / "20261010_183252_458888/retrospective_objects_volume_features.csv"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/operational_temporal_shape_sensitivity"

BASE_2024_WAN_M3 = 2228.14375
MODEL_2025_WAN_M3 = 3430.1875
MANUAL_2024_WAN_M3 = 17058.0
MANUAL_2025_WAN_M3 = 9174.0


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def number(value: str | None, default: float = 0.0) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def evaluate(increment_v75_m3: float) -> dict:
    model_24 = BASE_2024_WAN_M3 + increment_v75_m3 / 10000.0
    ratio = MODEL_2025_WAN_M3 / model_24
    manual_ratio = MANUAL_2025_WAN_M3 / MANUAL_2024_WAN_M3
    err_24 = abs(model_24 - MANUAL_2024_WAN_M3)
    err_25 = abs(MODEL_2025_WAN_M3 - MANUAL_2025_WAN_M3)
    return {
        "model_2024_wan_m3": model_24,
        "model_2025_wan_m3": MODEL_2025_WAN_M3,
        "model_2025_to_2024_ratio": ratio,
        "manual_2025_to_2024_ratio": manual_ratio,
        "direction_matches_manual": ratio < 1.0,
        "absolute_log_ratio_error": abs(math.log(ratio / manual_ratio)),
        "total_absolute_error_wan_m3": err_24 + err_25,
        "mape_percent": 50.0 * (
            err_24 / MANUAL_2024_WAN_M3 + err_25 / MANUAL_2025_WAN_M3
        ),
    }


def main() -> int:
    source_summary = json.loads((VOLUME_RESULT / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise RuntimeError("Exact V75 source experiment is incomplete")

    candidates = [
        row for row in read_csv(CANDIDATE_RESULT / "previous_strong_shape_candidates.csv")
        if truth(row["selected_compact_02_5ha"])
    ]
    selected_ids = {int(row["object_id"]) for row in candidates}

    persistent_volumes: dict[int, tuple[float, str]] = {}
    for row in read_csv(RETROSPECTIVE_FEATURES):
        object_id = int(row["object_id"])
        if object_id in selected_ids:
            persistent_volumes[object_id] = (number(row.get("v75_m3")), row.get("qa", ""))

    new_volumes: dict[int, tuple[float, str]] = {}
    new_table = VOLUME_RESULT / "selected_without_future_support_with_volume.csv"
    for row in read_csv(new_table):
        object_id = int(row["pond_id"])
        new_volumes[object_id] = (number(row.get("v75_m3")), row.get("qa", ""))

    overlap = set(persistent_volumes) & set(new_volumes)
    all_volumes = {**persistent_volumes, **new_volumes}
    if overlap or set(all_volumes) != selected_ids:
        raise RuntimeError("Exact V75 object coverage does not close")
    if abs(sum(value[0] for value in all_volumes.values()) - 29386875.0) > 0.001:
        raise RuntimeError("Loose-candidate V75 total changed")

    scenarios = [
        ("loose_reference", "reference", 0.25, 0.2, 5.0, 0.45, 3.0),
        ("previous_040", "previous support", 0.40, 0.2, 5.0, 0.45, 3.0),
        ("previous_050", "previous support", 0.50, 0.2, 5.0, 0.45, 3.0),
        ("previous_075", "previous support", 0.75, 0.2, 5.0, 0.45, 3.0),
        ("minimum_area_03ha", "minimum area", 0.25, 0.3, 5.0, 0.45, 3.0),
        ("minimum_area_05ha", "minimum area", 0.25, 0.5, 5.0, 0.45, 3.0),
        ("minimum_area_10ha", "minimum area", 0.25, 1.0, 5.0, 0.45, 3.0),
        ("maximum_area_4ha", "maximum area", 0.25, 0.2, 4.0, 0.45, 3.0),
        ("maximum_area_3ha", "maximum area", 0.25, 0.2, 3.0, 0.45, 3.0),
        ("bbox_fill_050", "bbox fill", 0.25, 0.2, 5.0, 0.50, 3.0),
        ("bbox_fill_055", "bbox fill", 0.25, 0.2, 5.0, 0.55, 3.0),
        ("elongation_25", "elongation", 0.25, 0.2, 5.0, 0.45, 2.5),
        ("elongation_20", "elongation", 0.25, 0.2, 5.0, 0.45, 2.0),
        ("strict_existing", "combined strict", 0.25, 0.2, 5.0, 0.55, 2.5),
    ]

    current = evaluate(0.0)
    required_model_24 = MODEL_2025_WAN_M3 / (
        MANUAL_2025_WAN_M3 / MANUAL_2024_WAN_M3
    )
    required_increment_wan_m3 = required_model_24 - BASE_2024_WAN_M3
    records = []
    for name, changed, prev_min, area_min_ha, area_max_ha, fill_min, elong_max in scenarios:
        chosen = []
        for row in candidates:
            area_ha = float(row["area_m2"]) / 10000.0
            if (
                float(row["previous_near_fraction"]) >= prev_min
                and area_min_ha <= area_ha < area_max_ha
                and float(row["bbox_fill"]) >= fill_min
                and float(row["elongation"]) <= elong_max
            ):
                chosen.append(row)
        ids = {int(row["object_id"]) for row in chosen}
        total_v75 = sum(all_volumes[object_id][0] for object_id in ids)
        persistent_ids = {
            int(row["object_id"]) for row in chosen
            if truth(row["retrospective_future_persistent"])
        }
        persistent_v75 = sum(all_volumes[object_id][0] for object_id in persistent_ids)
        result = evaluate(total_v75)
        record = {
            "scenario": name,
            "changed_dimension": changed,
            "previous_support_min": prev_min,
            "area_min_ha": area_min_ha,
            "area_max_ha_exclusive": area_max_ha,
            "bbox_fill_min": fill_min,
            "elongation_max": elong_max,
            "objects": len(ids),
            "area_m2": sum(float(row["area_m2"]) for row in chosen),
            "missing_v75": sum(1 for object_id in ids if all_volumes[object_id][1] != "OK"),
            "increment_v75_m3": total_v75,
            "increment_v75_wan_m3": total_v75 / 10000.0,
            "fraction_of_exact_trend_increment": (
                total_v75 / 10000.0 / required_increment_wan_m3
            ),
            "future_persistent_objects": len(persistent_ids),
            "future_persistent_object_fraction": len(persistent_ids) / len(ids) if ids else 0.0,
            "future_persistent_v75_fraction": persistent_v75 / total_v75 if total_v75 else 0.0,
            **result,
            "trend_log_error_reduction_vs_current": (
                current["absolute_log_ratio_error"] - result["absolute_log_ratio_error"]
            ),
            "total_absolute_error_reduction_vs_current_wan_m3": (
                current["total_absolute_error_wan_m3"]
                - result["total_absolute_error_wan_m3"]
            ),
        }
        records.append(record)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "threshold_sensitivity.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    report = {
        "status": "complete",
        "experiment": "Operational temporal-shape local threshold sensitivity",
        "region": "pihe",
        "source_candidate_objects": len(selected_ids),
        "required_increment_for_exact_manual_ratio_wan_m3": required_increment_wan_m3,
        "current_without_rescue": current,
        "scenarios": records,
        "output_table": str(table),
        "interpretation": [
            "Every scenario is a subset of the already evaluated loose candidate.",
            "Only one rule dimension is tightened at a time, except the existing strict comparison.",
            "Future persistence is a retrospective risk proxy and never affects selection.",
            "A stable neighborhood is more useful than a single numerically optimal threshold.",
        ],
        "limits": [
            "This audit cannot test looser thresholds outside the 0.2-5 ha, fill >=0.45, elongation <=3 scope.",
            "No threshold is selected solely by matching the two-date manual ratio.",
            "Only Pihe and two manual dates are evaluated.",
            "Manual survey spatial-scope equivalence remains unresolved.",
            "No GIS mask, production parameter, or formal result is changed.",
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
