#!/usr/bin/env python3
"""One-factor sensitivity audit for the Shihe temporal-shape candidate.

Reuses exact V75 values already computed for all candidate objects.  The
manual totals are evaluation references only; no threshold is optimized to a
manual target and no GIS/DEM calculation is repeated.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = (
    ROOT / "diagnostics/results/temporal_shape_cross_region_bridge"
    / "20261010_194531_518591/shihe/landuse_rejected_temporal_shape_bridge.csv"
)
VOLUME_RESULT = (
    ROOT / "diagnostics/results/shihe_temporal_shape_dem_volume"
    / "20261010_203830_787211"
)
BASELINE_AUDIT = (
    ROOT / "diagnostics/results/cross_region_baseline_trends"
    / "20261010_202616_329605/summary.json"
)
MANUAL_AUDIT = (
    ROOT / "diagnostics/results/cross_region_manual_totals"
    / "20261010_201437_787088/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/shihe_temporal_shape_sensitivity"
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
    features = {
        int(row["object_id"]): row
        for row in read_csv(BRIDGE)
        if truth(row["selected_loose_candidate"])
    }
    volumes = {int(row["pond_id"]): row for row in read_csv(VOLUME_RESULT / "shihe_candidates_with_volume.csv")}
    if set(features) != set(volumes):
        raise RuntimeError(
            f"Feature-volume ID mismatch: features={len(features)}, volumes={len(volumes)}"
        )

    objects = []
    for object_id, feature in features.items():
        volume_row = volumes[object_id]
        objects.append({
            "object_id": object_id,
            "area_m2": float(feature["area_m2"]),
            "previous_support": float(feature["previous_near_fraction"]),
            "future_proxy": truth(feature["future_fraction_ge_025_retrospective"]),
            "bbox_fill": float(feature["bbox_fill"]),
            "elongation": float(feature["elongation"]),
            "v75_wan_m3": float(volume_row["v75_m3"] or 0.0) / 10000.0,
            "qa": volume_row.get("qa", ""),
        })

    variants = [
        ("full_candidate", lambda row: True, "Qualified cross-region loose candidate"),
        ("previous_ge_040", lambda row: row["previous_support"] >= 0.40, "Raise historical support only"),
        ("previous_ge_050", lambda row: row["previous_support"] >= 0.50, "Raise historical support only"),
        ("previous_ge_075", lambda row: row["previous_support"] >= 0.75, "Raise historical support only"),
        ("min_area_0p3ha", lambda row: row["area_m2"] >= 3000, "Raise minimum area only"),
        ("min_area_0p5ha", lambda row: row["area_m2"] >= 5000, "Raise minimum area only"),
        ("min_area_1ha", lambda row: row["area_m2"] >= 10000, "Raise minimum area only"),
        ("max_area_3ha", lambda row: row["area_m2"] < 30000, "Lower maximum area only"),
        ("max_area_2ha", lambda row: row["area_m2"] < 20000, "Lower maximum area only"),
        ("bbox_fill_ge_050", lambda row: row["bbox_fill"] >= 0.50, "Raise bbox fill only"),
        ("bbox_fill_ge_055", lambda row: row["bbox_fill"] >= 0.55, "Raise bbox fill only"),
        ("elongation_le_250", lambda row: row["elongation"] <= 2.50, "Lower elongation maximum only"),
        ("elongation_le_200", lambda row: row["elongation"] <= 2.00, "Lower elongation maximum only"),
    ]

    baseline = json.loads(BASELINE_AUDIT.read_text(encoding="utf-8"))["regional"]["shihe"]["model"]
    manual = json.loads(MANUAL_AUDIT.read_text(encoding="utf-8"))["regional_trends"]["shihe"]
    model_24 = float(baseline[DATE_24]["v75_wan_m3"])
    model_25 = float(baseline[DATE_25]["v75_wan_m3"])
    manual_scenarios = {
        "source_total": (float(manual["source_total"][DATE_24]), float(manual["source_total"][DATE_25])),
        "component_sum": (float(manual["component_sum"][DATE_24]), float(manual["component_sum"][DATE_25])),
    }
    baseline_eval = {
        name: evaluate(model_24, model_25, *values)
        for name, values in manual_scenarios.items()
    }

    results = []
    for name, predicate, reason in variants:
        selected = [row for row in objects if predicate(row)]
        increment = sum(row["v75_wan_m3"] for row in selected)
        future = [row for row in selected if row["future_proxy"]]
        record = {
            "variant": name,
            "reason": reason,
            "objects": len(selected),
            "area_km2": sum(row["area_m2"] for row in selected) / 1e6,
            "v75_increment_wan_m3": increment,
            "retained_volume_fraction": increment / sum(row["v75_wan_m3"] for row in objects),
            "future_proxy_object_fraction": len(future) / len(selected) if selected else 0.0,
            "future_proxy_volume_fraction": (
                sum(row["v75_wan_m3"] for row in future) / increment if increment else 0.0
            ),
            "missing_v75": sum(row["qa"] != "OK" for row in selected),
            "evaluation": {},
        }
        for scenario, manual_values in manual_scenarios.items():
            after = evaluate(model_24 + increment, model_25, *manual_values)
            before = baseline_eval[scenario]
            after["trend_log_error_reduction"] = before["trend_log_error"] - after["trend_log_error"]
            after["total_absolute_error_reduction_wan_m3"] = (
                before["total_absolute_error_wan_m3"] - after["total_absolute_error_wan_m3"]
            )
            record["evaluation"][scenario] = after
        results.append(record)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    flat_rows = []
    for result in results:
        flat = {key: value for key, value in result.items() if key != "evaluation"}
        for scenario, values in result["evaluation"].items():
            for key, value in values.items():
                flat[f"{scenario}_{key}"] = value
        flat_rows.append(flat)
    table = output / "shihe_one_factor_sensitivity.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    report = {
        "status": "complete",
        "experiment": "Shihe temporal-shape one-factor sensitivity",
        "baseline_evaluation": baseline_eval,
        "manual_scenarios": manual_scenarios,
        "variants": results,
        "output_table": str(table),
        "interpretation": [
            "Each variant changes one existing mechanism boundary while all others remain fixed.",
            "Future persistence is reported only as a retrospective risk proxy and never selects objects.",
            "Manual totals evaluate outcomes but do not determine thresholds or object membership.",
        ],
        "limits": [
            "One-factor sensitivity does not estimate interactions among thresholds.",
            "Manual totals are regional aggregates and contain a 2024 total-versus-components discrepancy.",
            "No new GIS review or DEM calculation is performed in this step.",
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
