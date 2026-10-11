#!/usr/bin/env python3
"""Audit whether adding 2024 rescue capacity can improve regional trends.

This is a read-only arithmetic gate.  It compares the original V75 outputs
with the latest cross-region manual-total audit.  No raster, object selection,
DEM volume, production output, or source workbook is changed.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "diagnostics/results"
MANUAL_ROOT = RESULTS / "cross_region_manual_totals"
OUTPUT_ROOT = RESULTS / "cross_region_baseline_trends"
DATES = ("2024-06-01", "2025-05-21")
REGIONS = ("pihe", "shihe", "hangbuhe")


def latest_complete_summary(folder: Path) -> Path:
    candidates = []
    for path in folder.glob("*/summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") == "complete":
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No complete summary found under {folder}")
    return max(candidates, key=lambda path: path.parent.name)


def storage_csv(date: str, region: str) -> Path:
    compact = date.replace("-", "")
    return ROOT / "output" / compact / "final" / region / f"pond_storage_{date}.csv"


def sum_v75_wan_m3(path: Path) -> tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    count = 0
    total_m3 = 0.0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            value = row.get("v75_m3")
            if value not in (None, ""):
                total_m3 += float(value)
    return count, total_m3 / 10000.0


def metrics(model_2024: float, model_2025: float, manual_2024: float, manual_2025: float) -> dict:
    if min(model_2024, model_2025, manual_2024, manual_2025) <= 0:
        raise RuntimeError("Trend audit requires positive model and manual volumes")
    model_ratio = model_2025 / model_2024
    manual_ratio = manual_2025 / manual_2024
    target_2024 = model_2025 / manual_ratio
    target_increment = target_2024 - model_2024
    return {
        "model_ratio_2025_to_2024": model_ratio,
        "manual_ratio_2025_to_2024": manual_ratio,
        "model_direction": "decrease" if model_ratio < 1 else "increase" if model_ratio > 1 else "flat",
        "manual_direction": "decrease" if manual_ratio < 1 else "increase" if manual_ratio > 1 else "flat",
        "trend_log_error": abs(math.log(model_ratio / manual_ratio)),
        "model_ratio_above_manual": model_ratio > manual_ratio,
        "adding_2024_volume_initially_improves_trend": model_ratio > manual_ratio,
        "exact_match_2024_model_volume_wan_m3": target_2024,
        "exact_match_increment_wan_m3": target_increment,
        "direction_flip_increment_wan_m3": max(model_2025 - model_2024, 0.0),
    }


def main() -> int:
    manual_path = latest_complete_summary(MANUAL_ROOT)
    manual = json.loads(manual_path.read_text(encoding="utf-8"))

    model = {}
    for region in REGIONS:
        model[region] = {}
        for date in DATES:
            path = storage_csv(date, region)
            count, total = sum_v75_wan_m3(path)
            model[region][date] = {
                "objects": count,
                "v75_wan_m3": total,
                "source": str(path),
            }

    regional = {}
    rows = []
    for region in REGIONS:
        m24 = model[region][DATES[0]]["v75_wan_m3"]
        m25 = model[region][DATES[1]]["v75_wan_m3"]
        source = manual["regional_trends"][region]["source_total"]
        component = manual["regional_trends"][region]["component_sum"]
        source_metrics = metrics(m24, m25, source[DATES[0]], source[DATES[1]])
        component_metrics = metrics(m24, m25, component[DATES[0]], component[DATES[1]])
        regional[region] = {
            "model": model[region],
            "source_total_scenario": source_metrics,
            "component_sum_scenario": component_metrics,
            "manual_direction_robust": manual["regional_trends"][region][
                "direction_robust_to_total_discrepancy"
            ],
            "rescue_direction_robust": (
                source_metrics["adding_2024_volume_initially_improves_trend"]
                and component_metrics["adding_2024_volume_initially_improves_trend"]
            ),
        }
        for scenario, values in (
            ("source_total", source_metrics),
            ("component_sum", component_metrics),
        ):
            rows.append({
                "region": region,
                "scenario": scenario,
                "model_2024_v75_wan_m3": m24,
                "model_2025_v75_wan_m3": m25,
                **values,
            })

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "regional_baseline_trends.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "complete",
        "experiment": "Cross-region original-V75 baseline trend qualification",
        "manual_summary": str(manual_path),
        "regional": regional,
        "checks": {
            "all_manual_directions_robust": all(
                item["manual_direction_robust"] for item in regional.values()
            ),
            "adding_2024_volume_initially_improves_all_regions": all(
                item["rescue_direction_robust"] for item in regional.values()
            ),
        },
        "interpretation": [
            "The rescue changes only the 2024 denominator in this arithmetic gate.",
            "An increment is directionally useful when the baseline model ratio exceeds the manual ratio.",
            "Exact-match increments are diagnostic budgets, not selection targets or fitted parameters.",
            "For a manually increasing region, the direction-flip increment marks where excessive rescue would reverse the model trend.",
        ],
        "limits": [
            "Original V75 outputs are used for cross-region comparability and differ from the qualified Pihe shared-SAR pair.",
            "Manual irrigation-region totals are not spatially matched object labels.",
            "No candidate DEM volume is computed in this step.",
        ],
        "output_table": str(table),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
