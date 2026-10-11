#!/usr/bin/env python3
"""Compare Pihe upstream and temporal-shape contributions on two baselines."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = (
    ROOT / "diagnostics/results/pihe_temporal_shape_dem_volume"
    / "20261011_175105_696336/summary.json"
)
SHARED = (
    ROOT / "diagnostics/results/operational_temporal_shape_dem_volume"
    / "20261010_192503_120178/summary.json"
)
MANUAL = (
    ROOT / "diagnostics/results/cross_region_manual_totals"
    / "20261010_201437_787088/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/pihe_module_contributions"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"


def evaluate(model_24: float, model_25: float, manual_24: float, manual_25: float) -> dict:
    ratio = model_25 / model_24
    manual_ratio = manual_25 / manual_24
    error_24 = abs(model_24 - manual_24)
    error_25 = abs(model_25 - manual_25)
    return {
        "model_2024_wan_m3": model_24,
        "model_2025_wan_m3": model_25,
        "model_ratio": ratio,
        "manual_ratio": manual_ratio,
        "direction_matches": (ratio < 1) == (manual_ratio < 1),
        "trend_log_error": abs(math.log(ratio / manual_ratio)),
        "absolute_error_2024_wan_m3": error_24,
        "absolute_error_2025_wan_m3": error_25,
        "total_absolute_error_wan_m3": error_24 + error_25,
        "mape_percent": 50.0 * (error_24 / manual_24 + error_25 / manual_25),
    }


def delta(before: dict, after: dict) -> dict:
    trend_reduction = before["trend_log_error"] - after["trend_log_error"]
    absolute_reduction = before["total_absolute_error_wan_m3"] - after["total_absolute_error_wan_m3"]
    return {
        "trend_log_error_reduction": trend_reduction,
        "trend_error_reduction_fraction_of_before": trend_reduction / before["trend_log_error"],
        "total_absolute_error_reduction_wan_m3": absolute_reduction,
        "absolute_error_reduction_fraction_of_before": absolute_reduction / before["total_absolute_error_wan_m3"],
        "direction_before": before["direction_matches"],
        "direction_after": after["direction_matches"],
    }


def main() -> int:
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    shared = json.loads(SHARED.read_text(encoding="utf-8"))
    manual = json.loads(MANUAL.read_text(encoding="utf-8"))["regional_trends"]["pihe"]
    if original.get("status") != "complete" or shared.get("status") != "complete":
        raise RuntimeError("Required Pihe experiments are incomplete")

    original_before = original["evaluation"]["source_total"]["before"]["model_wan_m3"]
    original_after = original["evaluation"]["source_total"]["after"]["model_wan_m3"]
    shared_before = shared["evaluation"]["current_paired_candidate"]["model_wan_m3"]
    shared_after = shared["evaluation"]["baseline_plus_operational_shape_candidate"]["model_wan_m3"]
    stages = {
        "original_v75": (float(original_before[DATE_24]), float(original_before[DATE_25])),
        "original_v75_plus_temporal_shape": (float(original_after[DATE_24]), float(original_after[DATE_25])),
        "shared_sar_upstream": (float(shared_before[DATE_24]), float(shared_before[DATE_25])),
        "shared_sar_plus_temporal_shape": (float(shared_after[DATE_24]), float(shared_after[DATE_25])),
    }
    scenarios = {
        name: (float(values[DATE_24]), float(values[DATE_25]))
        for name, values in (
            ("source_total", manual["source_total"]),
            ("component_sum", manual["component_sum"]),
        )
    }

    evaluations = {
        scenario: {
            stage: evaluate(model_24, model_25, manual_24, manual_25)
            for stage, (model_24, model_25) in stages.items()
        }
        for scenario, (manual_24, manual_25) in scenarios.items()
    }
    comparisons = {}
    for scenario, values in evaluations.items():
        comparisons[scenario] = {
            "temporal_shape_on_original_v75": delta(
                values["original_v75"], values["original_v75_plus_temporal_shape"]
            ),
            "shared_sar_upstream_only": delta(
                values["original_v75"], values["shared_sar_upstream"]
            ),
            "temporal_shape_after_shared_sar": delta(
                values["shared_sar_upstream"], values["shared_sar_plus_temporal_shape"]
            ),
            "combined_vs_original_v75": delta(
                values["original_v75"], values["shared_sar_plus_temporal_shape"]
            ),
        }

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    flat_rows = []
    for scenario, values in evaluations.items():
        for stage, metrics in values.items():
            flat_rows.append({"manual_scenario": scenario, "stage": stage, **metrics})
    with (output / "stage_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    report = {
        "status": "complete",
        "experiment": "Pihe upstream and temporal-shape contribution decomposition",
        "stages": stages,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "interpretation": [
            "The original-V75 and shared-SAR paths use different upstream outputs, so the comparison is a stage attribution rather than a factorial causal estimate.",
            "The temporal-shape branch improves both trend and 2024 absolute error on either upstream baseline.",
            "Only the shared-SAR plus temporal-shape combination reaches the manual decline direction.",
            "Both manual-total scenarios are retained because the source workbook contains a known Pihe 2025 total discrepancy.",
        ],
        "limits": [
            "The shared-SAR stage changes both dates and therefore can improve trend while worsening absolute error.",
            "Manual totals are regional aggregates rather than object labels.",
            "No production mask, parameter, or model code is changed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
