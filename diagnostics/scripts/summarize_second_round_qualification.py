#!/usr/bin/env python3
"""Build the second-round cross-region qualification summary."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIHE = (
    ROOT / "diagnostics/results/pihe_module_contributions"
    / "20261011_180538_705248/summary.json"
)
SHIHE = (
    ROOT / "diagnostics/results/shihe_tiered_current_evidence"
    / "20261010_211915_014476/summary.json"
)
HANGBUHE = (
    ROOT / "diagnostics/results/hangbuhe_two_previous_support"
    / "20261011_163736_703291/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/second_round_qualification_summary"


def variant(report: dict, name: str) -> dict:
    for item in report["variants"]:
        if item["variant"] == name:
            return item
    raise KeyError(name)


def main() -> int:
    pihe = json.loads(PIHE.read_text(encoding="utf-8"))
    shihe = json.loads(SHIHE.read_text(encoding="utf-8"))
    hangbuhe = json.loads(HANGBUHE.read_text(encoding="utf-8"))
    if any(item.get("status") != "complete" for item in (pihe, shihe, hangbuhe)):
        raise RuntimeError("A required second-round result is incomplete")

    pihe_source = pihe["evaluations"]["source_total"]["shared_sar_plus_temporal_shape"]
    pihe_component = pihe["evaluations"]["component_sum"]["shared_sar_plus_temporal_shape"]
    pihe_source_delta = pihe["comparisons"]["source_total"]["combined_vs_original_v75"]
    pihe_component_delta = pihe["comparisons"]["component_sum"]["combined_vs_original_v75"]
    shihe_choice = variant(shihe, "plus_single_scl_ge_025")
    hangbuhe_choice = variant(hangbuhe, "may21_only_current")

    rows = [
        {
            "region": "pihe",
            "candidate": "shared_sar_plus_temporal_shape",
            "mechanism": "shared SAR upstream calibration + historical-support land-use rescue",
            "objects": "6565 temporal-shape objects",
            "increment_v75_wan_m3": 2938.6875,
            "model_ratio": pihe_source["model_ratio"],
            "manual_ratio_source_total": pihe_source["manual_ratio"],
            "manual_ratio_component_sum": pihe_component["manual_ratio"],
            "direction_matches_source_total": pihe_source["direction_matches"],
            "direction_matches_component_sum": pihe_component["direction_matches"],
            "trend_error_reduction_source_total": pihe_source_delta["trend_log_error_reduction"],
            "trend_error_reduction_component_sum": pihe_component_delta["trend_log_error_reduction"],
            "trend_error_reduction_fraction_source_total": pihe_source_delta["trend_error_reduction_fraction_of_before"],
            "trend_error_reduction_fraction_component_sum": pihe_component_delta["trend_error_reduction_fraction_of_before"],
            "absolute_error_reduction_source_total_wan_m3": pihe_source_delta["total_absolute_error_reduction_wan_m3"],
            "absolute_error_reduction_component_sum_wan_m3": pihe_component_delta["total_absolute_error_reduction_wan_m3"],
            "qualification": "qualified_for_fine_tuning",
            "main_risk": "Absolute-error conclusion changes slightly with the unresolved 2025 manual total; upstream-only correction underestimates 2025.",
        },
        {
            "region": "shihe",
            "candidate": "two_history_strong_plus_target_scl_ge_025",
            "mechanism": "two historical dates as strong evidence; target SCL as conservative secondary rescue",
            "objects": shihe_choice["objects"],
            "increment_v75_wan_m3": shihe_choice["v75_increment_wan_m3"],
            "model_ratio": shihe_choice["evaluation"]["source_total"]["model_ratio"],
            "manual_ratio_source_total": shihe_choice["evaluation"]["source_total"]["manual_ratio"],
            "manual_ratio_component_sum": shihe_choice["evaluation"]["component_sum"]["manual_ratio"],
            "direction_matches_source_total": shihe_choice["evaluation"]["source_total"]["direction_matches"],
            "direction_matches_component_sum": shihe_choice["evaluation"]["component_sum"]["direction_matches"],
            "trend_error_reduction_source_total": shihe_choice["evaluation"]["source_total"]["trend_log_error_reduction"],
            "trend_error_reduction_component_sum": shihe_choice["evaluation"]["component_sum"]["trend_log_error_reduction"],
            "trend_error_reduction_fraction_source_total": shihe_choice["evaluation"]["source_total"]["trend_log_error_reduction"] / (
                shihe_choice["evaluation"]["source_total"]["trend_log_error"]
                + shihe_choice["evaluation"]["source_total"]["trend_log_error_reduction"]
            ),
            "trend_error_reduction_fraction_component_sum": shihe_choice["evaluation"]["component_sum"]["trend_log_error_reduction"] / (
                shihe_choice["evaluation"]["component_sum"]["trend_log_error"]
                + shihe_choice["evaluation"]["component_sum"]["trend_log_error_reduction"]
            ),
            "absolute_error_reduction_source_total_wan_m3": shihe_choice["evaluation"]["source_total"]["total_absolute_error_reduction_wan_m3"],
            "absolute_error_reduction_component_sum_wan_m3": shihe_choice["evaluation"]["component_sum"]["total_absolute_error_reduction_wan_m3"],
            "qualification": "qualified_for_fine_tuning",
            "main_risk": "Manual 2024 total discrepancy remains; stable false positives can survive two historical dates.",
        },
        {
            "region": "hangbuhe",
            "candidate": "single_history_plus_loose_shape",
            "mechanism": "single pre-target historical support + loose object-shape bounds",
            "objects": hangbuhe_choice["objects"],
            "increment_v75_wan_m3": hangbuhe_choice["v75_increment_wan_m3"],
            "model_ratio": hangbuhe_choice["evaluation"]["source_total"]["model_ratio"],
            "manual_ratio_source_total": hangbuhe_choice["evaluation"]["source_total"]["manual_ratio"],
            "manual_ratio_component_sum": hangbuhe_choice["evaluation"]["component_sum"]["manual_ratio"],
            "direction_matches_source_total": hangbuhe_choice["evaluation"]["source_total"]["direction_matches"],
            "direction_matches_component_sum": hangbuhe_choice["evaluation"]["component_sum"]["direction_matches"],
            "trend_error_reduction_source_total": hangbuhe_choice["evaluation"]["source_total"]["trend_log_error_reduction"],
            "trend_error_reduction_component_sum": hangbuhe_choice["evaluation"]["component_sum"]["trend_log_error_reduction"],
            "trend_error_reduction_fraction_source_total": hangbuhe_choice["evaluation"]["source_total"]["trend_log_error_reduction"] / hangbuhe["baseline_evaluation"]["source_total"]["trend_log_error"],
            "trend_error_reduction_fraction_component_sum": hangbuhe_choice["evaluation"]["component_sum"]["trend_log_error_reduction"] / hangbuhe["baseline_evaluation"]["component_sum"]["trend_log_error"],
            "absolute_error_reduction_source_total_wan_m3": hangbuhe_choice["evaluation"]["source_total"]["total_absolute_error_reduction_wan_m3"],
            "absolute_error_reduction_component_sum_wan_m3": hangbuhe_choice["evaluation"]["component_sum"]["total_absolute_error_reduction_wan_m3"],
            "qualification": "qualified_for_fine_tuning",
            "main_risk": "Single-history candidate retains agricultural and edge false positives; hard confidence gates remove real ponds.",
        },
    ]

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "regional_qualification.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "complete",
        "experiment": "Second-round cross-region qualification summary",
        "qualification_rule": [
            "Candidate must improve trend error and retain the manual direction.",
            "Absolute-error effects must be reported under every unresolved manual-total scenario.",
            "Candidate mechanism must have GIS evidence and cannot be selected by manual totals alone.",
        ],
        "regions": rows,
        "overall_decision": "qualified_for_module_fine_tuning_not_production",
        "next_stage": [
            "Preserve the model chain and implement corrections as bounded module options.",
            "Separate cross-date upstream calibration from land-use rescue confidence scoring.",
            "Validate additional date pairs before freezing regional or shared parameters.",
            "Keep absolute volume and trend fit as separate objectives.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
