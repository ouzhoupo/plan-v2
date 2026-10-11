#!/usr/bin/env python3
"""Audit reliability and robustness of the second-round correction directions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION = (
    ROOT / "diagnostics/results/second_round_qualification_summary"
    / "20261011_181811_281295/summary.json"
)
PIHE_SENSITIVITY = (
    ROOT / "diagnostics/results/operational_temporal_shape_sensitivity"
    / "20261010_193042_444718/summary.json"
)
SHIHE_SENSITIVITY = (
    ROOT / "diagnostics/results/shihe_temporal_shape_sensitivity"
    / "20261010_205634_315195/summary.json"
)
HANGBUHE_SENSITIVITY = (
    ROOT / "diagnostics/results/hangbuhe_two_previous_support"
    / "20261011_163736_703291/summary.json"
)
PIHE_CONTRIBUTION = (
    ROOT / "diagnostics/results/pihe_module_contributions"
    / "20261011_180538_705248/summary.json"
)
TRANSFER_SCOPE = (
    ROOT / "diagnostics/results/temporal_shape_transfer_scope"
    / "20261010_194211_190941/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/second_round_robustness_audit"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def region_from_path(path: Path, row: dict) -> str:
    if row.get("region") in {"pihe", "shihe", "hangbuhe"}:
        return row["region"]
    lower = str(path).lower()
    if "hangbuhe" in lower:
        return "hangbuhe"
    if "shihe" in lower:
        return "shihe"
    return "pihe"


def scope_class(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "possible_target":
        return "possible_target"
    if value == "uncertain_target" or value.startswith("uncertain_"):
        return "uncertain"
    if value in {"non_target", "nonwater", "probable_non_target", "probable_nonwater"}:
        return "non_target"
    return "unclassified"


def direction_stability(items: list[dict], accessor) -> dict:
    flags = [bool(accessor(item)) for item in items]
    return {
        "tested_variants": len(flags),
        "direction_matching_variants": sum(flags),
        "direction_matching_fraction": sum(flags) / len(flags) if flags else None,
    }


def main() -> int:
    qualification = read_json(QUALIFICATION)
    pihe_sensitivity = read_json(PIHE_SENSITIVITY)
    shihe_sensitivity = read_json(SHIHE_SENSITIVITY)
    hangbuhe_sensitivity = read_json(HANGBUHE_SENSITIVITY)
    pihe_contribution = read_json(PIHE_CONTRIBUTION)
    transfer_scope = read_json(TRANSFER_SCOPE)
    if any(
        report.get("status") != "complete"
        for report in (
            qualification,
            pihe_sensitivity,
            shihe_sensitivity,
            hangbuhe_sensitivity,
            pihe_contribution,
            transfer_scope,
        )
    ):
        raise RuntimeError("A required robustness source is incomplete")

    qualification_rows = qualification["regions"]
    manual_scenario_direction = {
        row["region"]: bool(row["direction_matches_source_total"] and row["direction_matches_component_sum"])
        for row in qualification_rows
    }
    manual_scenario_absolute = {
        row["region"]: {
            "source_total_improves": float(row["absolute_error_reduction_source_total_wan_m3"]) > 0,
            "component_sum_improves": float(row["absolute_error_reduction_component_sum_wan_m3"]) > 0,
        }
        for row in qualification_rows
    }

    parameter_stability = {
        "pihe": direction_stability(
            pihe_sensitivity["scenarios"], lambda item: item["direction_matches_manual"]
        ),
        "shihe_source_total": direction_stability(
            shihe_sensitivity["variants"],
            lambda item: item["evaluation"]["source_total"]["direction_matches"],
        ),
        "shihe_component_sum": direction_stability(
            shihe_sensitivity["variants"],
            lambda item: item["evaluation"]["component_sum"]["direction_matches"],
        ),
        "hangbuhe": direction_stability(
            hangbuhe_sensitivity["variants"],
            lambda item: item["evaluation"]["source_total"]["direction_matches"],
        ),
    }

    label_files = []
    patterns = (
        "diagnostics/results/operational_temporal_shape_review/*/assistant_visual_labels.csv",
        "diagnostics/results/operational_temporal_shape_boundary_review/*/assistant_visual_labels.csv",
        "diagnostics/results/temporal_shape_cross_region_review/*/assistant_visual_labels.csv",
        "diagnostics/results/shihe_scl_secondary_rescue_review/*/assistant_visual_labels.csv",
        "diagnostics/results/hangbuhe_two_previous_support_review/*/assistant_visual_labels.csv",
        "diagnostics/results/hangbuhe_dual_source_secondary_review/*/assistant_visual_labels.csv",
        "diagnostics/results/shihe_two_previous_support_review/*/review_manifest.csv",
        "diagnostics/results/shihe_temporal_shape_boundary_review/*/review_manifest.csv",
    )
    for pattern in patterns:
        label_files.extend(ROOT.glob(pattern))
    counts = defaultdict(Counter)
    seen = set()
    for path in label_files:
        for row in read_csv(path):
            if (row.get("proxy_label") or "").strip().lower() in {"", "unreviewed"}:
                continue
            region = region_from_path(path, row)
            key = (region, row.get("object_id"), row.get("review_group"), row.get("proxy_label"))
            if key in seen:
                continue
            seen.add(key)
            counts[region][scope_class(row.get("target_scope_interpretation", ""))] += 1

    gis_rows = []
    for region in ("pihe", "shihe", "hangbuhe"):
        counter = counts[region]
        total = sum(counter.values())
        gis_rows.append({
            "region": region,
            "reviewed_objects": total,
            "possible_target": counter["possible_target"],
            "uncertain": counter["uncertain"],
            "non_target": counter["non_target"],
            "unclassified": counter["unclassified"],
            "possible_target_fraction": counter["possible_target"] / total if total else None,
            "non_target_fraction": counter["non_target"] / total if total else None,
        })

    pihe_original_temporal_direction = pihe_contribution["evaluations"]["source_total"][
        "original_v75_plus_temporal_shape"
    ]["direction_matches"]
    pihe_combined_direction = pihe_contribution["evaluations"]["source_total"][
        "shared_sar_plus_temporal_shape"
    ]["direction_matches"]

    matrix = [
        {
            "dimension": "mechanism_replication",
            "rating": "medium_high",
            "evidence": "Historical-support land-use rescue has non-zero, GIS-confirmed effects in Pihe, Shihe, and Hangbuhe, including both manual decline and increase cases.",
            "failure_condition": "Regional evidence strength differs, so one hard threshold is not transferable.",
        },
        {
            "dimension": "trend_direction_and_manual_scenario",
            "rating": "high",
            "evidence": f"All {sum(manual_scenario_direction.values())}/3 regional candidates match direction under both retained manual-total scenarios.",
            "failure_condition": "Pihe direction depends on combining upstream calibration with temporal rescue.",
        },
        {
            "dimension": "local_parameter_neighborhood",
            "rating": "medium_high",
            "evidence": "Most one-factor threshold variants retain direction; Pihe loses direction only under the strongest 1 ha minimum-area restriction.",
            "failure_condition": "Large minimum-area or strict fill gates remove too much small-pond capacity.",
        },
        {
            "dimension": "absolute_volume",
            "rating": "medium_low",
            "evidence": "Shihe and Hangbuhe improve under both manual scenarios; Pihe improves under source total but slightly worsens under component sum.",
            "failure_condition": "Pihe shared-SAR upstream correction strongly underestimates 2025 absolute volume.",
        },
        {
            "dimension": "upstream_baseline_dependence",
            "rating": "medium_low",
            "evidence": f"Pihe temporal rescue on original V75 matches direction={pihe_original_temporal_direction}, while shared-SAR plus rescue matches direction={pihe_combined_direction}.",
            "failure_condition": "The qualified Pihe trend is a module interaction and does not survive replacement by the original upstream baseline.",
        },
        {
            "dimension": "gis_object_evidence",
            "rating": "medium",
            "evidence": "Held-out reviews repeatedly show real ponds among rescued objects and false positives among high-confidence groups; this validates both benefit and known side effects.",
            "failure_condition": "Samples are mechanism checks rather than probability samples and cannot estimate accuracy.",
        },
        {
            "dimension": "cross_date_transfer",
            "rating": "low",
            "evidence": f"Inputs exist for {transfer_scope['mechanism_input_ready_pairs']}/{transfer_scope['tested_pairs']} pairs and {transfer_scope['short_gap_ready_pairs']} short-gap pairs, but the qualified chain is still centered on one target/manual pair.",
            "failure_condition": "Season, rainfall, SAR acquisition geometry, optical cloud, and land-use timing may change the evidence behavior.",
        },
        {
            "dimension": "production_readiness",
            "rating": "low",
            "evidence": "No production model, mask, or frozen parameter has been changed; manual data remain regional aggregates.",
            "failure_condition": "Object-level truth, multi-date validation, and bounded implementation are still missing.",
        },
    ]

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "robustness_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    with (output / "gis_evidence_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gis_rows[0]))
        writer.writeheader()
        writer.writerows(gis_rows)

    report = {
        "status": "complete",
        "experiment": "Second-round correction reliability and robustness audit",
        "manual_scenario_direction_stability": manual_scenario_direction,
        "manual_scenario_absolute_stability": manual_scenario_absolute,
        "parameter_direction_stability": parameter_stability,
        "gis_evidence_counts": gis_rows,
        "robustness_matrix": matrix,
        "overall_decision": "reliable_enough_for_controlled_fine_tuning_not_parameter_freezing",
        "decision_basis": [
            "Mechanisms replicate across three regions and opposite trend directions.",
            "Direction is stable across retained manual-total scenarios and most local parameter perturbations.",
            "Absolute volume, Pihe upstream dependence, and cross-date transfer remain the limiting dimensions.",
            "GIS evidence confirms both true-water recovery and false-positive risk, so confidence evidence must remain auditable rather than become an unchecked hard gate.",
        ],
        "required_before_production": [
            "Test at least two additional short-gap target dates spanning different observation conditions.",
            "Preserve both manual-total scenarios until source discrepancies are resolved.",
            "Implement upstream calibration and land-use rescue as separately switchable modules.",
            "Evaluate object-level stratified samples or matched manual polygons when available.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
