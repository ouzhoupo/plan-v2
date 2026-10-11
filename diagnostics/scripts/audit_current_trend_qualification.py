"""Audit existing full-v75 correction results against the trend-entry gate.

This is a read-only synthesis. It does not combine incompatible experiments,
select thresholds, modify masks, or rerun DEM volume calculations.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "diagnostics" / "results"

A1 = RESULTS / "a1_dem_volume" / "20260920_173636_938983" / "summary.json"
A2 = RESULTS / "a2_previous_date_dem_volume" / "20260922_144649_744718" / "summary.json"
A3 = RESULTS / "a3_dual_pol_dem_volume" / "20260922_134307_778901" / "summary.json"
A4 = RESULTS / "a4_dem_volume" / "20260921_160204_732788" / "summary.json"
EXACT = RESULTS / "a1b_exact_landuse_volume" / "20260921_113337_918956" / "summary.json"
COMBINED = RESULTS / "combined_dem_volume_c0_c4" / "20260922_192205_920757" / "summary.json"
P_RESCUE = RESULTS / "c4_2024_landuse_rescue_dem_volume" / "20260923_161303_896823" / "summary.json"
P1_CROSS = RESULTS / "p1_cross_date_dem_volume" / "20260923_184010_686360" / "summary.json"

MANUAL_2024 = 17058.0
MANUAL_2025_SCENARIOS = {"source_total_9174": 9174.0, "component_sum_9767": 9767.0}


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise RuntimeError(f"Incomplete source: {path}")
    return data


def wan(value_m3: float) -> float:
    return float(value_m3) / 10000.0


def add(rows: list[dict], name: str, v24: float, v25: float, source: Path,
        evidence_class: str, retained_status: str, comparability_note: str) -> None:
    ratio = v25 / v24
    row = {
        "candidate": name,
        "model_2024_wan_m3": v24,
        "model_2025_wan_m3": v25,
        "model_ratio_2025_to_2024": ratio,
        "direction_matches_manual": ratio < 1.0,
        "additional_2024_needed_for_flat_wan_m3": max(0.0, v25 - v24),
        "evidence_class": evidence_class,
        "retained_status": retained_status,
        "comparability_note": comparability_note,
        "source_summary": str(source),
    }
    for scenario, manual_2025 in MANUAL_2025_SCENARIOS.items():
        manual_ratio = manual_2025 / MANUAL_2024
        target_2024 = v25 / manual_ratio
        row[f"manual_ratio_{scenario}"] = manual_ratio
        row[f"absolute_log_ratio_error_{scenario}"] = abs(math.log(ratio / manual_ratio))
        row[f"additional_2024_needed_for_exact_ratio_{scenario}_wan_m3"] = max(0.0, target_2024 - v24)
        row[f"two_date_total_absolute_error_{scenario}_wan_m3"] = (
            abs(v24 - MANUAL_2024) + abs(v25 - manual_2025)
        )
    rows.append(row)


def main() -> int:
    a1, a2, a3, a4 = map(load, (A1, A2, A3, A4))
    exact, combined, p_rescue, p1_cross = map(load, (EXACT, COMBINED, P_RESCUE, P1_CROSS))

    rows: list[dict] = []
    add(rows, "frozen_baseline",
        wan(a1["dates"]["2024-06-01"]["baseline"]["v75_m3"]),
        wan(a1["dates"]["2025-05-21"]["baseline"]["v75_m3"]),
        A1, "baseline", "reference", "Frozen historical baseline.")
    add(rows, "A1_optical_confidence_bypass",
        wan(a1["dates"]["2024-06-01"]["a1"]["v75_m3"]),
        wan(a1["dates"]["2025-05-21"]["a1"]["v75_m3"]),
        A1, "single_factor", "mechanism_candidate", "2025 was unchanged by A1 in this experiment.")
    add(rows, "A2_previous_date_hard_fill_upper_bound",
        wan(a2["a2"]["v75_m3"]),
        wan(a1["dates"]["2025-05-21"]["baseline"]["v75_m3"]),
        A2, "diagnostic_upper_bound", "not_production", "2025 held at frozen baseline; hard fill is an upper bound.")
    add(rows, "A3_dual_pol_training",
        wan(a3["dates"]["2024-06-01"]["dual_pol"]["v75_m3"]),
        wan(a3["dates"]["2025-05-21"]["dual_pol"]["v75_m3"]),
        A3, "single_factor", "retained_candidate", "Uses paired retraining baseline; compare within A3 first.")
    add(rows, "A4_SAR_NaN_validity",
        wan(a4["dates"]["2024-06-01"]["a4"]["v75_m3"]),
        wan(a4["dates"]["2025-05-21"]["a4"]["v75_m3"]),
        A4, "single_factor", "model_completeness_candidate", "Mechanism-consistency fix with small trend contribution.")

    exact_group = exact["survey_error_account"]["groups"]["a1_exact"]["dates"]
    add(rows, "A1_plus_exact_landuse",
        wan(exact_group["2024-06-01"]["model_m3"]),
        wan(exact_group["2025-05-21"]["model_m3"]),
        EXACT, "coarse_combination", "candidate", "Exact land-use rule has its own documented scope and side effects.")

    for variant in ("c0", "c1", "c2", "c3", "c4"):
        ev = combined["survey_evaluation"][variant]
        add(rows, f"combined_{variant}",
            float(ev["model_wan_m3"]["2024-06-01"]),
            float(ev["model_wan_m3"]["2025-05-21"]),
            COMBINED, "paired_combination_stage", "diagnostic_sequence",
            "C0-C4 are sequential stages in one paired combination run.")

    cross = p1_cross["cross_date_model_wan_m3"]
    p1_values = cross["baseline_plus_frozen_p1"]
    add(rows, "current_baseline_plus_frozen_P1",
        float(p1_values["2024-06-01"]), float(p1_values["2025-05-21"]),
        P1_CROSS, "cross_date_candidate", "retained_candidate",
        "P1 is retained as a mechanism candidate but is not a universal rule.")

    p_eval = p_rescue["variant_evaluation"]["b3_baseline_plus_p1_p2"]
    add(rows, "P1_plus_P2_diagnostic_upper_bound",
        float(p_eval["model_wan_m3"]["2024-06-01"]),
        float(p_eval["model_wan_m3"]["2025-05-21"]),
        P_RESCUE, "diagnostic_upper_bound", "P2_rejected_after_new_holdout",
        "Included only to measure capacity; P2/T1/T2 did not pass later holdout review.")

    out = RESULTS / "trend_qualification_audit" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    fields = list(rows[0])
    with (out / "candidate_trend_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: float(row["absolute_log_ratio_error_source_total_9174"]))
    direction_matches = [row["candidate"] for row in rows if row["direction_matches_manual"]]
    summary = {
        "status": "complete" if direction_matches else "complete_no_direction_match",
        "experiment": "Existing correction trend-qualification audit",
        "manual_2024_wan_m3": MANUAL_2024,
        "manual_2025_scenarios_wan_m3": MANUAL_2025_SCENARIOS,
        "evaluated_rows": len(rows),
        "direction_matching_rows": direction_matches,
        "best_existing_ratio_row_by_9174_log_error": ranked[0]["candidate"],
        "best_existing_ratio": ranked[0]["model_ratio_2025_to_2024"],
        "best_existing_additional_2024_needed_for_flat_wan_m3": ranked[0]["additional_2024_needed_for_flat_wan_m3"],
        "limits": [
            "Rows from different baselines are not additive and are not a proposed combined model.",
            "Diagnostic upper bounds and rejected candidates are retained only to measure capacity.",
            "Direction agreement is a necessary entry gate, not sufficient evidence for adoption.",
            "No numerical tolerance for 'approximately matching' is selected in this audit.",
            "No masks, objects, DEM volumes, or production outputs are modified.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "This audit consolidates already completed full-v75 results.\n"
        "It checks the necessary direction gate before any local fine tuning.\n"
        "It does not combine experiments or create a new model output.\n",
        encoding="utf-8",
    )
    print(f"RESULT_FOLDER: {out}")
    print(f"STATUS: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
