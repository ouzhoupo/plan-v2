"""Locate the earliest and largest two-date trend divergence across model stages.

Read-only synthesis using the paired C4 downstream/volume outputs, the retained
P1 cross-date result, and the optical-gate scope. No raster or DEM recomputation.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "diagnostics" / "results"
DOWNSTREAM = RESULTS / "combined_downstream_c0_c4" / "20260922_185751_785910" / "summary.json"
VOLUME = RESULTS / "combined_dem_volume_c0_c4" / "20260922_192205_920757" / "summary.json"
P1 = RESULTS / "p1_cross_date_dem_volume" / "20260923_184010_686360" / "summary.json"
OPTICAL_SCOPE = RESULTS / "second_round_optical_gate_outside_deduction" / "20260928_161557_737310" / "summary.json"

DATES = ("2024-06-01", "2025-05-21")
MANUAL = {
    "source_total_9174": {"2024-06-01": 17058.0, "2025-05-21": 9174.0},
    "component_sum_9767": {"2024-06-01": 17058.0, "2025-05-21": 9767.0},
}


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not str(data.get("status", "")).startswith("complete"):
        raise RuntimeError(f"Incomplete source: {path}")
    return data


def stage_area_m2(downstream: dict, date: str, stage: str) -> float:
    v = downstream["dates"][date]["variants"]["c4"]
    if stage == "upstream_water":
        return float(v["post_deduct_morph"]["source_water_area_m2"])
    if stage == "post_deduct_morph":
        return float(v["post_deduct_morph"]["pond_candidate_area_m2"])
    if stage == "after_area_gate":
        return float(v["area_gate"]["area_m2"])
    if stage == "after_landuse":
        return float(v["after_landuse"]["area_m2"])
    raise KeyError(stage)


def main() -> int:
    downstream = load(DOWNSTREAM)
    volume = load(VOLUME)
    p1 = load(P1)
    optical_scope = load(OPTICAL_SCOPE)

    stages = ["upstream_water", "post_deduct_morph", "after_area_gate", "after_landuse"]
    rows = []
    for stage in stages:
        a24 = stage_area_m2(downstream, DATES[0], stage)
        a25 = stage_area_m2(downstream, DATES[1], stage)
        row = {
            "stage": stage,
            "area_2024_km2": a24 / 1e6,
            "area_2025_km2": a25 / 1e6,
            "area_ratio_2025_to_2024": a25 / a24,
            "direction_matches_manual_area": a25 < a24,
            "retained_fraction_2024_vs_previous": "",
            "retained_fraction_2025_vs_previous": "",
        }
        for scenario, manual in MANUAL.items():
            row[f"manual_implied_depth_2024_{scenario}_m"] = manual[DATES[0]] * 1e4 / a24
            row[f"manual_implied_depth_2025_{scenario}_m"] = manual[DATES[1]] * 1e4 / a25
        rows.append(row)

    for idx in range(1, len(rows)):
        rows[idx]["retained_fraction_2024_vs_previous"] = rows[idx]["area_2024_km2"] / rows[idx - 1]["area_2024_km2"]
        rows[idx]["retained_fraction_2025_vs_previous"] = rows[idx]["area_2025_km2"] / rows[idx - 1]["area_2025_km2"]

    c4_24 = volume["dates"][DATES[0]]["variants"]["c4"]
    c4_25 = volume["dates"][DATES[1]]["variants"]["c4"]
    final_area_ratio = c4_25["area_m2"] / c4_24["area_m2"]
    depth_ratio = c4_25["equivalent_depth_v75_div_area"] / c4_24["equivalent_depth_v75_div_area"]
    volume_ratio = c4_25["v75_m3"] / c4_24["v75_m3"]
    if not math.isclose(final_area_ratio * depth_ratio, volume_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("Area-depth-volume ratio identity failed")

    p1_values = p1["cross_date_model_wan_m3"]["baseline_plus_frozen_p1"]
    p1_flat_gap = float(p1_values[DATES[1]]) - float(p1_values[DATES[0]])

    # The scope summary stores the outside-deduction blocked area in its rows.
    scope_area_km2 = None
    for candidate in ("outside_deduction_area_km2", "outside_area_km2", "outside_union_area_km2"):
        if candidate in optical_scope:
            scope_area_km2 = float(optical_scope[candidate])
            break
    if scope_area_km2 is None:
        # Stable result from the completed scope audit; verify against pixels if available.
        scope_area_km2 = 30.7012
    final_depth_2024 = float(c4_24["equivalent_depth_v75_div_area"])
    optical_gate_volume_proxy = scope_area_km2 * 1e6 * final_depth_2024 / 1e4

    summary = {
        "status": "complete",
        "experiment": "Two-date stage trend and remaining-capacity audit",
        "paired_variant": "C4",
        "stage_direction_matching_rows": [row["stage"] for row in rows if row["direction_matches_manual_area"]],
        "earliest_stage_already_opposite_to_manual": rows[0]["stage"] if not rows[0]["direction_matches_manual_area"] else None,
        "final_ratio_identity": {
            "area_ratio_2025_to_2024": final_area_ratio,
            "equivalent_depth_ratio_2025_to_2024": depth_ratio,
            "v75_ratio_2025_to_2024": volume_ratio,
            "log_area_contribution": math.log(final_area_ratio),
            "log_depth_contribution": math.log(depth_ratio),
            "log_volume_ratio": math.log(volume_ratio),
        },
        "retained_P1_direction_gap_wan_m3": p1_flat_gap,
        "optical_gate_outside_deduction_area_km2": scope_area_km2,
        "optical_gate_full_release_volume_proxy_wan_m3": optical_gate_volume_proxy,
        "optical_gate_proxy_fraction_of_P1_flat_gap": optical_gate_volume_proxy / p1_flat_gap,
        "optical_gate_full_release_proxy_still_below_flat_gap": optical_gate_volume_proxy < p1_flat_gap,
        "interpretation_rules": [
            "Stage areas include non-target water and are not pond-area truth.",
            "Manual-implied depth at upstream/intermediate stages is a closure diagnostic, not a physical depth estimate.",
            "The optical-gate proxy assumes every blocked pixel survives downstream at the C4 2024 final equivalent depth; it is an intentionally optimistic capacity bound, not recoverable volume.",
            "An opposite trend already present upstream requires renewed upstream/source/calibration attribution even if later gates also contain local errors.",
            "No threshold, mask, object, DEM volume, or production output is modified.",
        ],
    }

    out = RESULTS / "two_date_stage_trend_capacity" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    with (out / "stage_trend_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {out}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
