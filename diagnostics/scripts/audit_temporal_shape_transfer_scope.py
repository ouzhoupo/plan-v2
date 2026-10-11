#!/usr/bin/env python3
"""Audit cross-region and cross-date inputs for temporal-shape transfer tests."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "diagnostics/results/temporal_shape_transfer_scope"
REGIONS = ("pihe", "shihe", "hangbuhe")
DATES = (
    "2024-05-01",
    "2024-05-21",
    "2024-06-01",
    "2024-07-01",
    "2024-09-11",
    "2025-05-21",
    "2025-06-01",
    "2025-08-01",
)
PREVIOUS = {current: previous for previous, current in zip(DATES, DATES[1:])}


def compact(value: str) -> str:
    return value.replace("-", "")


def upstream_mask(day: str, region: str) -> Path:
    return ROOT / f"output/{compact(day)}/{region}/water_mask_{day}.tif"


def final_mask(day: str, region: str) -> Path:
    return ROOT / f"output/{compact(day)}/final/{region}/pond_water_mask_{day}.tif"


def config(day: str, region: str) -> Path:
    return ROOT / f"output/{compact(day)}/final/{region}/resolved_monitor_cfg_{day}.yaml"


def main() -> int:
    shared = {
        "dem": ROOT / "data/dem/dem.tif",
        "landuse": ROOT / "data/landuse_water/PSH_土地利用水体.tif",
    }
    rows = []
    for target, previous in PREVIOUS.items():
        gap_days = (date.fromisoformat(target) - date.fromisoformat(previous)).days
        for region in REGIONS:
            paths = {
                "target_upstream": upstream_mask(target, region),
                "previous_upstream": upstream_mask(previous, region),
                "target_final": final_mask(target, region),
                "target_config": config(target, region),
                "target_s1": ROOT / f"data/s1/s1_stack_{target}.tif",
                "target_s2": ROOT / f"data/s2/s2_stack_{target}.tif",
                "previous_s2": ROOT / f"data/s2/s2_stack_{previous}.tif",
                "aoi": ROOT / f"data/aoi/{region}.shp",
                **shared,
            }
            present = {name: path.exists() for name, path in paths.items()}
            ready = all(present.values())
            rows.append({
                "region": region,
                "target_date": target,
                "previous_date": previous,
                "gap_days": gap_days,
                **{f"has_{name}": value for name, value in present.items()},
                "mechanism_input_ready": ready,
                "same_date_as_qualified_pihe_test": target == "2024-06-01",
                "manual_trend_evaluation_available_in_current_diagnostics": (
                    region == "pihe" and target == "2024-06-01"
                ),
                "requires_reconstruction_of_pre_landuse_area_gate": not (
                    region == "pihe" and target == "2024-06-01"
                ),
                "missing": ";".join(name for name, value in present.items() if not value),
            })

    ready = [row for row in rows if row["mechanism_input_ready"]]
    same_date_cross_region = [
        row for row in ready
        if row["target_date"] == "2024-06-01" and row["region"] != "pihe"
    ]
    short_gap = [row for row in ready if row["gap_days"] <= 31]

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "transfer_scope.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "complete",
        "experiment": "Temporal-shape transfer input and evaluation scope audit",
        "tested_pairs": len(rows),
        "mechanism_input_ready_pairs": len(ready),
        "short_gap_ready_pairs": len(short_gap),
        "same_date_cross_region_ready": [
            {key: row[key] for key in ("region", "target_date", "previous_date", "gap_days")}
            for row in same_date_cross_region
        ],
        "recommended_next_scope": (
            "2024-06-01 Shihe and Hangbuhe, using 2024-05-21 as previous evidence, "
            "because this holds dates and season fixed while testing spatial transfer."
        ),
        "output_table": str(table),
        "interpretation": [
            "Input-ready means the saved upstream masks, target final output, S1/S2 stacks, AOI, DEM, land use, and config exist.",
            "Most transfer cases still require reconstruction of the target pre-land-use area-gate stage.",
            "Cross-region mechanism transfer can be tested without claiming manual-trend accuracy for each subregion.",
            "Long temporal gaps are listed but should not be treated as equivalent to short previous-date support."
        ],
        "limits": [
            "File availability does not establish grid compatibility or successful stage reconstruction.",
            "Only the Pihe 2024-06 to 2025-05 pair currently has the established manual trend evaluation in this diagnostic chain.",
            "No model stage is rerun and no candidate is transferred by this audit."
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
