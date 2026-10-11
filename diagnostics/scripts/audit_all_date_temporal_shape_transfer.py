#!/usr/bin/env python3
"""Run the temporal-shape mechanism audit across every saved date and region.

Missing target/previous inputs are kept as blank rows instead of aborting the
audit. The first date has no previous date and the last date has no follow-up
proxy; both remain explicit in the output. No DEM volume is computed and no
production file is modified.
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import audit_temporal_shape_cross_region_bridge as bridge  # noqa: E402


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
REGIONS = ("pihe", "shihe", "hangbuhe")
OUTPUT_ROOT = ROOT / "diagnostics/results/all_date_temporal_shape_transfer"


def compact(day: str) -> str:
    return day.replace("-", "")


def required_paths(target: str, previous: str, region: str) -> dict[str, Path]:
    target_root = ROOT / "output" / compact(target)
    final = target_root / "final" / region
    return {
        "target_upstream": target_root / region / f"water_mask_{target}.tif",
        "previous_upstream": (
            ROOT / "output" / compact(previous) / region / f"water_mask_{previous}.tif"
        ),
        "target_config": final / f"resolved_monitor_cfg_{target}.yaml",
        "target_storage": final / f"pond_storage_{target}.csv",
        "target_final_mask": final / f"pond_water_mask_{target}.tif",
    }


def gap_class(days: int | None) -> str:
    if days is None:
        return ""
    if days <= 31:
        return "short_le_31d"
    if days <= 90:
        return "medium_32_90d"
    return "long_gt_90d"


def blank_row(
    region: str,
    target: str,
    previous: str | None,
    follow_up: str | None,
    gap_days: int | None,
    status: str,
    missing: str = "",
    error: str = "",
) -> dict:
    return {
        "region": region,
        "target_date": target,
        "previous_date": previous or "",
        "follow_up_proxy_date": follow_up or "",
        "gap_days": gap_days if gap_days is not None else "",
        "gap_class": gap_class(gap_days),
        "status": status,
        "missing_inputs": missing,
        "error": error,
        "baseline_exact": "",
        "final_objects": "",
        "final_area_m2": "",
        "rejected_objects": "",
        "rejected_area_m2": "",
        "candidate_objects": "",
        "candidate_area_m2": "",
        "candidate_fraction_rejected_objects": "",
        "candidate_fraction_rejected_area": "",
        "follow_up_proxy_objects": "",
        "follow_up_proxy_area_m2": "",
        "follow_up_proxy_object_fraction": "",
        "follow_up_proxy_area_fraction": "",
        "object_table": "",
    }


def result_row(
    region: str,
    target: str,
    previous: str,
    follow_up: str | None,
    gap_days: int,
    result: dict,
) -> dict:
    base = result["baseline_reproduction"]
    candidate = result["loose_candidate"]
    exact = (
        base["prepared_added_pixels"] == 0
        and base["prepared_removed_pixels"] == 0
        and base["expected_final_objects"] == base["reproduced_final_objects"]
        and abs(base["expected_final_area_m2"] - base["reproduced_final_area_m2"]) <= 0.001
    )
    return {
        "region": region,
        "target_date": target,
        "previous_date": previous,
        "follow_up_proxy_date": follow_up or "",
        "gap_days": gap_days,
        "gap_class": gap_class(gap_days),
        "status": "complete",
        "missing_inputs": "",
        "error": "",
        "baseline_exact": exact,
        "final_objects": result["after_landuse"]["object_count"],
        "final_area_m2": result["after_landuse"]["area_m2"],
        "rejected_objects": result["landuse_rejected"]["objects"],
        "rejected_area_m2": result["landuse_rejected"]["area_m2"],
        "candidate_objects": candidate["objects"],
        "candidate_area_m2": candidate["area_m2"],
        "candidate_fraction_rejected_objects": candidate["fraction_of_rejected_objects"],
        "candidate_fraction_rejected_area": candidate["fraction_of_rejected_area"],
        "follow_up_proxy_objects": (
            candidate["future_proxy_objects"]
            if candidate["future_proxy_objects"] is not None else ""
        ),
        "follow_up_proxy_area_m2": (
            candidate["future_proxy_area_m2"]
            if candidate["future_proxy_area_m2"] is not None else ""
        ),
        "follow_up_proxy_object_fraction": (
            candidate["future_proxy_object_fraction"]
            if candidate["future_proxy_object_fraction"] is not None else ""
        ),
        "follow_up_proxy_area_fraction": (
            candidate["future_proxy_area_fraction"]
            if candidate["future_proxy_area_fraction"] is not None else ""
        ),
        "object_table": result["output_table"],
    }


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    report = {
        "status": "running",
        "experiment": "All-date temporal-shape mechanism transfer audit",
        "dates": list(DATES),
        "regions": list(REGIONS),
        "selection": (
            "previous support >=0.25, area 0.2-5 ha, "
            "bbox fill >=0.45, elongation <=3"
        ),
        "rows": rows,
    }

    for index, target in enumerate(DATES):
        previous = DATES[index - 1] if index > 0 else None
        follow_up = DATES[index + 1] if index + 1 < len(DATES) else None
        gap_days = (
            (date.fromisoformat(target) - date.fromisoformat(previous)).days
            if previous is not None else None
        )
        for region in REGIONS:
            print(f"All-date audit: {target} {region}", flush=True)
            if previous is None:
                rows.append(blank_row(
                    region, target, None, follow_up, None, "no_previous_date"
                ))
                continue

            paths = required_paths(target, previous, region)
            missing = [name for name, path in paths.items() if not path.exists()]
            if missing:
                rows.append(blank_row(
                    region, target, previous, follow_up, gap_days,
                    "missing_input", ";".join(missing)
                ))
                continue

            if follow_up is not None:
                follow_path = (
                    ROOT / "output" / compact(follow_up) / region
                    / f"water_mask_{follow_up}.tif"
                )
                if not follow_path.exists():
                    follow_up_for_run = None
                else:
                    follow_up_for_run = follow_up
            else:
                follow_up_for_run = None

            bridge.TARGET = target
            bridge.PREVIOUS = previous
            bridge.FUTURE = follow_up_for_run
            pair_dir = output / f"{compact(previous)}_to_{compact(target)}"
            pair_dir.mkdir(exist_ok=True)
            try:
                result = bridge.process_region(region, pair_dir)
                rows.append(result_row(
                    region, target, previous, follow_up_for_run, gap_days, result
                ))
            except Exception as exc:  # keep later dates running and expose failure
                rows.append(blank_row(
                    region, target, previous, follow_up_for_run, gap_days,
                    "failed", error=f"{type(exc).__name__}: {exc}"
                ))
                (pair_dir / f"{region}_error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )

            write_table(output / "all_date_transfer_rows.partial.csv", rows)
            (output / "summary.partial.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    complete = [row for row in rows if row["status"] == "complete"]
    missing = [row for row in rows if row["status"] == "missing_input"]
    failed = [row for row in rows if row["status"] == "failed"]
    report["status"] = "complete" if not failed else "complete_with_failures"
    report["counts"] = {
        "total_date_region_rows": len(rows),
        "complete": len(complete),
        "no_previous_date": sum(row["status"] == "no_previous_date" for row in rows),
        "missing_input": len(missing),
        "failed": len(failed),
        "complete_with_follow_up_proxy": sum(
            row["status"] == "complete" and row["follow_up_proxy_date"] != ""
            for row in rows
        ),
    }
    report["missing_rows"] = [
        {
            "region": row["region"],
            "target_date": row["target_date"],
            "previous_date": row["previous_date"],
            "missing_inputs": row["missing_inputs"],
        }
        for row in missing
    ]
    report["interpretation"] = [
        "Blank rows are deliberate and preserve missing dates in the full panel.",
        "The first date cannot test historical support because no earlier saved date exists.",
        "The last date can test candidate formation but has blank follow-up proxy fields.",
        "Short, medium, and long gaps must be summarized separately.",
        "Follow-up support is a retrospective risk proxy, not a truth label.",
        "The audit does not calculate DEM volume or manual-trend accuracy.",
    ]
    table = output / "all_date_transfer_rows.csv"
    write_table(table, rows)
    report["output_table"] = str(table)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {report['status']}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
