#!/usr/bin/env python3
"""Audit where the retrospective rescue volume is concentrated.

This diagnostic joins exact DEM volumes, temporal evidence, and object shape.
It does not define an operational selection rule or change any model output.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import fiona
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
MATERIALIZED = (
    ROOT / "diagnostics/results/temporal_landuse_rescue_candidates_2024"
    / "20261010_181232_993446"
)
OPERATIONAL = (
    ROOT / "diagnostics/results/operational_temporal_rescue_dem_volume_2024"
    / "20261010_182317_438951"
)
RETROSPECTIVE = (
    ROOT / "diagnostics/results/retrospective_temporal_rescue_dem_volume_2024"
    / "20261010_182755_568239"
)
EVIDENCE = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_2024"
    / "20261010_173122_234024/rejected_objects_temporal_evidence.csv"
)
BASE_RESULT = ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
OUTPUT_ROOT = ROOT / "diagnostics/results/retrospective_rescue_volume_concentration"
MANUAL = {DATE_24: 17058.0, DATE_25: 9174.0}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def area_bin(area_m2: float) -> str:
    if area_m2 < 2000:
        return "005_02ha"
    if area_m2 < 10000:
        return "02_1ha"
    if area_m2 < 50000:
        return "1_5ha"
    if area_m2 < 200000:
        return "5_20ha"
    return "20_50ha"


def scl_bin(value: float) -> str:
    if value == 0:
        return "zero"
    if value < 0.10:
        return "gt0_lt010"
    if value < 0.25:
        return "010_lt025"
    return "ge025"


def aggregate(rows: list[dict], key: str) -> dict:
    result = {}
    for row in rows:
        label = str(row[key])
        item = result.setdefault(label, {"objects": 0, "area_m2": 0.0, "v75_m3": 0.0})
        item["objects"] += 1
        item["area_m2"] += float(row["area_m2"])
        item["v75_m3"] += float(row["v75_m3"])
    total = sum(float(row["v75_m3"]) for row in rows)
    for item in result.values():
        item["area_km2"] = item["area_m2"] / 1e6
        item["v75_wan_m3"] = item["v75_m3"] / 10000.0
        item["fraction_of_total_v75"] = item["v75_m3"] / total if total else 0.0
    return result


def main() -> int:
    materialized = json.loads((MATERIALIZED / "summary.json").read_text(encoding="utf-8"))
    retrospective = json.loads((RETROSPECTIVE / "summary.json").read_text(encoding="utf-8"))
    base = json.loads((BASE_RESULT / "summary.json").read_text(encoding="utf-8"))
    if any(item.get("status") != "complete" for item in (materialized, retrospective, base)):
        raise RuntimeError("Required source experiment is incomplete")

    memberships = read_csv(MATERIALIZED / "candidate_membership.csv")
    retrospective_ids = {
        int(row["object_id"])
        for row in memberships
        if row["branch"] == "retrospective_previous_and_future"
    }
    triple_ids = {
        int(row["object_id"])
        for row in memberships
        if row["branch"] == "triple_strong"
    }
    evidence = {int(row["object_id"]): row for row in read_csv(EVIDENCE)}

    volume_rows = {}
    for row in read_csv(OPERATIONAL / "operational_candidates_with_volume.csv"):
        object_id = int(row["pond_id"])
        if object_id in triple_ids:
            volume_rows[object_id] = row
    for row in read_csv(RETROSPECTIVE / "retrospective_exclusive_objects_with_volume.csv"):
        volume_rows[int(row["pond_id"])] = row
    if set(volume_rows) != retrospective_ids:
        raise RuntimeError(
            f"Volume join mismatch: {len(volume_rows)} rows for {len(retrospective_ids)} objects"
        )

    geometries = {}
    with fiona.open(MATERIALIZED / "retrospective_previous_and_future.gpkg") as source:
        for feature in source:
            props = dict(feature["properties"])
            geometries[int(props["object_id"])] = shape(feature["geometry"])
    if set(geometries) != retrospective_ids:
        raise RuntimeError("Geometry join mismatch")

    rows = []
    for object_id in sorted(retrospective_ids):
        geom = geometries[object_id]
        ev = evidence[object_id]
        vol = volume_rows[object_id]
        area = float(ev["area_m2"])
        minx, miny, maxx, maxy = geom.bounds
        width = maxx - minx
        height = maxy - miny
        short_side = min(width, height)
        elongation = max(width, height) / short_side if short_side > 0 else None
        bbox_fill = area / (width * height) if width > 0 and height > 0 else None
        perimeter = float(geom.length)
        compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else None
        target_scl = float(ev["scl_target_2024_06"])
        rows.append({
            "object_id": object_id,
            "is_triple_strong": object_id in triple_ids,
            "area_m2": area,
            "area_bin": area_bin(area),
            "v75_m3": float(vol.get("v75_m3") or 0.0),
            "qa": vol.get("qa") or "",
            "previous_near_fraction": float(ev["previous_near"]),
            "target_scl_fraction": target_scl,
            "target_scl_bin": scl_bin(target_scl),
            "future_near_fraction": float(ev["future_near"]),
            "bbox_fill": bbox_fill,
            "compactness": compactness,
            "elongation": elongation,
            "center_x": float(ev["center_x"]),
            "center_y": float(ev["center_y"]),
        })

    total_v75 = sum(row["v75_m3"] for row in rows)
    ranked = sorted(rows, key=lambda row: row["v75_m3"], reverse=True)
    concentration = {}
    for count in (10, 50, 100, 500, 1000):
        subtotal = sum(row["v75_m3"] for row in ranked[:count])
        concentration[f"top_{count}"] = {
            "objects": min(count, len(ranked)),
            "v75_m3": subtotal,
            "v75_wan_m3": subtotal / 10000.0,
            "fraction_of_total_v75": subtotal / total_v75 if total_v75 else 0.0,
        }

    baseline_24 = float(base["paired_reference_2024"]["volume"]["v75_m3"]) / 10000.0
    candidate_25 = float(base["conservative_gate_2025"]["volume"]["v75_m3"]) / 10000.0
    manual_ratio = MANUAL[DATE_25] / MANUAL[DATE_24]
    exact_ratio_2024 = candidate_25 / manual_ratio
    required_increment = max(0.0, exact_ratio_2024 - baseline_24)
    full_increment = total_v75 / 10000.0

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "retrospective_objects_volume_features.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "complete",
        "experiment": "Retrospective rescue volume concentration and shape audit",
        "region": "pihe",
        "checks": {
            "objects": len(rows),
            "total_v75_m3": total_v75,
            "expected_v75_m3": retrospective["retrospective_full_metrics"]["v75_m3"],
            "volume_difference_m3": total_v75
            - float(retrospective["retrospective_full_metrics"]["v75_m3"]),
        },
        "trend_budget": {
            "current_2024_wan_m3": baseline_24,
            "fixed_2025_candidate_wan_m3": candidate_25,
            "manual_ratio": manual_ratio,
            "2024_needed_for_exact_manual_ratio_wan_m3": exact_ratio_2024,
            "required_rescue_increment_wan_m3": required_increment,
            "full_retrospective_increment_wan_m3": full_increment,
            "fraction_of_full_increment_needed": required_increment / full_increment,
        },
        "v75_concentration": concentration,
        "by_area_bin": aggregate(rows, "area_bin"),
        "by_target_scl_bin": aggregate(rows, "target_scl_bin"),
        "by_triple_membership": aggregate(rows, "is_triple_strong"),
        "output_table": str(table),
        "next_step": (
            "Use concentration and shape strata to choose new GIS-review objects, especially high-volume "
            "retrospective-exclusive contributors and compact versus elongated objects."
        ),
        "limits": [
            "Ranking by V75 is diagnostic and is not a valid water-selection rule.",
            "Matching the manual ratio numerically is not sufficient for adoption.",
            "Future model support remains retrospective and prohibited at inference time.",
            "Shape measures describe raster components and do not establish survey membership.",
            "Production code and formal outputs are unchanged.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print("STATUS: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
