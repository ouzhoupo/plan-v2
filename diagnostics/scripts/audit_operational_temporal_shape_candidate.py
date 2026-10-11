#!/usr/bin/env python3
"""Audit operational previous-date plus shape candidates against future persistence.

Selection uses only previous-date evidence and object morphology. Future-date
support is reported as a retrospective evaluation proxy, never as an input.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_RESULT = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_2024/20261010_173122_234024"
)
AREA_GATE = (
    ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
    / "paired_reference_downstream/2024-06-01/shared_reference_after_area_gate.tif"
)
VOLUME_RESULT = (
    ROOT / "diagnostics/results/retrospective_rescue_volume_concentration"
    / "20261010_183252_458888"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/operational_temporal_shape_candidate"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def summarize(selected: list[dict], retrospective_ids: set[int], v75: dict[int, float], totals: dict) -> dict:
    selected_ids = {row["object_id"] for row in selected}
    future_ids = selected_ids & retrospective_ids
    selected_area = sum(row["area_m2"] for row in selected)
    future_area = sum(row["area_m2"] for row in selected if row["object_id"] in future_ids)
    retained_v75 = sum(v75.get(object_id, 0.0) for object_id in future_ids)
    return {
        "selected": {
            "objects": len(selected),
            "area_m2": selected_area,
            "area_km2": selected_area / 1e6,
        },
        "future_persistent_proxy": {
            "objects": len(future_ids),
            "area_m2": future_area,
            "area_km2": future_area / 1e6,
            "object_fraction_of_selected": len(future_ids) / len(selected) if selected else 0.0,
            "area_fraction_of_selected": future_area / selected_area if selected_area else 0.0,
            "recall_of_retrospective_objects": len(future_ids) / totals["objects"],
            "recall_of_retrospective_area": future_area / totals["area_m2"],
            "retained_retrospective_v75_m3": retained_v75,
            "retained_retrospective_v75_wan_m3": retained_v75 / 10000.0,
            "recall_of_retrospective_v75": retained_v75 / totals["v75_m3"],
        },
        "without_future_support": {
            "objects": len(selected_ids - retrospective_ids),
            "area_m2": selected_area - future_area,
            "area_km2": (selected_area - future_area) / 1e6,
        },
    }


def main() -> int:
    evidence_summary = json.loads((EVIDENCE_RESULT / "summary.json").read_text(encoding="utf-8"))
    volume_summary = json.loads((VOLUME_RESULT / "summary.json").read_text(encoding="utf-8"))
    if evidence_summary.get("status") != "complete" or volume_summary.get("status") != "complete":
        raise RuntimeError("Required source experiment is incomplete")
    evidence_rows = read_csv(EVIDENCE_RESULT / "rejected_objects_temporal_evidence.csv")
    evidence = {int(row["object_id"]): row for row in evidence_rows}

    with rasterio.open(AREA_GATE) as source:
        area_gate = source.read(1) > 0
        transform = source.transform
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    if count != int(evidence_summary["checks"]["area_gate_objects"]):
        raise RuntimeError("Area-gate object count changed")
    geometries = {
        int(value): shape(geometry)
        for geometry, value in shapes(
            labels.astype("int32"), mask=area_gate, transform=transform, connectivity=4
        )
        if int(value) > 0
    }

    previous_rows = []
    for object_id, raw in evidence.items():
        if not truth(raw["previous_fraction_ge_025"]):
            continue
        geom = geometries[object_id]
        area = float(raw["area_m2"])
        minx, miny, maxx, maxy = geom.bounds
        width, height = maxx - minx, maxy - miny
        short_side = min(width, height)
        previous_rows.append({
            "object_id": object_id,
            "area_m2": area,
            "previous_near_fraction": float(raw["previous_near"]),
            "future_near_fraction": float(raw["future_near"]),
            "future_fraction_ge_025": truth(raw["future_fraction_ge_025_retrospective"]),
            "target_scl_fraction": float(raw["scl_target_2024_06"]),
            "bbox_fill": area / (width * height) if width > 0 and height > 0 else 0.0,
            "elongation": max(width, height) / short_side if short_side > 0 else float("inf"),
            "compactness": (
                4.0 * math.pi * area / (geom.length * geom.length) if geom.length > 0 else 0.0
            ),
            "center_x": float(raw["center_x"]),
            "center_y": float(raw["center_y"]),
        })

    volume_rows = read_csv(VOLUME_RESULT / "retrospective_objects_volume_features.csv")
    v75 = {int(row["object_id"]): float(row["v75_m3"]) for row in volume_rows}
    retrospective_ids = set(v75)
    totals = {
        "objects": len(retrospective_ids),
        "area_m2": sum(float(evidence[i]["area_m2"]) for i in retrospective_ids),
        "v75_m3": sum(v75.values()),
    }

    branches = {
        "compact_02_5ha": [
            row for row in previous_rows
            if 2000 <= row["area_m2"] < 50000
            and row["bbox_fill"] >= 0.45 and row["elongation"] <= 3.0
        ],
        "compact_02_5ha_strict": [
            row for row in previous_rows
            if 2000 <= row["area_m2"] < 50000
            and row["bbox_fill"] >= 0.55 and row["elongation"] <= 2.5
        ],
    }

    all_ids = {row["object_id"] for row in previous_rows}
    if not retrospective_ids <= all_ids:
        raise RuntimeError("Retrospective group is not a subset of previous-strong objects")
    branch_summary = {
        name: summarize(rows, retrospective_ids, v75, totals)
        for name, rows in branches.items()
    }

    membership = []
    branch_sets = {name: {row["object_id"] for row in rows} for name, rows in branches.items()}
    for row in previous_rows:
        membership.append({
            **row,
            "selected_compact_02_5ha": row["object_id"] in branch_sets["compact_02_5ha"],
            "selected_compact_02_5ha_strict": row["object_id"] in branch_sets["compact_02_5ha_strict"],
            "retrospective_future_persistent": row["object_id"] in retrospective_ids,
            "retrospective_v75_m3": v75.get(row["object_id"], ""),
        })

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "previous_strong_shape_candidates.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(membership[0]))
        writer.writeheader()
        writer.writerows(membership)
    report = {
        "status": "complete",
        "experiment": "Operational previous-date plus morphology candidate audit",
        "region": "pihe",
        "previous_strong_scope": {
            "objects": len(previous_rows),
            "area_m2": sum(row["area_m2"] for row in previous_rows),
        },
        "retrospective_evaluation_reference": totals,
        "branches": branch_summary,
        "output_table": str(table),
        "interpretation": [
            "Selection uses only previous-date persistence, area, bbox fill, and elongation.",
            "Future persistence is an evaluation proxy, not a selection input or field truth.",
            "The two morphology variants are exploratory sensitivity bounds, not final thresholds.",
        ],
        "next_step": (
            "Review selected objects without future support and future-persistent objects rejected by shape "
            "before any DEM propagation."
        ),
        "limits": [
            "Future absence can represent real drying and is not proof of a false positive.",
            "Future persistence can repeat systematic errors and is not proof of a true pond.",
            "Object shape does not establish manual survey membership.",
            "No mask, DEM volume, production parameter, or formal output is changed.",
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
