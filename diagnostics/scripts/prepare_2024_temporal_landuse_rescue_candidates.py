#!/usr/bin/env python3
"""Materialize three overlapping 2024 temporal-evidence rescue scopes.

This step reconstructs candidate geometries only. It does not restore the
production mask and does not calculate DEM volumes. Future-date evidence is
kept in a retrospective branch and must not be used as a 2024 model input.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import fiona
import numpy as np
import rasterio
from fiona.crs import CRS as FionaCRS
from pyproj import CRS
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]

DATE = "2024-06-01"
AUDIT = ROOT / "diagnostics/results/landuse_temporal_evidence_2024/20261010_173122_234024"
REVIEW = ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024/20261010_175140_447294"
AREA_GATE = (
    ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
    / "paired_reference_downstream" / DATE
    / "shared_reference_after_area_gate.tif"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/temporal_landuse_rescue_candidates_2024"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def summarize(rows: list[dict]) -> dict:
    area_m2 = sum(float(row["area_m2"]) for row in rows)
    return {"object_count": len(rows), "area_m2": area_m2, "area_km2": area_m2 / 1e6}


def write_gpkg(path: Path, rows: list[dict], objects: dict, crs, branch: str) -> None:
    schema = {
        "geometry": "Unknown",
        "properties": {"object_id": "int", "branch": "str:48", "area_m2": "float"},
    }
    with fiona.open(
        path, "w", driver="GPKG", layer=path.stem,
        crs=FionaCRS.from_user_input(CRS.from_user_input(crs).to_string()), schema=schema,
    ) as sink:
        for row in rows:
            object_id = int(row["object_id"])
            sink.write({
                "geometry": mapping(objects[object_id]),
                "properties": {
                    "object_id": object_id,
                    "branch": branch,
                    "area_m2": float(row["area_m2"]),
                },
            })


def main() -> int:
    audit_summary = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(AUDIT / "rejected_objects_temporal_evidence.csv")
    labels = read_csv(REVIEW / "assistant_visual_labels.csv")
    labelled_ids = {int(row["object_id"]) for row in labels}
    if len(labelled_ids) != 15:
        raise RuntimeError("Expected 15 unique visual-review labels")

    branches = {
        "operational_previous_and_target_scl": [
            row for row in rows
            if truth(row["previous_fraction_ge_025"])
            and truth(row["target_scl_fraction_ge_025"])
        ],
        "retrospective_previous_and_future": [
            row for row in rows
            if truth(row["previous_fraction_ge_025"])
            and truth(row["future_fraction_ge_025_retrospective"])
        ],
        "triple_strong": [
            row for row in rows
            if truth(row["previous_fraction_ge_025"])
            and truth(row["target_scl_fraction_ge_025"])
            and truth(row["future_fraction_ge_025_retrospective"])
        ],
    }

    # Temporal-audit IDs were assigned on this rasterized area-gate mask.
    # Rebuild geometry from the same label grid so IDs and pixel areas remain
    # one-to-one, including where earlier polygons touched after rasterization.
    with rasterio.open(AREA_GATE) as source:
        area_gate = source.read(1) > 0
        transform = source.transform
        crs = source.crs
        pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    labels_grid, object_count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    by_id = {
        int(value): shape(geometry)
        for geometry, value in shapes(
            labels_grid.astype("int32"),
            mask=area_gate,
            transform=transform,
            connectivity=4,
        )
        if int(value) > 0
    }
    expected = audit_summary["checks"]
    total_area_m2 = float(area_gate.sum()) * pixel_area
    if object_count != int(expected["area_gate_objects"]):
        raise RuntimeError(
            f"Area-gate object count mismatch: {object_count} vs {expected['area_gate_objects']}"
        )
    if abs(total_area_m2 - float(expected["area_gate_area_km2"]) * 1e6) > 0.001:
        raise RuntimeError("Area-gate object area mismatch")
    if len(by_id) != object_count:
        raise RuntimeError("Area-gate labels did not polygonize one-to-one")

    selected_ids = {int(row["object_id"]) for values in branches.values() for row in values}
    missing = selected_ids - set(by_id)
    if missing:
        raise RuntimeError(f"Area-gate reconstruction is missing {len(missing)} selected IDs")
    for values in branches.values():
        for row in values:
            object_id = int(row["object_id"])
            if abs(float(row["area_m2"]) - float(by_id[object_id].area)) > 0.001:
                raise RuntimeError(f"Candidate area mismatch for object {object_id}")

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    branch_report = {}
    membership_rows = []
    for name, values in branches.items():
        gpkg = output / f"{name}.gpkg"
        write_gpkg(gpkg, values, by_id, crs, name)
        branch_report[name] = {
            **summarize(values),
            "gpkg": str(gpkg),
            "reviewed_object_ids_in_branch": sorted(
                {int(row["object_id"]) for row in values} & labelled_ids
            ),
        }
        for row in values:
            membership_rows.append({
                "branch": name,
                "object_id": int(row["object_id"]),
                "area_m2": float(row["area_m2"]),
                "previous_near_fraction": float(row["previous_near"]),
                "target_scl_fraction": float(row["scl_target_2024_06"]),
                "future_near_fraction": float(row["future_near"]),
            })

    with (output / "candidate_membership.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(membership_rows[0]))
        writer.writeheader()
        writer.writerows(membership_rows)

    sets = {name: {int(row["object_id"]) for row in values} for name, values in branches.items()}
    overlap = {
        "triple_is_subset_of_operational": sets["triple_strong"]
        <= sets["operational_previous_and_target_scl"],
        "triple_is_subset_of_retrospective": sets["triple_strong"]
        <= sets["retrospective_previous_and_future"],
        "operational_retrospective_overlap_objects": len(
            sets["operational_previous_and_target_scl"]
            & sets["retrospective_previous_and_future"]
        ),
    }
    if not overlap["triple_is_subset_of_operational"] or not overlap[
        "triple_is_subset_of_retrospective"
    ]:
        raise RuntimeError("Triple-strong subset check failed")

    report = {
        "status": "complete",
        "experiment": "2024 temporal-evidence land-use rescue geometry materialization",
        "date": DATE,
        "region": "pihe",
        "area_gate_reproduction": {"object_count": object_count, "area_m2": total_area_m2},
        "branches": branch_report,
        "overlap_checks": overlap,
        "next_step": "Calculate DEM volumes for each branch separately; never add overlaps.",
        "limits": [
            "The 0.25 evidence fraction is a coarse diagnostic bin, not a calibrated threshold.",
            "The retrospective branch uses 2025 evidence and cannot be an operational 2024 rule.",
            "Temporal model persistence can repeat stable false positives.",
            "Real water can still lie outside the manual pond-survey target definition.",
            "Production masks, parameters, model code, and formal outputs are unchanged.",
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
