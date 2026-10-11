#!/usr/bin/env python3
"""Compute exact V75 capacity for the operational loose temporal-shape candidate.

Selection uses previous-date support and morphology only. Existing DEM results
are reused for selected future-persistent objects; only selected objects without
future persistence are newly processed. Future evidence never changes selection.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import fiona
import numpy as np
import rasterio
import yaml
from fiona.crs import CRS as FionaCRS
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import pond_storage_monitor as monitor  # noqa: E402
import run_a1_dem_volume as volume  # noqa: E402

DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
CANDIDATE_RESULT = (
    ROOT / "diagnostics/results/operational_temporal_shape_candidate/20261010_184856_285062"
)
RETROSPECTIVE_FEATURES = (
    ROOT / "diagnostics/results/retrospective_rescue_volume_concentration"
    / "20261010_183252_458888/retrospective_objects_volume_features.csv"
)
BASE_RESULT = ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
AREA_GATE = (
    ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
    / "paired_reference_downstream/2024-06-01/shared_reference_after_area_gate.tif"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/operational_temporal_shape_dem_volume"
MANUAL = {DATE_24: 17058.0, DATE_25: 9174.0}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def evaluate(model_24_wan_m3: float, model_25_wan_m3: float) -> dict:
    model = {DATE_24: model_24_wan_m3, DATE_25: model_25_wan_m3}
    errors = {date: abs(model[date] - MANUAL[date]) for date in MANUAL}
    ratio = model[DATE_25] / model[DATE_24]
    manual_ratio = MANUAL[DATE_25] / MANUAL[DATE_24]
    return {
        "model_wan_m3": model,
        "manual_wan_m3": MANUAL,
        "absolute_error_wan_m3": errors,
        "total_absolute_error_wan_m3": sum(errors.values()),
        "mape_percent": 50.0 * sum(errors[d] / MANUAL[d] for d in MANUAL),
        "model_2025_to_2024_ratio": ratio,
        "manual_2025_to_2024_ratio": manual_ratio,
        "absolute_log_ratio_error": abs(math.log(ratio / manual_ratio)),
        "direction_matches_manual": (ratio < 1) == (manual_ratio < 1),
    }


def main() -> int:
    candidate = json.loads((CANDIDATE_RESULT / "summary.json").read_text(encoding="utf-8"))
    base = json.loads((BASE_RESULT / "summary.json").read_text(encoding="utf-8"))
    if candidate.get("status") != "complete" or base.get("status") != "complete":
        raise RuntimeError("Required source experiment is incomplete")

    candidate_rows = read_csv(CANDIDATE_RESULT / "previous_strong_shape_candidates.csv")
    selected_rows = [row for row in candidate_rows if truth(row["selected_compact_02_5ha"])]
    selected_ids = {int(row["object_id"]) for row in selected_rows}
    persistent_ids = {
        int(row["object_id"]) for row in selected_rows
        if truth(row["retrospective_future_persistent"])
    }
    new_ids = selected_ids - persistent_ids
    expected = candidate["branches"]["compact_02_5ha"]["selected"]
    if len(selected_ids) != int(expected["objects"]):
        raise RuntimeError("Selected candidate count changed")

    known_rows = {
        int(row["object_id"]): row for row in read_csv(RETROSPECTIVE_FEATURES)
        if int(row["object_id"]) in persistent_ids
    }
    if set(known_rows) != persistent_ids:
        raise RuntimeError("Previously computed persistent volumes are incomplete")
    known_v75 = sum(float(row["v75_m3"]) for row in known_rows.values())
    expected_known = candidate["branches"]["compact_02_5ha"]["future_persistent_proxy"]
    if abs(known_v75 - float(expected_known["retained_retrospective_v75_m3"])) > 0.001:
        raise RuntimeError("Reused V75 total changed")

    with rasterio.open(AREA_GATE) as source:
        area_gate = source.read(1) > 0
        transform = source.transform
        crs = source.crs
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    if count != 28078:
        raise RuntimeError("Area-gate object count changed")
    geometries = {
        int(value): shape(geometry)
        for geometry, value in shapes(
            labels.astype("int32"), mask=area_gate, transform=transform, connectivity=4
        )
        if int(value) in new_ids
    }
    if set(geometries) != new_ids:
        raise RuntimeError("New candidate geometry reconstruction is incomplete")
    area_by_id = {int(row["object_id"]): float(row["area_m2"]) for row in selected_rows}
    objects = []
    for object_id in sorted(new_ids):
        obj = monitor.WaterObject(
            obj_id=object_id, geom=geometries[object_id], area_m2=area_by_id[object_id]
        )
        obj.props.update({
            "area_m2": area_by_id[object_id],
            "area_mu": area_by_id[object_id] / 666.6666667,
            "rescue_branch": "compact_02_5ha_without_future_support",
        })
        objects.append(obj)

    cfg_path = ROOT / "output/20240601/final/pihe/resolved_monitor_cfg_2024-06-01.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("dem_tif", "aoi_shp", "landuse_water_tif"):
        if cfg.get(key):
            cfg[key] = volume.remap_path(cfg[key])

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    partial = {
        "status": "running",
        "experiment": "Operational temporal-shape candidate exact V75",
        "selected_objects": len(selected_ids),
        "reused_persistent_objects": len(persistent_ids),
        "newly_processed_objects": len(objects),
    }
    (output / "summary.partial.json").write_text(
        json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Boundary levels for {len(objects)} selected objects without future support", flush=True)
    monitor.estimate_water_levels(
        objects,
        cfg["dem_tif"],
        water_crs=monitor.parse_crs(crs),
        sample_step_m=float(cfg.get("boundary_sample_step_m", 5.0)),
        max_boundary_samples=int(cfg.get("max_boundary_samples", 5000)),
    )
    print("DEM volumes for newly processed selected objects", flush=True)
    monitor.compute_volumes(
        objects,
        cfg["dem_tif"],
        water_crs=monitor.parse_crs(crs),
        block_size=int(cfg.get("block_size", 512)),
        all_touched=bool(cfg.get("all_touched", False)),
    )
    new_metrics = volume.compute_metrics(objects)
    total_v75 = known_v75 + float(new_metrics["v75_m3"])
    total_missing = (
        sum(1 for row in known_rows.values() if (row.get("qa") or "") != "OK")
        + int(new_metrics["missing_v75"])
    )

    gpkg = output / "selected_without_future_support_with_volume.gpkg"
    table = output / "selected_without_future_support_with_volume.csv"
    monitor.write_objects_gpkg(
        objects, str(gpkg), crs=FionaCRS.from_user_input(crs.to_string())
    )
    monitor.write_objects_csv(objects, str(table))

    baseline_24 = float(base["paired_reference_2024"]["volume"]["v75_m3"]) / 10000.0
    candidate_25 = float(base["conservative_gate_2025"]["volume"]["v75_m3"]) / 10000.0
    current_eval = evaluate(baseline_24, candidate_25)
    shape_eval = evaluate(baseline_24 + total_v75 / 10000.0, candidate_25)
    shape_eval["total_absolute_error_reduction_vs_current_wan_m3"] = (
        current_eval["total_absolute_error_wan_m3"] - shape_eval["total_absolute_error_wan_m3"]
    )
    shape_eval["trend_log_error_reduction_vs_current"] = (
        current_eval["absolute_log_ratio_error"] - shape_eval["absolute_log_ratio_error"]
    )

    report = {
        "status": "complete",
        "experiment": "Operational temporal-shape candidate exact V75",
        "region": "pihe",
        "selection": (
            "Previous-date model coverage >=0.25, area 0.2-5 ha, bbox fill >=0.45, "
            "and elongation <=3; no future evidence is used for selection."
        ),
        "candidate": {
            "object_count": len(selected_ids),
            "area_m2": float(expected["area_m2"]),
            "area_km2": float(expected["area_km2"]),
            "reused_persistent_objects": len(persistent_ids),
            "newly_processed_without_future_objects": len(objects),
            "reused_persistent_v75_m3": known_v75,
            "new_without_future_metrics": new_metrics,
            "total_v75_m3": total_v75,
            "total_v75_wan_m3": total_v75 / 10000.0,
            "total_missing_v75": total_missing,
        },
        "evaluation": {
            "current_paired_candidate": current_eval,
            "baseline_plus_operational_shape_candidate": shape_eval,
        },
        "outputs": {"new_objects_gpkg": str(gpkg), "new_objects_csv": str(table)},
        "limits": [
            "Thresholds are exploratory sensitivity values and are not frozen production parameters.",
            "Historical model persistence can repeat stable false positives.",
            "The held-out review found both nonwater risk and real-water losses at the area boundary.",
            "Manual survey spatial-scope equivalence remains unresolved.",
            "Only Pihe is tested; Shihe and Hangbuhe remain untested.",
            "Production code, masks, and formal outputs are unchanged.",
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
