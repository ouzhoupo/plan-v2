#!/usr/bin/env python3
"""Compute the retrospective upper-bound DEM capacity for 2024 land-use rejects.

Future 2025 model support is used only to test whether this error stage contains
enough capacity to explain the trend mismatch. It is not an operational rule.
Triple-strong volumes are reused from the completed operational DEM run; only
the retrospective-exclusive objects are newly processed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import fiona
import yaml
from fiona.crs import CRS as FionaCRS
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import pond_storage_monitor as monitor  # noqa: E402
import run_a1_dem_volume as volume  # noqa: E402

DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
MATERIALIZED = (
    ROOT / "diagnostics/results/temporal_landuse_rescue_candidates_2024"
    / "20261010_181232_993446"
)
OPERATIONAL_DEM = (
    ROOT / "diagnostics/results/operational_temporal_rescue_dem_volume_2024"
    / "20261010_182317_438951"
)
BASE_RESULT = ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
REVIEW = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024"
    / "20261010_175140_447294"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/retrospective_temporal_rescue_dem_volume_2024"
MANUAL = {DATE_24: 17058.0, DATE_25: 9174.0}
VOLUME_FIELDS = ("v05_m3", "v50_m3", "v75_m3", "v95_m3", "vmean_m3")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_objects(path: Path):
    objects = []
    with fiona.open(path) as source:
        water_crs = monitor.parse_crs(source.crs_wkt or source.crs)
        for feature in source:
            props = dict(feature["properties"])
            geom = shape(feature["geometry"])
            area = float(props.get("area_m2") or geom.area)
            obj = monitor.WaterObject(
                obj_id=int(props["object_id"]), geom=geom, area_m2=area
            )
            obj.props.update({"area_m2": area, "area_mu": area / 666.6666667})
            objects.append(obj)
    if water_crs is None:
        raise RuntimeError(f"Missing object CRS: {path}")
    return objects, water_crs


def combine_metrics(first: dict, second: dict) -> dict:
    area = float(first["area_m2"]) + float(second["area_m2"])
    qa_counts = dict(first["qa_counts"])
    for key, value in second["qa_counts"].items():
        qa_counts[key] = qa_counts.get(key, 0) + int(value)
    result = {
        "object_count": int(first["object_count"]) + int(second["object_count"]),
        "area_m2": area,
        "area_km2": area / 1e6,
        "qa_counts": qa_counts,
        "missing_v75": int(first["missing_v75"]) + int(second["missing_v75"]),
    }
    for field in VOLUME_FIELDS:
        result[field] = float(first[field]) + float(second[field])
    result["equivalent_depth_v75_div_area"] = result["v75_m3"] / area if area else None
    return result


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
    materialized = json.loads((MATERIALIZED / "summary.json").read_text(encoding="utf-8"))
    operational = json.loads((OPERATIONAL_DEM / "summary.json").read_text(encoding="utf-8"))
    base = json.loads((BASE_RESULT / "summary.json").read_text(encoding="utf-8"))
    if any(item.get("status") != "complete" for item in (materialized, operational, base)):
        raise RuntimeError("Required source experiment is incomplete")

    retrospective_name = "retrospective_previous_and_future"
    triple_name = "triple_strong"
    objects, water_crs = load_objects(MATERIALIZED / f"{retrospective_name}.gpkg")
    by_id = {obj.obj_id: obj for obj in objects}
    memberships = read_csv(MATERIALIZED / "candidate_membership.csv")
    retrospective_ids = {
        int(row["object_id"]) for row in memberships if row["branch"] == retrospective_name
    }
    triple_ids = {
        int(row["object_id"]) for row in memberships if row["branch"] == triple_name
    }
    if retrospective_ids != set(by_id):
        raise RuntimeError("Retrospective membership does not match geometry")
    if not triple_ids <= retrospective_ids:
        raise RuntimeError("Triple branch is not a retrospective subset")
    exclusive_ids = retrospective_ids - triple_ids
    exclusive_objects = [by_id[i] for i in sorted(exclusive_ids)]

    expected = materialized["branches"][retrospective_name]
    if len(objects) != int(expected["object_count"]):
        raise RuntimeError("Retrospective object count changed")
    if abs(sum(obj.area_m2 for obj in objects) - float(expected["area_m2"])) > 0.001:
        raise RuntimeError("Retrospective object area changed")

    cfg_path = ROOT / "output/20240601/final/pihe/resolved_monitor_cfg_2024-06-01.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("dem_tif", "aoi_shp", "landuse_water_tif"):
        if cfg.get(key):
            cfg[key] = volume.remap_path(cfg[key])
    threshold = cfg.get("skip_volume_above_area_m2")
    if threshold is not None:
        for obj in exclusive_objects:
            if obj.area_m2 > float(threshold):
                obj.props["skip_volume"] = True
                obj.props["qa"] = "TOO_LARGE_SKIP_VOLUME"

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    partial = {
        "status": "running",
        "experiment": "2024 retrospective temporal rescue DEM capacity upper bound",
        "region": "pihe",
        "newly_processed_objects": len(exclusive_objects),
        "reused_triple_objects": len(triple_ids),
    }
    (output / "summary.partial.json").write_text(
        json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Boundary levels for {len(exclusive_objects)} retrospective-exclusive objects", flush=True)
    monitor.estimate_water_levels(
        exclusive_objects,
        cfg["dem_tif"],
        water_crs=water_crs,
        sample_step_m=float(cfg.get("boundary_sample_step_m", 5.0)),
        max_boundary_samples=int(cfg.get("max_boundary_samples", 5000)),
    )
    print("DEM volumes for retrospective-exclusive objects", flush=True)
    monitor.compute_volumes(
        exclusive_objects,
        cfg["dem_tif"],
        water_crs=water_crs,
        block_size=int(cfg.get("block_size", 512)),
        all_touched=bool(cfg.get("all_touched", False)),
    )

    exclusive_metrics = volume.compute_metrics(exclusive_objects)
    triple_metrics = operational["branches"][triple_name]["metrics"]
    retrospective_metrics = combine_metrics(triple_metrics, exclusive_metrics)
    if retrospective_metrics["object_count"] != int(expected["object_count"]):
        raise RuntimeError("Combined retrospective metrics lost objects")
    if abs(retrospective_metrics["area_m2"] - float(expected["area_m2"])) > 0.001:
        raise RuntimeError("Combined retrospective metrics changed area")

    gpkg = output / "retrospective_exclusive_objects_with_volume.gpkg"
    table = output / "retrospective_exclusive_objects_with_volume.csv"
    monitor.write_objects_gpkg(
        exclusive_objects, str(gpkg),
        crs=FionaCRS.from_user_input(water_crs.to_string()),
    )
    monitor.write_objects_csv(exclusive_objects, str(table))

    baseline_24 = float(base["paired_reference_2024"]["volume"]["v75_m3"]) / 10000.0
    candidate_25 = float(base["conservative_gate_2025"]["volume"]["v75_m3"]) / 10000.0
    current_eval = evaluate(baseline_24, candidate_25)
    retrospective_eval = evaluate(
        baseline_24 + retrospective_metrics["v75_m3"] / 10000.0,
        candidate_25,
    )
    retrospective_eval["total_absolute_error_reduction_vs_current_wan_m3"] = (
        current_eval["total_absolute_error_wan_m3"]
        - retrospective_eval["total_absolute_error_wan_m3"]
    )
    retrospective_eval["trend_log_error_reduction_vs_current"] = (
        current_eval["absolute_log_ratio_error"]
        - retrospective_eval["absolute_log_ratio_error"]
    )

    reviewed_labels = read_csv(REVIEW / "assistant_visual_labels.csv")
    reviewed_membership = [
        row for row in reviewed_labels if int(row["object_id"]) in retrospective_ids
    ]
    report = {
        "status": "complete",
        "experiment": "2024 retrospective temporal rescue DEM capacity upper bound",
        "region": "pihe",
        "source_materialized": str(MATERIALIZED),
        "source_reused_triple_dem": str(OPERATIONAL_DEM),
        "processing": {
            "newly_processed_retrospective_exclusive_objects": len(exclusive_objects),
            "reused_triple_objects": len(triple_ids),
            "branches_are_not_added_twice": True,
        },
        "triple_reused_metrics": triple_metrics,
        "retrospective_exclusive_metrics": exclusive_metrics,
        "retrospective_full_metrics": retrospective_metrics,
        "evaluation": {
            "current_paired_candidate": current_eval,
            "baseline_plus_retrospective_upper_bound": retrospective_eval,
        },
        "reviewed_proxy_objects_in_scope": reviewed_membership,
        "outputs": {"exclusive_gpkg": str(gpkg), "exclusive_csv": str(table)},
        "limits": [
            "2025 support is unavailable in 2024 and this branch is prohibited as a production rule.",
            "The result tests stage capacity, not achievable operational accuracy.",
            "Temporal persistence can repeat systematic false positives and include non-target real water.",
            "Only six reviewed proxy objects fall in this 12681-object branch.",
            "Manual survey spatial-scope equivalence remains unresolved.",
            "Only Pihe is tested; no whole-Pishihang inference is made.",
            "Production code, masks, and formal outputs remain unchanged.",
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
