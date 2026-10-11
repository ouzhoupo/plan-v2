#!/usr/bin/env python3
"""Compute DEM capacity for the operational 2024 temporal rescue scope.

The operational branch uses only previous-date model water and target-date
SCL evidence. The triple-strong subset is summarized from the same DEM run.
No production mask or model parameter is changed.
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
BASE_RESULT = (
    ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
)
REVIEW = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024"
    / "20261010_175140_447294"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/operational_temporal_rescue_dem_volume_2024"
MANUAL = {DATE_24: 17058.0, DATE_25: 9174.0}


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
            obj.props.update({
                "area_m2": area,
                "area_mu": area / 666.6666667,
                "rescue_branch": str(props.get("branch") or ""),
            })
            objects.append(obj)
    if water_crs is None:
        raise RuntimeError(f"Missing object CRS: {path}")
    return objects, water_crs


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


def proxy_volume(objects_by_id: dict[int, object], labels: list[dict], allowed: set[int]) -> dict:
    result = {}
    for row in labels:
        object_id = int(row["object_id"])
        if object_id not in allowed:
            continue
        obj = objects_by_id[object_id]
        label = row["proxy_label"]
        item = result.setdefault(label, {"objects": 0, "area_m2": 0.0, "v75_m3": 0.0})
        item["objects"] += 1
        item["area_m2"] += float(obj.area_m2)
        item["v75_m3"] += float(obj.props.get("v75_m3") or 0.0)
    for item in result.values():
        item["v75_wan_m3"] = item["v75_m3"] / 10000.0
    return result


def main() -> int:
    materialized = json.loads((MATERIALIZED / "summary.json").read_text(encoding="utf-8"))
    base = json.loads((BASE_RESULT / "summary.json").read_text(encoding="utf-8"))
    if materialized.get("status") != "complete" or base.get("status") != "complete":
        raise RuntimeError("Required source experiment is incomplete")

    operational_name = "operational_previous_and_target_scl"
    triple_name = "triple_strong"
    objects, water_crs = load_objects(MATERIALIZED / f"{operational_name}.gpkg")
    expected_operational = materialized["branches"][operational_name]
    if len(objects) != int(expected_operational["object_count"]):
        raise RuntimeError("Operational object count changed")
    if abs(sum(obj.area_m2 for obj in objects) - float(expected_operational["area_m2"])) > 0.001:
        raise RuntimeError("Operational object area changed")

    memberships = read_csv(MATERIALIZED / "candidate_membership.csv")
    operational_ids = {
        int(row["object_id"]) for row in memberships if row["branch"] == operational_name
    }
    triple_ids = {
        int(row["object_id"]) for row in memberships if row["branch"] == triple_name
    }
    by_id = {obj.obj_id: obj for obj in objects}
    if operational_ids != set(by_id):
        raise RuntimeError("Operational membership does not match geometry")
    if not triple_ids <= operational_ids:
        raise RuntimeError("Triple-strong branch is not an operational subset")

    cfg_path = ROOT / "output/20240601/final/pihe/resolved_monitor_cfg_2024-06-01.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("dem_tif", "aoi_shp", "landuse_water_tif"):
        if cfg.get(key):
            cfg[key] = volume.remap_path(cfg[key])
    threshold = cfg.get("skip_volume_above_area_m2")
    if threshold is not None:
        for obj in objects:
            if obj.area_m2 > float(threshold):
                obj.props["skip_volume"] = True
                obj.props["qa"] = "TOO_LARGE_SKIP_VOLUME"

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "experiment": "2024 operational temporal land-use rescue DEM capacity",
        "region": "pihe",
        "source_materialized": str(MATERIALIZED),
    }
    (output / "summary.partial.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Boundary levels for {len(objects)} operational candidates", flush=True)
    monitor.estimate_water_levels(
        objects,
        cfg["dem_tif"],
        water_crs=water_crs,
        sample_step_m=float(cfg.get("boundary_sample_step_m", 5.0)),
        max_boundary_samples=int(cfg.get("max_boundary_samples", 5000)),
    )
    print("DEM volumes for operational candidates", flush=True)
    monitor.compute_volumes(
        objects,
        cfg["dem_tif"],
        water_crs=water_crs,
        block_size=int(cfg.get("block_size", 512)),
        all_touched=bool(cfg.get("all_touched", False)),
    )

    operational_objects = [by_id[i] for i in sorted(operational_ids)]
    triple_objects = [by_id[i] for i in sorted(triple_ids)]
    operational_metrics = volume.compute_metrics(operational_objects)
    triple_metrics = volume.compute_metrics(triple_objects)
    if operational_metrics["object_count"] != int(expected_operational["object_count"]):
        raise RuntimeError("Operational DEM metrics lost objects")
    if triple_metrics["object_count"] != int(
        materialized["branches"][triple_name]["object_count"]
    ):
        raise RuntimeError("Triple-strong DEM metrics lost objects")

    gpkg = output / "operational_candidates_with_volume.gpkg"
    table = output / "operational_candidates_with_volume.csv"
    monitor.write_objects_gpkg(
        operational_objects, str(gpkg),
        crs=FionaCRS.from_user_input(water_crs.to_string()),
    )
    monitor.write_objects_csv(operational_objects, str(table))

    baseline_24 = float(base["paired_reference_2024"]["volume"]["v75_m3"]) / 10000.0
    candidate_25 = float(base["conservative_gate_2025"]["volume"]["v75_m3"]) / 10000.0
    variants = {
        "current_paired_candidate": baseline_24,
        "baseline_plus_triple_strong": baseline_24 + triple_metrics["v75_m3"] / 10000.0,
        "baseline_plus_operational": baseline_24 + operational_metrics["v75_m3"] / 10000.0,
    }
    evaluations = {name: evaluate(value, candidate_25) for name, value in variants.items()}
    baseline_eval = evaluations["current_paired_candidate"]
    for name, item in evaluations.items():
        item["total_absolute_error_reduction_vs_current_wan_m3"] = (
            baseline_eval["total_absolute_error_wan_m3"]
            - item["total_absolute_error_wan_m3"]
        )
        item["trend_log_error_reduction_vs_current"] = (
            baseline_eval["absolute_log_ratio_error"]
            - item["absolute_log_ratio_error"]
        )

    labels = read_csv(REVIEW / "assistant_visual_labels.csv")
    report.update({
        "status": "complete",
        "branches": {
            triple_name: {
                "metrics": triple_metrics,
                "proxy_review_volume": proxy_volume(by_id, labels, triple_ids),
            },
            operational_name: {
                "metrics": operational_metrics,
                "exclusive_of_triple": {
                    "object_count": len(operational_ids - triple_ids),
                    "area_m2": sum(by_id[i].area_m2 for i in operational_ids - triple_ids),
                    "v75_m3": sum(
                        float(by_id[i].props.get("v75_m3") or 0.0)
                        for i in operational_ids - triple_ids
                    ),
                },
                "proxy_review_volume": proxy_volume(by_id, labels, operational_ids),
            },
        },
        "variant_evaluation": evaluations,
        "outputs": {"gpkg": str(gpkg), "csv": str(table)},
        "next_decision": (
            "Use exact operational capacity and side effects to decide whether the much larger "
            "retrospective branch is worth a separate diagnostic DEM run."
        ),
        "limits": [
            "The 0.25 evidence fraction is a coarse diagnostic bin, not a fitted threshold.",
            "Previous-date model water and target-date SCL share optical/systematic errors and are not field truth.",
            "The three reviewed objects cannot estimate accuracy for all 1417 candidates.",
            "Manual survey spatial-scope equivalence remains unresolved.",
            "Only Pihe is tested; Shihe and Hangbuhe remain untested.",
            "Production code, production masks, and formal model outputs are unchanged.",
        ],
    })
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print("STATUS: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
