#!/usr/bin/env python3
"""Compute exact V75 capacity for the Hangbuhe temporal-shape candidate.

The candidate was selected by the frozen cross-region bridge using only
2024-05-21 historical support and 2024-06-01 morphology.  Future evidence is
not used here.  This diagnostic does not modify production outputs.
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


REGION = "hangbuhe"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
BRIDGE = (
    ROOT / "diagnostics/results/temporal_shape_cross_region_bridge"
    / "20261010_194531_518591"
)
BASELINE_AUDIT = (
    ROOT / "diagnostics/results/cross_region_baseline_trends"
    / "20261010_202616_329605/summary.json"
)
MANUAL_AUDIT = (
    ROOT / "diagnostics/results/cross_region_manual_totals"
    / "20261010_201437_787088/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/hangbuhe_temporal_shape_dem_volume"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def evaluate(model_24: float, model_25: float, manual_24: float, manual_25: float) -> dict:
    model_ratio = model_25 / model_24
    manual_ratio = manual_25 / manual_24
    errors = (abs(model_24 - manual_24), abs(model_25 - manual_25))
    return {
        "model_wan_m3": {DATE_24: model_24, DATE_25: model_25},
        "manual_wan_m3": {DATE_24: manual_24, DATE_25: manual_25},
        "model_ratio_2025_to_2024": model_ratio,
        "manual_ratio_2025_to_2024": manual_ratio,
        "model_direction": "decrease" if model_ratio < 1 else "increase" if model_ratio > 1 else "flat",
        "manual_direction": "decrease" if manual_ratio < 1 else "increase" if manual_ratio > 1 else "flat",
        "direction_matches_manual": (model_ratio < 1) == (manual_ratio < 1),
        "trend_log_error": abs(math.log(model_ratio / manual_ratio)),
        "absolute_error_wan_m3": {DATE_24: errors[0], DATE_25: errors[1]},
        "total_absolute_error_wan_m3": sum(errors),
        "mape_percent": 50.0 * (errors[0] / manual_24 + errors[1] / manual_25),
    }


def main() -> int:
    bridge = json.loads((BRIDGE / "summary.json").read_text(encoding="utf-8"))
    baseline_audit = json.loads(BASELINE_AUDIT.read_text(encoding="utf-8"))
    manual_audit = json.loads(MANUAL_AUDIT.read_text(encoding="utf-8"))
    if any(item.get("status") != "complete" for item in (bridge, baseline_audit, manual_audit)):
        raise RuntimeError("Required source experiment is incomplete")

    region_dir = BRIDGE / REGION
    table = region_dir / "landuse_rejected_temporal_shape_bridge.csv"
    rows = read_csv(table)
    selected = [row for row in rows if truth(row["selected_loose_candidate"])]
    selected_ids = {int(row["object_id"]) for row in selected}
    expected = bridge["regions"][REGION]["loose_candidate"]
    if len(selected_ids) != int(expected["objects"]):
        raise RuntimeError("Selected candidate count changed")
    selected_area = sum(float(row["area_m2"]) for row in selected)
    if abs(selected_area - float(expected["area_m2"])) > 0.001:
        raise RuntimeError("Selected candidate area changed")

    area_gate_path = region_dir / "original_v75_bridge_after_area_gate.tif"
    with rasterio.open(area_gate_path) as source:
        area_gate = source.read(1) > 0
        transform = source.transform
        crs = source.crs
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    expected_count = int(bridge["regions"][REGION]["area_gate"]["object_count"])
    if count != expected_count:
        raise RuntimeError(f"Area-gate object count changed: {count} != {expected_count}")
    geometries = {
        int(value): shape(geometry)
        for geometry, value in shapes(
            labels.astype("int32"), mask=area_gate, transform=transform, connectivity=4
        )
        if int(value) in selected_ids
    }
    if set(geometries) != selected_ids:
        missing = selected_ids - set(geometries)
        raise RuntimeError(f"Candidate geometry reconstruction missing {len(missing)} objects")

    area_by_id = {int(row["object_id"]): float(row["area_m2"]) for row in selected}
    objects = []
    for object_id in sorted(selected_ids):
        obj = monitor.WaterObject(
            obj_id=object_id,
            geom=geometries[object_id],
            area_m2=area_by_id[object_id],
        )
        obj.props.update({
            "area_m2": area_by_id[object_id],
            "area_mu": area_by_id[object_id] / 666.6666667,
            "rescue_branch": "historical_support_plus_loose_shape",
        })
        objects.append(obj)

    cfg_path = ROOT / f"output/20240601/final/{REGION}/resolved_monitor_cfg_{DATE_24}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("dem_tif", "aoi_shp", "landuse_water_tif"):
        if cfg.get(key):
            cfg[key] = volume.remap_path(cfg[key])

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    partial = {
        "status": "running",
        "experiment": "Hangbuhe temporal-shape exact V75",
        "selected_objects": len(objects),
        "selected_area_m2": selected_area,
    }
    (output / "summary.partial.json").write_text(
        json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Boundary levels for {len(objects)} Hangbuhe candidates", flush=True)
    monitor.estimate_water_levels(
        objects,
        cfg["dem_tif"],
        water_crs=monitor.parse_crs(crs),
        sample_step_m=float(cfg.get("boundary_sample_step_m", 5.0)),
        max_boundary_samples=int(cfg.get("max_boundary_samples", 5000)),
    )
    print("DEM volumes for Hangbuhe candidates", flush=True)
    monitor.compute_volumes(
        objects,
        cfg["dem_tif"],
        water_crs=monitor.parse_crs(crs),
        block_size=int(cfg.get("block_size", 512)),
        all_touched=bool(cfg.get("all_touched", False)),
    )
    candidate_metrics = volume.compute_metrics(objects)
    increment = float(candidate_metrics["v75_m3"]) / 10000.0

    gpkg = output / "hangbuhe_candidates_with_volume.gpkg"
    object_table = output / "hangbuhe_candidates_with_volume.csv"
    monitor.write_objects_gpkg(
        objects, str(gpkg), crs=FionaCRS.from_user_input(crs.to_string())
    )
    monitor.write_objects_csv(objects, str(object_table))

    model = baseline_audit["regional"][REGION]["model"]
    model_24 = float(model[DATE_24]["v75_wan_m3"])
    model_25 = float(model[DATE_25]["v75_wan_m3"])
    evaluations = {}
    for scenario in ("source_total", "component_sum"):
        manual = manual_audit["regional_trends"][REGION][scenario]
        before = evaluate(model_24, model_25, float(manual[DATE_24]), float(manual[DATE_25]))
        after = evaluate(model_24 + increment, model_25, float(manual[DATE_24]), float(manual[DATE_25]))
        after["trend_log_error_reduction"] = before["trend_log_error"] - after["trend_log_error"]
        after["total_absolute_error_reduction_wan_m3"] = (
            before["total_absolute_error_wan_m3"] - after["total_absolute_error_wan_m3"]
        )
        evaluations[scenario] = {"before": before, "after": after}

    budget = baseline_audit["regional"][REGION]["source_total_scenario"]
    exact_increment = float(budget["exact_match_increment_wan_m3"])
    flip_increment = float(budget["direction_flip_increment_wan_m3"])
    if increment < exact_increment:
        guardrail = "improves_but_under_corrects_trend"
    elif increment < flip_increment:
        guardrail = "over_corrects_magnitude_but_keeps_manual_direction"
    else:
        guardrail = "exceeds_direction_flip_boundary"

    report = {
        "status": "complete",
        "experiment": "Hangbuhe temporal-shape candidate exact V75",
        "region": REGION,
        "selection": bridge["selection"],
        "candidate": {
            "object_count": len(objects),
            "area_m2": selected_area,
            "area_km2": selected_area / 1e6,
            "volume": candidate_metrics,
            "increment_v75_wan_m3": increment,
        },
        "trend_budget": {
            "exact_match_increment_wan_m3": exact_increment,
            "direction_flip_increment_wan_m3": flip_increment,
            "candidate_fraction_of_exact_match": increment / exact_increment,
            "candidate_fraction_of_direction_flip": increment / flip_increment,
            "guardrail_result": guardrail,
        },
        "evaluation": evaluations,
        "outputs": {"gpkg": str(gpkg), "csv": str(object_table)},
        "interpretation": [
            "Candidate selection uses only information available by the 2024 target date.",
            "Trend improvement, magnitude overshoot, and direction reversal are evaluated separately.",
            "The manual target is used only for evaluation and never to select objects or tune thresholds.",
        ],
        "limits": [
            "Manual totals are regional aggregates rather than object labels.",
            "Historical model support may preserve stable false positives.",
            "The candidate is an experimental full branch and not a production correction.",
            "No production code, masks, parameters, or formal outputs are changed.",
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
