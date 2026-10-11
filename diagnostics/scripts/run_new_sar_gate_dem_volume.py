#!/usr/bin/env python3
"""Compute paired DEM volume and trend effect for the conservative SAR gate.

The unchanged 2024-06 shared-SAR/shared-alpha mask supplies the paired temporal
reference.  For 2025, both the shared baseline and conservative gate downstream
objects are processed with the same DEM functions.  Production outputs are not
modified.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import yaml

import run_a1_dem_volume as volume
from run_a1_downstream import prepare_context, run_variant


ROOT = volume.ROOT
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
DOWNSTREAM_25 = (
    ROOT / "diagnostics/results/new_sar_gate_downstream/20261010_144117_101667"
)
SHARED_UPSTREAM_24 = (
    ROOT / "diagnostics/results/shared_sar_calibration_upstream/20261009_193736_367365"
    / DATE_24 / "shared_sar_shared_alpha_c4_water_mask.tif"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/new_sar_gate_dem_volume"
SURVEY = volume.SURVEY_WAN_M3


def resolved_config(date: str) -> dict:
    path = (
        ROOT / "output" / date.replace("-", "") / "final" / "pihe"
        / f"resolved_monitor_cfg_{date}.yaml"
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dem_tif", "aoi_shp", "landuse_water_tif"):
        if cfg.get(key):
            cfg[key] = volume.remap_path(cfg[key])
    return cfg


def validate_metrics(name: str, metrics: dict, expected: dict) -> None:
    if metrics["object_count"] != expected["object_count"]:
        raise RuntimeError(
            f"{name} object count mismatch: {metrics['object_count']} != {expected['object_count']}"
        )
    if abs(metrics["area_m2"] - expected["area_m2"]) > 0.01:
        raise RuntimeError(
            f"{name} area mismatch: {metrics['area_m2']} != {expected['area_m2']}"
        )


def survey_metrics(model_24_m3: float, model_25_m3: float) -> dict:
    model = {DATE_24: model_24_m3 / 10000.0, DATE_25: model_25_m3 / 10000.0}
    errors = {date: abs(model[date] - SURVEY[date]) for date in (DATE_24, DATE_25)}
    ratio = model[DATE_25] / model[DATE_24]
    survey_ratio = SURVEY[DATE_25] / SURVEY[DATE_24]
    return {
        "model_wan_m3": model,
        "survey_wan_m3": {date: SURVEY[date] for date in (DATE_24, DATE_25)},
        "absolute_error_wan_m3": errors,
        "total_absolute_error_wan_m3": sum(errors.values()),
        "mape_percent": 50.0 * sum(errors[date] / SURVEY[date] for date in errors),
        "model_2025_to_2024_ratio": ratio,
        "survey_2025_to_2024_ratio": survey_ratio,
        "absolute_log_ratio_error": abs(math.log(ratio / survey_ratio)),
        "direction_matches_survey": (ratio < 1) == (survey_ratio < 1),
    }


def main() -> int:
    downstream_25 = json.loads((DOWNSTREAM_25 / "summary.json").read_text(encoding="utf-8"))
    if downstream_25.get("status") != "complete":
        raise RuntimeError("2025 conservative-gate downstream propagation is incomplete")
    if not SHARED_UPSTREAM_24.exists():
        raise FileNotFoundError(SHARED_UPSTREAM_24)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    support_downstream = output / "paired_reference_downstream"
    support_date = support_downstream / DATE_24
    support_date.mkdir(parents=True)

    cfg24, water_crs, aoi, process_bounds, layers = prepare_context(DATE_24)
    reference_downstream = run_variant(
        DATE_24, "shared_reference", SHARED_UPSTREAM_24, support_date,
        cfg24, water_crs, aoi, process_bounds, layers,
    )

    volume_output_24 = output / DATE_24
    volume_output_25 = output / DATE_25
    volume_output_24.mkdir()
    volume_output_25.mkdir()

    volume.DOWNSTREAM = support_downstream
    _, metrics_24, outputs_24 = volume.compute_variant(
        DATE_24, "shared_reference", volume_output_24, resolved_config(DATE_24)
    )
    validate_metrics("2024 shared reference", metrics_24, reference_downstream["after_landuse"])

    volume.DOWNSTREAM = DOWNSTREAM_25
    _, shared_25, shared_outputs = volume.compute_variant(
        DATE_25, "shared_baseline", volume_output_25, resolved_config(DATE_25)
    )
    _, gate_25, gate_outputs = volume.compute_variant(
        DATE_25, "conservative_gate", volume_output_25, resolved_config(DATE_25)
    )
    validate_metrics("2025 shared baseline", shared_25, downstream_25["paired_baseline"]["after_landuse"])
    validate_metrics("2025 conservative gate", gate_25, downstream_25["conservative_gate"]["after_landuse"])

    shared_eval = survey_metrics(metrics_24["v75_m3"], shared_25["v75_m3"])
    gate_eval = survey_metrics(metrics_24["v75_m3"], gate_25["v75_m3"])
    effect = {
        "object_count_change_2025": gate_25["object_count"] - shared_25["object_count"],
        "area_change_m2_2025": round(gate_25["area_m2"] - shared_25["area_m2"], 3),
        "v75_change_m3_2025": round(gate_25["v75_m3"] - shared_25["v75_m3"], 3),
        "v75_percent_change_2025": (
            100.0 * (gate_25["v75_m3"] - shared_25["v75_m3"]) / shared_25["v75_m3"]
            if shared_25["v75_m3"] else None
        ),
        "total_absolute_error_reduction_wan_m3": (
            shared_eval["total_absolute_error_wan_m3"]
            - gate_eval["total_absolute_error_wan_m3"]
        ),
        "trend_log_error_reduction": (
            shared_eval["absolute_log_ratio_error"]
            - gate_eval["absolute_log_ratio_error"]
        ),
    }
    report = {
        "status": "complete",
        "experiment": "Conservative new-SAR gate paired DEM and trend propagation",
        "region": "pihe",
        "source_2025_downstream": str(DOWNSTREAM_25),
        "source_2024_shared_upstream": str(SHARED_UPSTREAM_24),
        "paired_reference_2024": {
            "downstream": reference_downstream,
            "volume": metrics_24,
            "outputs": outputs_24,
        },
        "shared_baseline_2025": {"volume": shared_25, "outputs": shared_outputs},
        "conservative_gate_2025": {"volume": gate_25, "outputs": gate_outputs},
        "survey_evaluation": {
            "shared_baseline": shared_eval,
            "conservative_gate": gate_eval,
        },
        "paired_effect": effect,
        "limits": [
            "The 2024 reference is unchanged; the selective gate is a 2025 new-coverage mechanism.",
            "Volumes remain conditional on the existing DEM boundary-level and V75 calculations.",
            "Two uncertain full-removal holdout cases remain a side-effect risk.",
            "This is Pihe only and does not establish transfer to Shihe, Hangbuhe, or all Pishihang.",
            "No production output or model parameter is modified.",
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
