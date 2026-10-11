#!/usr/bin/env python3
"""Propagate the conservative new-SAR gate through the frozen downstream rules.

The shared-SAR/shared-alpha upstream mask is run as the paired baseline and the
conservative clear-low-no-aux mask as the experiment.  Only 2025 Pihe is run.
DEM, level, and volume are deliberately excluded from this step.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from run_a1_downstream import ROOT, compare_masks, prepare_context, run_variant


DATE = "2025-05-21"
UPSTREAM_RESULT = (
    ROOT / "diagnostics/results/new_sar_evidence_gate_upstream/20261010_134344_780095"
)
SHARED_BASELINE = (
    ROOT / "diagnostics/results/shared_sar_calibration_upstream/20261009_193736_367365"
    / DATE / "shared_sar_shared_alpha_c4_water_mask.tif"
)
CONSERVATIVE = UPSTREAM_RESULT / "clear_low_no_aux_gate_c4_water_mask.tif"
HOLDOUT_REVIEW = (
    ROOT / "diagnostics/results/new_sar_gate_holdout_review/20261010_142919_137639"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/new_sar_gate_downstream"


def stage_paths(metrics: dict) -> dict[str, Path]:
    return {
        "post_deduct_morph": Path(metrics["outputs"]["post_deduct_morph_tif"]),
        "area_gate": Path(metrics["outputs"]["area_gate_tif"]),
        "after_landuse": Path(metrics["outputs"]["final_mask_tif"]),
    }


def main() -> int:
    upstream = json.loads((UPSTREAM_RESULT / "summary.json").read_text(encoding="utf-8"))
    review = json.loads((HOLDOUT_REVIEW / "summary.json").read_text(encoding="utf-8"))
    labels = HOLDOUT_REVIEW / "assistant_visual_labels.csv"
    if upstream.get("status") != "complete":
        raise RuntimeError("Selective new-SAR upstream experiment is incomplete")
    if review.get("status") != "complete" or review.get("target_count") != 12 or not labels.exists():
        raise RuntimeError("Independent holdout review is incomplete")
    for path in (SHARED_BASELINE, CONSERVATIVE):
        if not path.exists():
            raise FileNotFoundError(path)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    date_dir = output / DATE
    date_dir.mkdir()
    cfg, water_crs, aoi, process_bounds, layers = prepare_context(DATE)

    print(f"Paired downstream propagation: {DATE}", flush=True)
    baseline = run_variant(
        DATE, "shared_baseline", SHARED_BASELINE, date_dir,
        cfg, water_crs, aoi, process_bounds, layers,
    )
    experiment = run_variant(
        DATE, "conservative_gate", CONSERVATIVE, date_dir,
        cfg, water_crs, aoi, process_bounds, layers,
    )
    baseline_paths = stage_paths(baseline)
    experiment_paths = stage_paths(experiment)
    changes = {
        stage: compare_masks(
            baseline_paths[stage], experiment_paths[stage],
            date_dir / f"conservative_vs_shared_{stage}_change.tif",
        )
        for stage in baseline_paths
    }

    report = {
        "status": "complete",
        "experiment": "Conservative new-SAR gate paired downstream propagation",
        "date": DATE,
        "region": "pihe",
        "source_upstream": str(UPSTREAM_RESULT),
        "source_holdout_review": str(HOLDOUT_REVIEW),
        "upstream_water_km2": {
            "shared_baseline": 414.8044,
            "conservative_gate": upstream["variants"]["clear_low_no_aux_gate"]["water_km2"],
            "upstream_removed": upstream["variants"]["clear_low_no_aux_gate"]["removed_vs_shared_km2"],
        },
        "paired_baseline": baseline,
        "conservative_gate": experiment,
        "stage_changes": changes,
        "dem_and_volume_computed": False,
        "limits": [
            "This is a paired propagation through the existing deduction, morphology, area, and land-use rules.",
            "The holdout review supports continuation but includes two uncertain full-removal cases.",
            "Downstream survival does not establish field-truth accuracy or survey membership.",
            "Only the 2025 Pihe AOI is tested; no Shihe, Hangbuhe, or whole-Pishihang inference is made.",
            "No production code, production output, DEM volume, or final parameter is modified.",
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
