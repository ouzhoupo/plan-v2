#!/usr/bin/env python3
"""Split the 2024-06 C4 upstream loss into deduction and morphology.

This is a read-only diagnostic experiment. It reuses the frozen C4 water
mask and resolved production configuration, writes under diagnostics/results,
and does not restore objects or calculate DEM volume.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_a1_downstream import ROOT, compare_masks, prepare_context

sys.path.insert(0, str(ROOT / "code"))
import pond_storage_monitor as monitor  # noqa: E402


DATE = "2024-06-01"
UPSTREAM = ROOT / "diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712"
DOWNSTREAM = ROOT / "diagnostics/results/combined_downstream_c0_c4/20260922_185751_785910"


def run_prepare(
    source: Path,
    output: Path,
    cfg: dict[str, Any],
    aoi,
    process_bounds,
    layers,
    *,
    morphology: bool,
) -> dict[str, Any]:
    _, summary = monitor.prepare_water_mask(
        str(source),
        str(output),
        process_bounds=process_bounds,
        aoi_geom=aoi,
        deduct_layers=layers,
        water_value=cfg.get("water_value"),
        respect_source_mask=bool(cfg.get("respect_source_mask", False)),
        block_size=int(cfg.get("block_size", 512)),
        close_radius_px=int(cfg.get("close_radius_px", 0)) if morphology else 0,
        open_radius_px=int(cfg.get("open_radius_px", 0)) if morphology else 0,
        fill_holes=bool(cfg.get("fill_holes", False)) if morphology else False,
        all_touched=bool(cfg.get("all_touched", False)),
        cropland_mask_tif=cfg.get("cropland_mask_tif") or None,
        landuse_water_hard_mask_tif=(
            cfg.get("landuse_water_tif")
            if bool(cfg.get("landuse_water_hard_mask", False))
            else None
        ),
        landuse_water_value=int(cfg.get("landuse_water_value", 1)),
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    upstream_summary = json.loads((UPSTREAM / "summary.json").read_text(encoding="utf-8"))
    downstream_summary = json.loads((DOWNSTREAM / "summary.json").read_text(encoding="utf-8"))
    if upstream_summary.get("status") != "complete":
        raise RuntimeError("Frozen C0-C4 upstream result is incomplete")
    if downstream_summary.get("status") != "complete":
        raise RuntimeError("Frozen C0-C4 downstream result is incomplete")

    source = UPSTREAM / DATE / "c4_water_mask.tif"
    frozen = DOWNSTREAM / DATE / "c4_post_deduct_morph.tif"
    if not source.exists() or not frozen.exists():
        raise FileNotFoundError("Required frozen C4 raster is missing")

    expected = downstream_summary["dates"][DATE]["variants"]["c4"]["post_deduct_morph"]
    cfg, _water_crs, aoi, process_bounds, layers = prepare_context(DATE)

    output = (
        ROOT
        / "diagnostics/results/second_round_deduction_morphology_split"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    pre_deduction = output / "c4_pre_deduction_no_morph.tif"
    post_deduction = output / "c4_post_deduction_no_morph.tif"
    reproduced_final = output / "c4_post_deduction_with_morph.tif"

    print("1/3 Reconstructing the C4 mask before deduction", flush=True)
    pre_summary = run_prepare(
        source, pre_deduction, cfg, aoi, process_bounds, [], morphology=False
    )
    print("2/3 Applying deduction without morphology", flush=True)
    post_summary = run_prepare(
        source, post_deduction, cfg, aoi, process_bounds, layers, morphology=False
    )
    print("3/3 Reproducing deduction plus the original morphology", flush=True)
    final_summary = run_prepare(
        source, reproduced_final, cfg, aoi, process_bounds, layers, morphology=True
    )

    deduction_change = compare_masks(
        pre_deduction, post_deduction, output / "deduction_change.tif"
    )
    morphology_change = compare_masks(
        post_deduction, reproduced_final, output / "morphology_change.tif"
    )
    frozen_check = compare_masks(
        frozen, reproduced_final, output / "frozen_reproduction_change.tif"
    )

    validations = {
        "frozen_reproduction_exact": (
            frozen_check["added_pixels"] == 0 and frozen_check["removed_pixels"] == 0
        ),
        "deduction_has_no_added_pixels": deduction_change["added_pixels"] == 0,
        "deduction_union_matches_frozen_summary": (
            post_summary["deducted_union_pixels"] == expected["deducted_union_pixels"]
        ),
        "final_candidate_matches_frozen_summary": (
            final_summary["pond_candidate_pixels"] == expected["pond_candidate_pixels"]
        ),
    }

    pixel_area = float(final_summary["pixel_area_m2"])
    pre_pixels = int(pre_summary["pond_candidate_pixels"])
    post_pixels = int(post_summary["pond_candidate_pixels"])
    final_pixels = int(final_summary["pond_candidate_pixels"])
    deduction_removed = int(deduction_change["removed_pixels"])
    morph_added = int(morphology_change["added_pixels"])
    morph_removed = int(morphology_change["removed_pixels"])
    gross_loss = pre_pixels - final_pixels
    reconstructed_loss = deduction_removed + morph_removed - morph_added
    validations["stage_accounting_closes"] = gross_loss == reconstructed_loss

    if not all(validations.values()):
        report = {
            "status": "validation_failed",
            "experiment": "Second-round 2024-06 C4 deduction and morphology split",
            "validations": validations,
            "deduction_change": deduction_change,
            "morphology_change": morphology_change,
            "frozen_reproduction": frozen_check,
        }
        (output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"RESULT_FOLDER: {output}", flush=True)
        print("STATUS: validation_failed", flush=True)
        return 2

    stage_rows = [
        {
            "stage": "pre_deduction",
            "pixels": pre_pixels,
            "area_km2": pre_pixels * pixel_area / 1_000_000.0,
            "meaning": "C4 water before vector deduction and morphology",
        },
        {
            "stage": "deduction_removed",
            "pixels": deduction_removed,
            "area_km2": deduction_removed * pixel_area / 1_000_000.0,
            "meaning": "C4 water intersecting the deduction-layer union",
        },
        {
            "stage": "post_deduction",
            "pixels": post_pixels,
            "area_km2": post_pixels * pixel_area / 1_000_000.0,
            "meaning": "Water remaining after deduction and before morphology",
        },
        {
            "stage": "morphology_added",
            "pixels": morph_added,
            "area_km2": morph_added * pixel_area / 1_000_000.0,
            "meaning": "Pixels added by the configured morphology",
        },
        {
            "stage": "morphology_removed",
            "pixels": morph_removed,
            "area_km2": morph_removed * pixel_area / 1_000_000.0,
            "meaning": "Pixels removed by the configured morphology",
        },
        {
            "stage": "post_deduction_and_morphology",
            "pixels": final_pixels,
            "area_km2": final_pixels * pixel_area / 1_000_000.0,
            "meaning": "Frozen C4 candidate mask before the area gate",
        },
        {
            "stage": "net_pre_to_final_loss",
            "pixels": gross_loss,
            "area_km2": gross_loss * pixel_area / 1_000_000.0,
            "meaning": "Accounting loss, not confirmed model error",
        },
    ]
    write_csv(output / "stage_accounting.csv", stage_rows)

    layer_rows = []
    for name, item in post_summary.get("layers", {}).items():
        layer_rows.append(
            {
                "layer": name,
                "source": item.get("source", ""),
                "raw_overlap_pixels": int(item.get("raw_overlap_pixels", 0)),
                "raw_overlap_km2": float(item.get("raw_overlap_m2", 0.0)) / 1_000_000.0,
                "note": "Layer overlaps can overlap each other; do not sum them as union area",
            }
        )
    layer_rows.sort(key=lambda row: row["raw_overlap_pixels"], reverse=True)
    write_csv(output / "deduction_layers_raw_overlap.csv", layer_rows)

    report = {
        "status": "complete",
        "experiment": "Second-round 2024-06 C4 deduction and morphology split",
        "date": DATE,
        "source_c4_water_mask": str(source),
        "frozen_downstream_reference": str(frozen),
        "configuration": {
            "deduction_layers": len(layers),
            "close_radius_px": int(cfg.get("close_radius_px", 0)),
            "open_radius_px": int(cfg.get("open_radius_px", 0)),
            "fill_holes": bool(cfg.get("fill_holes", False)),
            "all_touched": bool(cfg.get("all_touched", False)),
        },
        "validations": validations,
        "stage_accounting": {
            row["stage"]: {"pixels": row["pixels"], "area_km2": row["area_km2"]}
            for row in stage_rows
        },
        "deduction_change": deduction_change,
        "morphology_change": morphology_change,
        "frozen_reproduction": frozen_check,
        "outputs": {
            "pre_deduction_tif": str(pre_deduction),
            "post_deduction_tif": str(post_deduction),
            "reproduced_final_tif": str(reproduced_final),
            "deduction_change_tif": deduction_change["change_tif"],
            "morphology_change_tif": morphology_change["change_tif"],
            "stage_csv": str(output / "stage_accounting.csv"),
            "layer_csv": str(output / "deduction_layers_raw_overlap.csv"),
        },
        "interpretation_limits": [
            "Removed area is a processing-stage quantity, not a confirmed error contribution.",
            "Raw per-layer overlap areas can overlap and must not be added together.",
            "This experiment does not restore objects, run object gates, or calculate DEM volume.",
            "Spatial review is required before any deduction rule is relaxed.",
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
