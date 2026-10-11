#!/usr/bin/env python3
"""Partition 2024-06 C4 deduction loss into reservoir, lake and linear groups."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from run_a1_downstream import ROOT, prepare_context

sys.path.insert(0, str(ROOT / "code"))
import pond_storage_monitor as monitor  # noqa: E402


DATE = "2024-06-01"
SOURCE_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_morphology_split"
    / "20260926_203928_115516"
)
UPSTREAM = ROOT / "diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712"
RESERVOIR_NAMES = {"psh_res", "known_reservoir"}
LAKE_NAMES = {"__4", "__4326"}
GROUP_BITS = {"reservoir": 1, "lake": 2, "linear_river_canal": 4}


def run_group(source, output, cfg, aoi, process_bounds, layers):
    monitor.prepare_water_mask(
        str(source),
        str(output),
        process_bounds=process_bounds,
        aoi_geom=aoi,
        deduct_layers=layers,
        water_value=cfg.get("water_value"),
        respect_source_mask=bool(cfg.get("respect_source_mask", False)),
        block_size=int(cfg.get("block_size", 512)),
        close_radius_px=0,
        open_radius_px=0,
        fill_holes=False,
        all_touched=bool(cfg.get("all_touched", False)),
        cropland_mask_tif=cfg.get("cropland_mask_tif") or None,
        landuse_water_hard_mask_tif=(
            cfg.get("landuse_water_tif")
            if bool(cfg.get("landuse_water_hard_mask", False))
            else None
        ),
        landuse_water_value=int(cfg.get("landuse_water_value", 1)),
    )


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    previous = json.loads((SOURCE_RESULT / "summary.json").read_text(encoding="utf-8"))
    if previous.get("status") != "complete":
        raise RuntimeError("Deduction/morphology split result is incomplete")

    source = UPSTREAM / DATE / "c4_water_mask.tif"
    pre_path = SOURCE_RESULT / "c4_pre_deduction_no_morph.tif"
    full_post_path = SOURCE_RESULT / "c4_post_deduction_no_morph.tif"
    if not source.exists() or not pre_path.exists() or not full_post_path.exists():
        raise FileNotFoundError("Required frozen raster is missing")

    cfg, _water_crs, aoi, process_bounds, layers = prepare_context(DATE)
    groups = {
        "reservoir": [layer for layer in layers if layer.name in RESERVOIR_NAMES],
        "lake": [layer for layer in layers if layer.name in LAKE_NAMES],
        "linear_river_canal": [
            layer
            for layer in layers
            if layer.name not in RESERVOIR_NAMES and layer.name not in LAKE_NAMES
        ],
    }
    if any(not items for items in groups.values()):
        raise RuntimeError({name: [layer.name for layer in items] for name, items in groups.items()})

    output = (
        ROOT
        / "diagnostics/results/second_round_deduction_groups"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    group_paths = {}
    for name, items in groups.items():
        print(f"Reconstructing deduction group: {name}", flush=True)
        path = output / f"post_{name}_deduction.tif"
        run_group(source, path, cfg, aoi, process_bounds, items)
        group_paths[name] = path

    membership_path = output / "deduction_group_membership.tif"
    combination_counts = {code: 0 for code in range(1, 8)}
    group_counts = {name: 0 for name in groups}
    full_union_count = 0
    mismatch_pixels = 0

    with rasterio.open(pre_path) as pre, rasterio.open(full_post_path) as full_post:
        opened = {name: rasterio.open(path) for name, path in group_paths.items()}
        try:
            profile = pre.profile.copy()
            profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True)
            pixel_area = abs(pre.transform.a * pre.transform.e - pre.transform.b * pre.transform.d)
            with rasterio.open(membership_path, "w", **profile) as destination:
                for _, window in pre.block_windows(1):
                    before = pre.read(1, window=window) > 0
                    full_removed = before & ~(full_post.read(1, window=window) > 0)
                    code = np.zeros(before.shape, dtype="uint8")
                    for name, dataset in opened.items():
                        removed = before & ~(dataset.read(1, window=window) > 0)
                        group_counts[name] += int(np.count_nonzero(removed))
                        code[removed] |= GROUP_BITS[name]
                    group_union = code > 0
                    mismatch_pixels += int(np.count_nonzero(group_union ^ full_removed))
                    full_union_count += int(np.count_nonzero(full_removed))
                    for value in range(1, 8):
                        combination_counts[value] += int(np.count_nonzero(code == value))
                    destination.write(code, 1, window=window)
        finally:
            for dataset in opened.values():
                dataset.close()

    code_labels = {
        1: "reservoir_only",
        2: "lake_only",
        3: "reservoir_and_lake",
        4: "linear_only",
        5: "reservoir_and_linear",
        6: "lake_and_linear",
        7: "reservoir_lake_and_linear",
    }
    rows = [
        {
            "code": code,
            "combination": code_labels[code],
            "pixels": combination_counts[code],
            "area_km2": combination_counts[code] * pixel_area / 1_000_000.0,
            "share_of_full_deduction_percent": (
                100.0 * combination_counts[code] / full_union_count if full_union_count else 0.0
            ),
        }
        for code in range(1, 8)
    ]
    write_csv(output / "deduction_group_combinations.csv", rows)

    report = {
        "status": "complete" if mismatch_pixels == 0 else "validation_failed",
        "experiment": "Second-round 2024-06 C4 deduction group partition",
        "date": DATE,
        "groups": {
            name: {
                "layer_names": [layer.name for layer in items],
                "sources": [layer.source for layer in items],
                "union_pixels": group_counts[name],
                "union_area_km2": group_counts[name] * pixel_area / 1_000_000.0,
            }
            for name, items in groups.items()
        },
        "full_deduction_union": {
            "pixels": full_union_count,
            "area_km2": full_union_count * pixel_area / 1_000_000.0,
        },
        "group_union_vs_full_deduction_mismatch_pixels": mismatch_pixels,
        "membership_codes": {str(key): value for key, value in code_labels.items()},
        "combinations": rows,
        "outputs": {
            "membership_tif": str(membership_path),
            "combination_csv": str(output / "deduction_group_combinations.csv"),
            "group_post_deduction_tifs": {name: str(path) for name, path in group_paths.items()},
        },
        "limits": [
            "The three group areas overlap; combination rows provide the mutually exclusive accounting.",
            "A deduction group describes which vector mask covers a pixel, not whether deletion is correct.",
            "No objects are restored and no DEM volume is calculated.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {report['status']}", flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
