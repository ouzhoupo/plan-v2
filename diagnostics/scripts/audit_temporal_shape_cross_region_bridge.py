#!/usr/bin/env python3
"""Bridge the temporal-shape mechanism from Pihe to Shihe and Hangbuhe.

The original V75 upstream mask is reproduced through the frozen downstream
chain for all three regions on 2024-06-01. Candidate selection then uses only
2024-05-21 upstream support and the qualified loose morphology rule. 2025
support is retrospective reporting only. No DEM volume is computed.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import pond_storage_monitor as monitor  # noqa: E402
import run_a1_downstream as downstream  # noqa: E402

TARGET = "2024-06-01"
PREVIOUS = "2024-05-21"
FUTURE: str | None = "2025-05-21"
REGIONS = ("pihe", "shihe", "hangbuhe")
OUTPUT_ROOT = ROOT / "diagnostics/results/temporal_shape_cross_region_bridge"


def compact(day: str) -> str:
    return day.replace("-", "")


def cfg_path(region: str) -> Path:
    return ROOT / f"output/{compact(TARGET)}/final/{region}/resolved_monitor_cfg_{TARGET}.yaml"


def upstream(day: str, region: str) -> Path:
    return ROOT / f"output/{compact(day)}/{region}/water_mask_{day}.tif"


def final_storage(region: str) -> Path:
    return ROOT / f"output/{compact(TARGET)}/final/{region}/pond_storage_{TARGET}.csv"


def saved_prepared(region: str) -> Path:
    return ROOT / f"output/{compact(TARGET)}/final/{region}/pond_water_mask_{TARGET}.tif"


def prepare_context(region: str):
    cfg = downstream.remap_path(yaml.safe_load(cfg_path(region).read_text(encoding="utf-8")))
    history = upstream(TARGET, region)
    with rasterio.open(history) as water:
        water_crs = monitor.parse_crs(water.crs)
        water_bounds = tuple(water.bounds)
    if water_crs is None:
        raise RuntimeError(f"Missing CRS: {history}")
    aoi = monitor.read_union_geometry(cfg["aoi_shp"], water_crs)
    process_bounds = monitor.intersect_bounds(water_bounds, aoi.bounds)
    with rasterio.open(cfg["dem_tif"]) as dem:
        dem_crs = monitor.parse_crs(dem.crs)
        dem_bounds = (
            monitor.transform_bounds(dem_crs, water_crs, *dem.bounds, densify_pts=21)
            if dem_crs != water_crs else tuple(dem.bounds)
        )
    process_bounds = monitor.intersect_bounds(process_bounds, dem_bounds)
    line_buffer = float(cfg.get("line_buffer_m", 15.0))
    deduct_bounds = (
        process_bounds[0] - line_buffer,
        process_bounds[1] - line_buffer,
        process_bounds[2] + line_buffer,
        process_bounds[3] + line_buffer,
    )
    extra = cfg.get("extra_vectors", []) or []
    if isinstance(extra, str):
        extra = [extra]
    layers = monitor.build_deduct_layers(
        river_dir=cfg.get("river_dir"),
        reservoir_shp=cfg.get("reservoir_shp"),
        extra_vectors=extra,
        target_crs=water_crs,
        bbox_filter=deduct_bounds,
        line_buffer_m=line_buffer,
    )
    return cfg, water_crs, aoi, process_bounds, layers


def historical_metrics(region: str) -> tuple[int, float]:
    with final_storage(region).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return len(rows), sum(float(row["area_m2"]) for row in rows)


def mask_difference(left: Path, right: Path) -> tuple[int, int]:
    added = removed = 0
    with rasterio.open(left) as a, rasterio.open(right) as b:
        if (
            a.shape != b.shape or a.crs != b.crs
            or not a.transform.almost_equals(b.transform)
        ):
            raise RuntimeError("Baseline reproduction grid mismatch")
        for _, window in a.block_windows(1):
            av = a.read(1, window=window) > 0
            bv = b.read(1, window=window) > 0
            added += int(np.count_nonzero(bv & ~av))
            removed += int(np.count_nonzero(av & ~bv))
    return added, removed


def aligned_mask(stack: ExitStack, path: Path, reference) -> np.ndarray:
    source = stack.enter_context(rasterio.open(path))
    vrt = stack.enter_context(WarpedVRT(
        source,
        crs=reference.crs,
        transform=reference.transform,
        width=reference.width,
        height=reference.height,
        resampling=Resampling.nearest,
    ))
    return vrt.read(1) > 0


def process_region(region: str, output: Path) -> dict:
    region_dir = output / region
    region_dir.mkdir()
    cfg, water_crs, aoi, process_bounds, layers = prepare_context(region)
    metrics = downstream.run_variant(
        TARGET,
        "original_v75_bridge",
        upstream(TARGET, region),
        region_dir,
        cfg,
        water_crs,
        aoi,
        process_bounds,
        layers,
    )
    area_path = Path(metrics["outputs"]["area_gate_tif"])
    final_path = Path(metrics["outputs"]["final_mask_tif"])
    prepared_path = Path(metrics["outputs"]["post_deduct_morph_tif"])
    added, removed = mask_difference(saved_prepared(region), prepared_path)
    expected_objects, expected_area = historical_metrics(region)
    final_metrics = metrics["after_landuse"]
    reproduction = {
        "prepared_added_pixels": added,
        "prepared_removed_pixels": removed,
        "expected_final_objects": expected_objects,
        "reproduced_final_objects": final_metrics["object_count"],
        "expected_final_area_m2": expected_area,
        "reproduced_final_area_m2": final_metrics["area_m2"],
    }
    if (
        added or removed
        or final_metrics["object_count"] != expected_objects
        or abs(float(final_metrics["area_m2"]) - expected_area) > 0.001
    ):
        raise RuntimeError(f"{region} original downstream baseline reproduction failed")

    with rasterio.open(area_path) as reference:
        area_gate = reference.read(1) > 0
        transform = reference.transform
        pixel_area = abs(float(transform.a * transform.e))
        with ExitStack() as stack:
            previous = aligned_mask(stack, upstream(PREVIOUS, region), reference)
            future = (
                aligned_mask(stack, upstream(FUTURE, region), reference)
                if FUTURE is not None else None
            )
    with rasterio.open(final_path) as source:
        final = source.read(1) > 0

    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    pixels = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    final_counts = np.bincount(labels[final], minlength=count + 1)[1:]
    previous_near = ndimage.binary_dilation(previous, iterations=1)
    previous_counts = np.bincount(labels[previous_near], minlength=count + 1)[1:]
    if future is not None:
        future_near = ndimage.binary_dilation(future, iterations=1)
        future_counts = np.bincount(labels[future_near], minlength=count + 1)[1:]
    else:
        future_near = None
        future_counts = None
    rejected = set(np.flatnonzero(final_counts == 0) + 1)

    geometries = {
        int(value): shape(geometry)
        for geometry, value in shapes(
            labels.astype("int32"), mask=area_gate, transform=transform, connectivity=4
        )
        if int(value) in rejected
    }
    rows = []
    for object_id in sorted(rejected):
        index = object_id - 1
        n = int(pixels[index])
        area = n * pixel_area
        previous_fraction = int(previous_counts[index]) / n
        future_fraction = (
            int(future_counts[index]) / n if future_counts is not None else None
        )
        geom = geometries[object_id]
        minx, miny, maxx, maxy = geom.bounds
        width, height = maxx - minx, maxy - miny
        short = min(width, height)
        fill = area / (width * height) if width > 0 and height > 0 else 0.0
        elongation = max(width, height) / short if short > 0 else float("inf")
        selected = (
            previous_fraction >= 0.25
            and 2000 <= area < 50000
            and fill >= 0.45
            and elongation <= 3.0
        )
        rows.append({
            "region": region,
            "object_id": object_id,
            "area_m2": area,
            "previous_near_fraction": previous_fraction,
            "future_near_fraction_retrospective": future_fraction,
            "bbox_fill": fill,
            "elongation": elongation,
            "compactness": 4 * math.pi * area / geom.length ** 2 if geom.length else 0.0,
            "selected_loose_candidate": selected,
            "future_fraction_ge_025_retrospective": (
                future_fraction >= 0.25 if future_fraction is not None else ""
            ),
        })

    table = region_dir / "landuse_rejected_temporal_shape_bridge.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row["selected_loose_candidate"]]
    future_selected = [
        row for row in selected
        if row["future_fraction_ge_025_retrospective"] is True
    ]
    selected_area = sum(row["area_m2"] for row in selected)
    future_area = sum(row["area_m2"] for row in future_selected)
    report = {
        "region": region,
        "baseline_reproduction": reproduction,
        "area_gate": metrics["area_gate"],
        "after_landuse": metrics["after_landuse"],
        "landuse_rejected": {
            "objects": len(rows),
            "area_m2": sum(row["area_m2"] for row in rows),
        },
        "loose_candidate": {
            "objects": len(selected),
            "area_m2": selected_area,
            "area_km2": selected_area / 1e6,
            "fraction_of_rejected_objects": len(selected) / len(rows) if rows else 0.0,
            "fraction_of_rejected_area": (
                selected_area / sum(row["area_m2"] for row in rows) if rows else 0.0
            ),
            "future_proxy_objects": (
                len(future_selected) if future_counts is not None else None
            ),
            "future_proxy_area_m2": future_area if future_counts is not None else None,
            "future_proxy_object_fraction": (
                len(future_selected) / len(selected)
                if future_counts is not None and selected else None
            ),
            "future_proxy_area_fraction": (
                future_area / selected_area
                if future_counts is not None and selected_area else None
            ),
        },
        "output_table": str(table),
    }
    del area_gate, final, labels, previous, future, previous_near, future_near, geometries
    gc.collect()
    return report


def main() -> int:
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "experiment": "Temporal-shape original-V75 cross-region bridge",
        "target_date": TARGET,
        "previous_date": PREVIOUS,
        "future_proxy_date": FUTURE,
        "regions": {},
        "selection": "previous support >=0.25, area 0.2-5 ha, bbox fill >=0.45, elongation <=3",
    }
    for region in REGIONS:
        print(f"Cross-region bridge: {region}", flush=True)
        report["regions"][region] = process_region(region, output)
        (output / "summary.partial.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    report["status"] = "complete"
    report["interpretation"] = [
        "The same original V75 baseline and downstream rules are used in all three regions.",
        "The Pihe bridge quantifies sensitivity to changing from the qualified shared-SAR baseline back to original V75.",
        "Future support is a retrospective comparability proxy and never selects an object.",
        "Candidate capacity and proxy fractions test spatial transfer of the mechanism, not manual-volume accuracy."
    ]
    report["limits"] = [
        "No DEM volume or manual trend evaluation is computed for Shihe or Hangbuhe.",
        "The original V75 bridge is not numerically identical to the qualified Pihe shared-SAR experiment.",
        "Future persistence can repeat stable model errors.",
        "No production code, mask, parameter, or formal output is changed."
    ]
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print("STATUS: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
