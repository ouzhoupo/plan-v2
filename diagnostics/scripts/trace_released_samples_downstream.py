"""Trace 31 fixed released samples through saved-mask downstream logic; no training/DEM."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import pond_storage_monitor as monitor  # noqa: E402


SOURCE = (
    ROOT
    / "diagnostics/results/positive_label_experiment"
    / "20260913_125712_794605"
)
OLD_WINDOWS = (
    ROOT
    / "diagnostics/results/local_downstream"
    / "20260913_094837_488085"
)
OUT = (
    ROOT
    / "diagnostics/results/released_samples_downstream"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)


def patch_experiment_tiles(date, baseline, transform):
    changed = baseline.copy()
    coverage = np.zeros_like(baseline, dtype=bool)
    used = []
    for path in sorted(SOURCE.glob(f"{date}_*_masks.tif")):
        with rasterio.open(path) as tile:
            tw = from_bounds(*tile.bounds, transform=transform).round_offsets().round_lengths()
            x, y = int(tw.col_off), int(tw.row_off)
            x0, y0 = max(0, x), max(0, y)
            x1 = min(changed.shape[1], x + tile.width)
            y1 = min(changed.shape[0], y + tile.height)
            if x1 <= x0 or y1 <= y0:
                continue
            expected = rasterio.windows.transform(
                Window(x, y, tile.width, tile.height), transform
            )
            if not tile.transform.almost_equals(expected):
                raise RuntimeError(f"Tile grid mismatch: {path}")
            arr = tile.read(window=Window(x0 - x, y0 - y, x1 - x0, y1 - y0)) > 0
            if not np.array_equal(arr[0], baseline[y0:y1, x0:x1]):
                raise RuntimeError(f"Historical tile mismatch: {path}")
            changed[y0:y1, x0:x1] = arr[2]
            coverage[y0:y1, x0:x1] = True
            used.append(path.name)
    return changed, coverage, used


def raster_overlap_pixels(array, geom, transform):
    inside = geometry_mask(
        [geom.__geo_interface__],
        out_shape=array.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    return int((array & inside).sum())


def main():
    experiment = json.loads((SOURCE / "experiment.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for key, item in experiment["released_samples"].items():
        groups[(item["date"], int(item["window_id"]))].append(
            {"key": key, **item, "geom": shape(item["geometry"])}
        )

    OUT.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": (
            "Fixed 31 reviewed samples; local historical water outside saved experiment "
            "tiles is unchanged. Production downstream functions only; no training, DEM, "
            "basin totals, or accuracy extrapolation."
        ),
        "experiment_source": str(SOURCE),
        "groups": [],
        "samples": [],
    }

    for (date, window_id), samples in sorted(groups.items()):
        day = ROOT / "output" / date.replace("-", "")
        cfg = yaml.safe_load(
            (day / f"final/pihe/resolved_monitor_cfg_{date}.yaml").read_text(
                encoding="utf-8"
            )
        )
        active_unsupported = [
            key
            for key in (
                "watershed_split_enable",
                "linear_artifact_filter_enable",
                "landuse_water_hard_mask",
                "cropland_mask_tif",
                "extra_vectors",
            )
            if cfg.get(key)
        ]
        if active_unsupported:
            raise RuntimeError(f"Unsupported active configuration: {active_unsupported}")

        old_window = OLD_WINDOWS / f"{date}_{window_id}" / "baseline_water.tif"
        if not old_window.exists():
            raise FileNotFoundError(old_window)
        with rasterio.open(old_window) as local:
            bounds = local.bounds
            transform = local.transform
            height, width = local.height, local.width
            crs = local.crs
        with rasterio.open(day / f"pihe/water_mask_{date}.tif") as src:
            src_win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            if not src.window_transform(src_win).almost_equals(transform):
                raise RuntimeError(f"Historical grid mismatch: {date} {window_id}")
            baseline = src.read(1, window=src_win) > 0
        if baseline.shape != (height, width):
            raise RuntimeError(f"Window shape mismatch: {date} {window_id}")
        experiment_mask, coverage, used_tiles = patch_experiment_tiles(
            date, baseline, transform
        )

        with rasterio.open(day / f"final/pihe/pond_water_mask_{date}.tif") as saved:
            saved_win = from_bounds(*bounds, transform=saved.transform).round_offsets().round_lengths()
            historical_candidate = saved.read(1, window=saved_win) > 0

        aoi = monitor.read_union_geometry(str(ROOT / "data/aoi/pihe.shp"), crs)
        rivers = day / "pihe/buffered_river"
        reservoir = ROOT / "data/reservoir/psh_res.shp"
        landuse = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
        layers = monitor.build_deduct_layers(
            river_dir=str(rivers),
            reservoir_shp=str(reservoir),
            extra_vectors=[],
            target_crs=crs,
            bbox_filter=bounds,
            line_buffer_m=float(cfg.get("line_buffer_m", 0)),
        )
        variants = {}
        for name, array in (("baseline", baseline), ("experiment", experiment_mask)):
            water_path = OUT / f"{date}_{window_id}_{name}_water.tif"
            candidate_path = OUT / f"{date}_{window_id}_{name}_candidate.tif"
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "crs": crs,
                "transform": transform,
                "nodata": 0,
                "compress": "deflate",
            }
            with rasterio.open(water_path, "w", **profile) as dst:
                dst.write(array.astype("uint8"), 1)
            monitor.prepare_water_mask(
                str(water_path),
                str(candidate_path),
                process_bounds=bounds,
                aoi_geom=aoi,
                deduct_layers=layers,
                water_value=None,
                respect_source_mask=bool(cfg.get("respect_source_mask", False)),
                block_size=int(cfg.get("block_size", 512)),
                close_radius_px=int(cfg.get("close_radius_px", 0)),
                open_radius_px=int(cfg.get("open_radius_px", 0)),
                fill_holes=bool(cfg.get("fill_holes", False)),
                all_touched=bool(cfg.get("all_touched", False)),
            )
            with rasterio.open(candidate_path) as src:
                candidate = src.read(1) > 0
            if name == "baseline":
                mismatch = int((candidate != historical_candidate).sum())
                if mismatch:
                    raise RuntimeError(
                        f"Baseline candidate mismatch {date} {window_id}: {mismatch}"
                    )
            objects = monitor.polygonize_water_mask(
                str(candidate_path),
                min_area_m2=float(cfg["min_water_area_m2"]),
                max_area_m2=float(cfg["max_water_area_m2"]),
                simplify_tolerance=float(cfg.get("simplify_tolerance_m", 0)),
                watershed_split_enable=False,
            )
            landuse_kept = monitor.filter_by_landuse_water(objects, str(landuse), crs)
            variants[name] = {
                "water": array,
                "candidate": candidate,
                "objects": objects,
                "kept_ids": {obj.obj_id for obj in landuse_kept},
            }

        report["groups"].append(
            {
                "date": date,
                "window_id": window_id,
                "sample_count": len(samples),
                "baseline_candidate_mismatch_pixels": 0,
                "used_experiment_tiles": used_tiles,
                "experiment_coverage_pixels": int(coverage.sum()),
            }
        )

        for sample in sorted(samples, key=lambda item: int(item["local_id"])):
            geom = sample["geom"]
            row = {
                key: value
                for key, value in sample.items()
                if key not in ("geometry", "geom")
            }
            row["variants"] = {}
            for name, data in variants.items():
                matched = [obj for obj in data["objects"] if obj.geom.intersects(geom)]
                overlap = [
                    obj for obj in matched if obj.geom.intersection(geom).area > 0
                ]
                kept = [obj for obj in overlap if obj.obj_id in data["kept_ids"]]
                row["variants"][name] = {
                    "water_pixels_in_sample": raster_overlap_pixels(
                        data["water"], geom, transform
                    ),
                    "candidate_pixels_in_sample": raster_overlap_pixels(
                        data["candidate"], geom, transform
                    ),
                    "area_qualified_objects": len(overlap),
                    "area_qualified_full_area_m2": float(
                        sum(obj.area_m2 for obj in overlap)
                    ),
                    "area_qualified_overlap_m2": float(
                        sum(obj.geom.intersection(geom).area for obj in overlap)
                    ),
                    "landuse_kept_objects": len(kept),
                    "landuse_kept_overlap_m2": float(
                        sum(obj.geom.intersection(geom).area for obj in kept)
                    ),
                    "final_retained": bool(kept),
                    "matched_local_ids": [obj.obj_id for obj in overlap],
                    "kept_local_ids": [obj.obj_id for obj in kept],
                    "touches_window_boundary": any(
                        obj.geom.intersects(box(*bounds).boundary) for obj in overlap
                    ),
                }
            report["samples"].append(row)
        print(date, window_id, "complete", flush=True)

    json_path = OUT / "trace.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "date",
        "window_id",
        "local_id",
        "stratum",
        "original_area_m2",
        "variant",
        "water_pixels_in_sample",
        "candidate_pixels_in_sample",
        "area_qualified_objects",
        "area_qualified_full_area_m2",
        "area_qualified_overlap_m2",
        "landuse_kept_objects",
        "landuse_kept_overlap_m2",
        "final_retained",
        "touches_window_boundary",
    ]
    with (OUT / "sample_stage_status.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in report["samples"]:
            for variant, stats in sample["variants"].items():
                writer.writerow(
                    {
                        "date": sample["date"],
                        "window_id": sample["window_id"],
                        "local_id": sample["local_id"],
                        "stratum": sample["stratum"],
                        "original_area_m2": sample["area_m2"],
                        "variant": variant,
                        **{field: stats[field] for field in fields if field in stats},
                    }
                )
    print("RESULT:", json_path)


if __name__ == "__main__":
    main()
