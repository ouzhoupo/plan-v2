"""Fixed-seed local sample of experiment candidates kept by land-use screening."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform, unary_union
from pyproj import Transformer
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import pond_storage_monitor as monitor  # noqa: E402


DOWNSTREAM = (
    ROOT
    / "diagnostics/results/released_samples_downstream"
    / "20260913_143146_669833"
)
EXPERIMENT = (
    ROOT
    / "diagnostics/results/positive_label_experiment"
    / "20260913_125712_794605/experiment.json"
)
LANDUSE = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
OUT = (
    ROOT
    / "diagnostics/results/landuse_kept_control_review"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
SEED = 20260913
PER_SIZE = 2


def landuse_counts(geom, geom_crs, lu):
    transformer = Transformer.from_crs(geom_crs, lu.crs, always_xy=True)
    projected = shapely_transform(transformer.transform, geom)
    win = geometry_window(lu, [projected.__geo_interface__], pad_x=1, pad_y=1)
    data = lu.read(1, window=win)
    transform = lu.window_transform(win)
    inside = geometry_mask(
        [projected.__geo_interface__],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return int((data == 1).sum()), int(((data == 1) & inside).sum())


def stretch(array):
    values = array[np.isfinite(array)]
    lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    return np.nan_to_num(np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1)) ** 0.7


def main():
    source = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    excluded = defaultdict(list)
    for item in source["released_samples"].values():
        excluded[(item["date"], int(item["window_id"]))].append(shape(item["geometry"]))

    rng = np.random.default_rng(SEED)
    report = {
        "seed": SEED,
        "per_window_size_stratum": PER_SIZE,
        "scope": (
            "Local experiment candidates retained by current land-use rule, excluding "
            "the fixed 31 released samples and window-edge objects. Fixed stratified "
            "visual controls, not an accuracy estimate."
        ),
        "windows": [],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    with rasterio.open(LANDUSE) as lu:
        for date, window_id in sorted(excluded):
            path = DOWNSTREAM / f"{date}_{window_id}_experiment_candidate.tif"
            cfg = yaml.safe_load(
                (
                    ROOT
                    / f"output/{date.replace('-', '')}/final/pihe/resolved_monitor_cfg_{date}.yaml"
                ).read_text(encoding="utf-8")
            )
            with rasterio.open(path) as candidate:
                crs = candidate.crs
                boundary = box(*candidate.bounds).boundary
            objects = monitor.polygonize_water_mask(
                str(path),
                min_area_m2=float(cfg["min_water_area_m2"]),
                max_area_m2=float(cfg["max_water_area_m2"]),
                simplify_tolerance=float(cfg.get("simplify_tolerance_m", 0)),
                watershed_split_enable=False,
            )
            kept_ids = {
                item.obj_id
                for item in monitor.filter_by_landuse_water(objects, str(LANDUSE), crs)
            }
            excluded_union = unary_union(excluded[(date, window_id)])
            groups = defaultdict(list)
            for obj in objects:
                if obj.obj_id not in kept_ids or obj.geom.intersects(boundary):
                    continue
                if obj.geom.intersection(excluded_union).area > 0:
                    continue
                size = (
                    "small_under_1ha"
                    if obj.area_m2 < 10000
                    else ("medium_1to5ha" if obj.area_m2 < 50000 else "large_5ha_plus")
                )
                window_hits, object_hits = landuse_counts(obj.geom, crs, lu)
                groups[size].append(
                    {
                        "local_id": obj.obj_id,
                        "area_m2": obj.area_m2,
                        "size": size,
                        "landuse_window_value1_pixels": window_hits,
                        "landuse_object_all_touched_value1_pixels": object_hits,
                        "geometry": obj.geom,
                    }
                )
            samples = []
            totals = {}
            for size, items in sorted(groups.items()):
                items.sort(key=lambda item: item["local_id"])
                totals[size] = {
                    "objects": len(items),
                    "area_m2": float(sum(item["area_m2"] for item in items)),
                }
                indices = rng.choice(len(items), size=min(PER_SIZE, len(items)), replace=False)
                samples.extend(items[int(index)] for index in indices)

            pages = []
            with rasterio.open(ROOT / f"data/s2/s2_stack_{date}.tif") as s2:
                for start in range(0, len(samples), 4):
                    subset = samples[start : start + 4]
                    fig, axes = plt.subplots(
                        len(subset), 2, figsize=(10, 4 * len(subset)), squeeze=False,
                        constrained_layout=True,
                    )
                    for row, item in enumerate(subset):
                        geom = item["geometry"]
                        requested = geom.buffer(80).bounds
                        with rasterio.open(path) as ref:
                            win = from_bounds(*requested, transform=ref.transform).round_offsets().round_lengths()
                            transform = ref.window_transform(win)
                            height, width = int(win.height), int(win.width)
                            bounds = ref.window_bounds(win)
                            with WarpedVRT(
                                s2, crs=ref.crs, transform=transform,
                                width=width, height=height,
                                resampling=Resampling.nearest,
                            ) as vrt:
                                bands = vrt.read([3, 2, 1, 5, 4], masked=True).astype("float32").filled(np.nan)
                            with WarpedVRT(
                                lu, crs=ref.crs, transform=transform,
                                width=width, height=height,
                                resampling=Resampling.nearest,
                            ) as lu_vrt:
                                lu_water = lu_vrt.read(1) == 1
                        extent = (bounds[0], bounds[2], bounds[1], bounds[3])
                        for column, band_ids in enumerate(([0, 1, 2], [3, 4, 0])):
                            ax = axes[row, column]
                            rgb = stretch(np.moveaxis(bands[list(band_ids)], 0, -1))
                            ax.imshow(rgb, extent=extent, interpolation="nearest")
                            overlay = np.zeros((height, width, 4), dtype=float)
                            overlay[lu_water] = (0.0, 1.0, 1.0, 0.45)
                            ax.imshow(overlay, extent=extent, interpolation="nearest")
                            polygons = [geom] if geom.geom_type == "Polygon" else geom.geoms
                            for polygon in polygons:
                                for ring in [polygon.exterior, *polygon.interiors]:
                                    x, y = ring.xy
                                    ax.plot(x, y, color="yellow", linewidth=0.9)
                            ax.set_xlim(extent[:2])
                            ax.set_ylim(extent[2:])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            title = "RGB" if column == 0 else "SWIR / NIR / Red"
                            ax.set_title(
                                f"ID {item['local_id']} | {item['area_m2']/10000:.2f} ha | {item['size']}\n"
                                f"{title}; cyan=land-use water; object hits={item['landuse_object_all_touched_value1_pixels']}",
                                fontsize=8,
                            )
                    fig.suptitle(f"{date} window {window_id} | land-use-kept controls")
                    image = OUT / f"{date}_{window_id}_page{start // 4 + 1}.png"
                    fig.savefig(image, dpi=140)
                    plt.close(fig)
                    pages.append(str(image))
            report["windows"].append(
                {
                    "date": date,
                    "window_id": window_id,
                    "stratum_totals": totals,
                    "samples": [
                        {
                            key: (value.__geo_interface__ if key == "geometry" else value)
                            for key, value in item.items()
                        }
                        for item in samples
                    ],
                    "images": pages,
                }
            )
            print(date, window_id, "sampled", len(samples), flush=True)
    result = OUT / "review.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT:", result)


if __name__ == "__main__":
    main()
