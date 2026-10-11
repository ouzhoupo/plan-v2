#!/usr/bin/env python3
"""Create fixed GIS review pages for 2024-06 C4 deduction groups."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[2]
DATE = "2024-06-01"
GROUP_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_groups"
    / "20260926_204740_877320"
)
SPLIT_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_morphology_split"
    / "20260926_203928_115516"
)
UPSTREAM = ROOT / "diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712"

CODE_LABELS = {
    1: "reservoir_only",
    2: "lake_only",
    3: "reservoir_and_lake",
    4: "linear_only",
    5: "reservoir_and_linear",
}
TARGETS_PER_CODE = {1: 5, 2: 3, 3: 4, 4: 5, 5: 3}
CODE_COLORS = {
    1: (0.95, 0.25, 0.20),
    2: (0.95, 0.65, 0.10),
    3: (0.65, 0.20, 0.80),
    4: (0.10, 0.65, 0.95),
    5: (0.10, 0.75, 0.35),
}


def stretch(band):
    values = band[np.isfinite(band)]
    if values.size == 0:
        return np.zeros_like(band, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    return np.nan_to_num(np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)) ** 0.75


def plot_boundary(axis, geom, color="yellow", linewidth=1.3):
    polygons = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for polygon in polygons:
        if polygon.is_empty:
            continue
        x, y = polygon.exterior.xy
        axis.plot(x, y, color=color, linewidth=linewidth)


def spatial_pick(candidates, count, selected, min_distance=800.0):
    for item in candidates:
        if len(selected) >= count:
            break
        point = item["geometry"].representative_point()
        if all(point.distance(old["geometry"].representative_point()) >= min_distance for old in selected):
            selected.append(item)
    if len(selected) < count:
        for item in candidates:
            if len(selected) >= count:
                break
            if item not in selected:
                item["distance_relaxed"] = True
                selected.append(item)
    return selected


def choose_targets(records):
    chosen = []
    for code, count in TARGETS_PER_CODE.items():
        items = [row for row in records if row["code"] == code]
        if not items:
            continue
        items.sort(key=lambda row: (-row["area_m2"], row["component_id"]))
        selected = [{**items[0], "selection_reason": "largest_control"}]
        pond = [row for row in items[1:] if 500 <= row["area_m2"] <= 20_000]
        pond.sort(key=lambda row: (row["area_m2"], row["component_id"]))
        if pond:
            positions = sorted(set([0, len(pond) // 2, len(pond) - 1]))
            pond_candidates = [
                {**pond[index], "selection_reason": "pond_scale_area_stratum"}
                for index in positions
            ]
            remaining = [
                {**row, "selection_reason": "pond_scale_spatial_fill"}
                for row in pond
                if row not in [pond[index] for index in positions]
            ]
            spatial_pick(pond_candidates + remaining, min(count, 4), selected)
        mid = [
            {**row, "selection_reason": "mid_scale_control"}
            for row in items[1:]
            if 20_000 < row["area_m2"] <= 500_000
        ]
        spatial_pick(mid, count, selected)
        spatial_pick(
            [{**row, "selection_reason": "fallback_size_rank"} for row in items[1:]],
            count,
            selected,
            min_distance=400.0,
        )
        chosen.extend(selected[:count])
    return chosen


def read_window(target, membership_path, pre_path, probability_path, s2_path):
    point = target["geometry"].representative_point()
    half = 500.0
    requested = (point.x - half, point.y - half, point.x + half, point.y + half)
    with rasterio.open(membership_path) as reference:
        window = from_bounds(*requested, transform=reference.transform).round_offsets().round_lengths()
        transform = reference.window_transform(window)
        bounds = reference.window_bounds(window)
        height, width = int(window.height), int(window.width)
        membership = reference.read(1, window=window, boundless=True, fill_value=0)
        with rasterio.open(pre_path) as source:
            pre = source.read(1, window=window, boundless=True, fill_value=0) > 0
        with rasterio.open(probability_path) as source:
            with WarpedVRT(
                source,
                crs=reference.crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
            ) as vrt:
                probability = vrt.read(1, masked=True).astype("float32").filled(np.nan)
        with rasterio.open(s2_path) as source:
            with WarpedVRT(
                source,
                crs=reference.crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
            ) as vrt:
                bands = vrt.read([3, 2, 1], masked=True).astype("float32").filled(np.nan)
    target_mask = rasterize(
        [(target["geometry"], 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ) > 0
    rgb = np.dstack([stretch(bands[index]) for index in range(3)])
    return rgb, pre, membership, probability, target_mask, bounds


def main() -> int:
    source_summary = json.loads((GROUP_RESULT / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise RuntimeError("Deduction group result is incomplete")
    membership_path = GROUP_RESULT / "deduction_group_membership.tif"
    pre_path = SPLIT_RESULT / "c4_pre_deduction_no_morph.tif"
    probability_path = UPSTREAM / DATE / "c2_fused_probability.tif"
    s2_path = ROOT / "data/s2/s2_stack_2024-06-01.tif"

    with rasterio.open(membership_path) as source:
        array = source.read(1)
        transform = source.transform
        pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    records = []
    component_id = 0
    for geom_mapping, value in shapes(array, mask=array > 0, transform=transform, connectivity=4):
        code = int(value)
        if code not in CODE_LABELS:
            continue
        component_id += 1
        geom = shape(geom_mapping)
        records.append(
            {
                "component_id": component_id,
                "code": code,
                "group": CODE_LABELS[code],
                "area_m2": float(geom.area),
                "pixels": int(round(float(geom.area) / pixel_area)),
                "geometry": geom,
                "distance_relaxed": False,
            }
        )
    del array
    targets = choose_targets(records)
    if not targets:
        raise RuntimeError("No deduction review targets selected")

    output = (
        ROOT
        / "diagnostics/results/second_round_deduction_group_review"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for page, start in enumerate(range(0, len(targets), 3), start=1):
        subset = targets[start : start + 3]
        fig, axes = plt.subplots(
            len(subset), 4, figsize=(20, 4.7 * len(subset)), squeeze=False, constrained_layout=True
        )
        for row_index, target in enumerate(subset):
            rgb, pre, membership, probability, target_mask, bounds = read_window(
                target, membership_path, pre_path, probability_path, s2_path
            )
            extent = (bounds[0], bounds[2], bounds[1], bounds[3])
            axes[row_index, 0].imshow(rgb, extent=extent, origin="upper")
            plot_boundary(axes[row_index, 0], target["geometry"])
            axes[row_index, 0].set_title(
                f"S2 RGB | {target['group']} | id={target['component_id']}\n"
                f"area={target['area_m2']/10000:.2f} ha | {target['selection_reason']}"
            )

            overlay = np.zeros((*pre.shape, 4), dtype="float32")
            overlay[pre & (membership == 0)] = (0.0, 0.75, 1.0, 0.65)
            overlay[membership > 0] = (1.0, 0.15, 0.10, 0.65)
            overlay[target_mask] = (1.0, 1.0, 0.0, 0.90)
            axes[row_index, 1].imshow(rgb, extent=extent, origin="upper")
            axes[row_index, 1].imshow(overlay, extent=extent, origin="upper")
            axes[row_index, 1].set_title("C4 water | cyan retained, red deducted, yellow target")

            group_rgb = np.zeros((*membership.shape, 3), dtype="float32")
            for code, color in CODE_COLORS.items():
                group_rgb[membership == code] = color
            axes[row_index, 2].imshow(group_rgb, extent=extent, origin="upper")
            plot_boundary(axes[row_index, 2], target["geometry"])
            axes[row_index, 2].set_title("Deduction membership groups")
            axes[row_index, 2].legend(
                handles=[Patch(color=color, label=CODE_LABELS[code]) for code, color in CODE_COLORS.items()],
                fontsize=6,
                loc="lower right",
            )

            image = axes[row_index, 3].imshow(
                probability, extent=extent, origin="upper", cmap="viridis", vmin=0.5, vmax=1.0
            )
            plot_boundary(axes[row_index, 3], target["geometry"])
            axes[row_index, 3].set_title("C2 fused water probability")
            fig.colorbar(image, ax=axes[row_index, 3], fraction=0.045, pad=0.02)
            for axis in axes[row_index]:
                axis.set_aspect("equal")
                axis.tick_params(labelsize=7)
            manifest.append(
                {
                    "component_id": target["component_id"],
                    "code": target["code"],
                    "group": target["group"],
                    "area_m2": target["area_m2"],
                    "pixels": target["pixels"],
                    "selection_reason": target["selection_reason"],
                    "distance_relaxed": target.get("distance_relaxed", False),
                    "page": page,
                    "image": f"review_page_{page:02d}.png",
                    "proxy_label": "unreviewed",
                    "notes": "",
                }
            )
        fig.savefig(output / f"review_page_{page:02d}.png", dpi=150)
        plt.close(fig)
        print(f"Wrote review_page_{page:02d}.png", flush=True)

    csv_rows = [{key: value for key, value in row.items() if key != "image"} for row in manifest]
    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "status": "complete",
        "experiment": "Second-round 2024-06 C4 deduction group visual review",
        "source_group_result": str(GROUP_RESULT),
        "selection": (
            "Per non-empty exclusive deduction group: one largest-area control plus pond-scale, "
            "mid-scale, and spatially separated fallback components. Mechanism review, not accuracy sampling."
        ),
        "component_population": len(records),
        "target_count": len(manifest),
        "page_count": (len(manifest) + 2) // 3,
        "targets": manifest,
        "limits": [
            "Sentinel-2 RGB is visual proxy evidence rather than field truth.",
            "Samples are deliberately stratified and cannot estimate group-wide accuracy.",
            "A colored target is a connected deducted-pixel component, not necessarily a complete pond object.",
            "No water is restored and no DEM volume or production output is changed.",
        ],
    }
    (output / "review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print("STATUS: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
