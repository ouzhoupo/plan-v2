#!/usr/bin/env python3
"""Plot complete pre-deduction components for full-removal and partial-cut review."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DATE = "2024-06-01"
COMPONENT_RESULT = (
    ROOT
    / "diagnostics/results/second_round_pre_deduction_components"
    / "20260926_212547_194707"
)
SPLIT_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_morphology_split"
    / "20260926_203928_115516"
)
GROUP_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_groups"
    / "20260926_204740_877320"
)
UPSTREAM = ROOT / "diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712"

BRANCH_COUNTS = {
    "fully_removed_reservoir_target": 10,
    "partially_cut_reservoir_target": 8,
    "partially_cut_linear_target": 6,
}
CODE_COLORS = {
    1: (0.95, 0.25, 0.20),
    2: (0.95, 0.65, 0.10),
    3: (0.65, 0.20, 0.80),
    4: (0.10, 0.65, 0.95),
    5: (0.10, 0.75, 0.35),
}


def boolean(value):
    return str(value).strip().lower() == "true"


def number(row, key):
    return float(row[key])


def stretch(band):
    values = band[np.isfinite(band)]
    if values.size == 0:
        return np.zeros_like(band, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    return np.nan_to_num(np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)) ** 0.75


def select_spatial(rows, count):
    rows = sorted(rows, key=lambda row: (-number(row, "area_m2"), int(row["component_id"])))
    if not rows:
        return []
    seeds = [rows[0]]
    positions = sorted(set([len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1]))
    candidates = [rows[index] for index in positions] + rows[1:]
    for minimum_distance in (1500.0, 800.0, 0.0):
        for row in candidates:
            if len(seeds) >= count:
                break
            if row in seeds:
                continue
            x, y = number(row, "centroid_x"), number(row, "centroid_y")
            if all(
                ((x - number(old, "centroid_x")) ** 2 + (y - number(old, "centroid_y")) ** 2) ** 0.5
                >= minimum_distance
                for old in seeds
            ):
                selected = dict(row)
                selected["distance_relaxed"] = minimum_distance < 1500.0
                seeds.append(selected)
        if len(seeds) >= count:
            break
    return seeds[:count]


def main() -> int:
    summary = json.loads((COMPONENT_RESULT / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError("Pre-deduction component audit is incomplete")
    with (COMPONENT_RESULT / "all_pre_deduction_components.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    target_range = [row for row in rows if 500 <= number(row, "area_m2") <= 500_000]
    branches = {
        "fully_removed_reservoir_target": [
            row
            for row in target_range
            if boolean(row["fully_removed"]) and number(row, "reservoir_involved_pixels") > 0
        ],
        "partially_cut_reservoir_target": [
            row
            for row in target_range
            if boolean(row["partially_removed"])
            and number(row, "reservoir_involved_pixels") > 0
            and row["dominant_deduction_group"].startswith("reservoir")
        ],
        "partially_cut_linear_target": [
            row
            for row in target_range
            if boolean(row["partially_removed"])
            and number(row, "linear_involved_pixels") > 0
            and row["dominant_deduction_group"] == "linear_only"
        ],
    }
    targets = []
    for branch, count in BRANCH_COUNTS.items():
        selected = select_spatial(branches[branch], count)
        for row in selected:
            targets.append({**row, "review_branch": branch})

    pre_path = SPLIT_RESULT / "c4_pre_deduction_no_morph.tif"
    membership_path = GROUP_RESULT / "deduction_group_membership.tif"
    probability_path = UPSTREAM / DATE / "c2_fused_probability.tif"
    s2_path = ROOT / "data/s2/s2_stack_2024-06-01.tif"
    with rasterio.open(pre_path) as pre_source:
        pre = pre_source.read(1) > 0
    labels, component_count = ndimage.label(
        pre, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8")
    )
    slices = ndimage.find_objects(labels)
    if component_count != int(summary["all_components"]["objects"]):
        raise RuntimeError("Connected-component count does not reproduce the audit")

    output = (
        ROOT
        / "diagnostics/results/second_round_pre_deduction_component_review"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    with rasterio.open(pre_path) as reference, rasterio.open(membership_path) as member_source:
        for page, start in enumerate(range(0, len(targets), 3), start=1):
            subset = targets[start : start + 3]
            fig, axes = plt.subplots(
                len(subset), 4, figsize=(20, 4.7 * len(subset)), squeeze=False, constrained_layout=True
            )
            for row_index, target in enumerate(subset):
                component_id = int(target["component_id"])
                component_slice = slices[component_id - 1]
                if component_slice is None:
                    raise RuntimeError(f"Missing component slice: {component_id}")
                row0 = max(0, component_slice[0].start - 30)
                row1 = min(reference.height, component_slice[0].stop + 30)
                col0 = max(0, component_slice[1].start - 30)
                col1 = min(reference.width, component_slice[1].stop + 30)
                window = Window(col0, row0, col1 - col0, row1 - row0)
                transform = reference.window_transform(window)
                bounds = reference.window_bounds(window)
                extent = (bounds[0], bounds[2], bounds[1], bounds[3])
                target_mask = labels[row0:row1, col0:col1] == component_id
                membership = member_source.read(1, window=window)
                removed = target_mask & (membership > 0)
                surviving = target_mask & (membership == 0)
                height, width = target_mask.shape
                with rasterio.open(probability_path) as probability_source:
                    with WarpedVRT(
                        probability_source,
                        crs=reference.crs,
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=Resampling.nearest,
                    ) as vrt:
                        probability = vrt.read(1, masked=True).astype("float32").filled(np.nan)
                with rasterio.open(s2_path) as s2_source:
                    with WarpedVRT(
                        s2_source,
                        crs=reference.crs,
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=Resampling.nearest,
                    ) as vrt:
                        bands = vrt.read([3, 2, 1], masked=True).astype("float32").filled(np.nan)
                rgb = np.dstack([stretch(bands[index]) for index in range(3)])

                axes[row_index, 0].imshow(rgb, extent=extent, origin="upper")
                axes[row_index, 0].contour(
                    target_mask.astype("uint8"), levels=[0.5], colors="yellow", linewidths=1.2,
                    extent=extent, origin="upper"
                )
                axes[row_index, 0].set_title(
                    f"S2 RGB | {target['review_branch']} | id={component_id}\n"
                    f"area={number(target, 'area_m2')/10000:.2f} ha | removed={number(target, 'removed_fraction'):.1%}"
                )

                overlay = np.zeros((*target_mask.shape, 4), dtype="float32")
                overlay[surviving] = (0.0, 0.75, 1.0, 0.75)
                overlay[removed] = (1.0, 0.15, 0.10, 0.75)
                axes[row_index, 1].imshow(rgb, extent=extent, origin="upper")
                axes[row_index, 1].imshow(overlay, extent=extent, origin="upper")
                axes[row_index, 1].set_title("Complete component | cyan survives, red deducted")

                group_rgb = np.zeros((*membership.shape, 3), dtype="float32")
                for code, color in CODE_COLORS.items():
                    group_rgb[target_mask & (membership == code)] = color
                axes[row_index, 2].imshow(group_rgb, extent=extent, origin="upper")
                axes[row_index, 2].contour(
                    target_mask.astype("uint8"), levels=[0.5], colors="yellow", linewidths=1.0,
                    extent=extent, origin="upper"
                )
                axes[row_index, 2].set_title("Deduction groups inside complete component")
                axes[row_index, 2].legend(
                    handles=[Patch(color=color, label=str(code)) for code, color in CODE_COLORS.items()],
                    fontsize=6, loc="lower right"
                )

                image = axes[row_index, 3].imshow(
                    probability, extent=extent, origin="upper", cmap="viridis", vmin=0.5, vmax=1.0
                )
                axes[row_index, 3].contour(
                    target_mask.astype("uint8"), levels=[0.5], colors="yellow", linewidths=1.0,
                    extent=extent, origin="upper"
                )
                axes[row_index, 3].set_title("C2 fused water probability")
                fig.colorbar(image, ax=axes[row_index, 3], fraction=0.045, pad=0.02)
                for axis in axes[row_index]:
                    axis.set_aspect("equal")
                    axis.tick_params(labelsize=7)
                manifest.append(
                    {
                        "component_id": component_id,
                        "review_branch": target["review_branch"],
                        "area_m2": number(target, "area_m2"),
                        "removed_fraction": number(target, "removed_fraction"),
                        "dominant_deduction_group": target["dominant_deduction_group"],
                        "bbox_fill_fraction": number(target, "bbox_fill_fraction"),
                        "bbox_elongation": number(target, "bbox_elongation"),
                        "page": page,
                        "image": f"review_page_{page:02d}.png",
                        "proxy_label": "unreviewed",
                        "notes": "",
                    }
                )
            fig.savefig(output / f"review_page_{page:02d}.png", dpi=150)
            plt.close(fig)
            print(f"Wrote review_page_{page:02d}.png", flush=True)

    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "complete",
        "experiment": "Second-round complete pre-deduction component visual review",
        "source_component_result": str(COMPONENT_RESULT),
        "branch_population": {name: len(items) for name, items in branches.items()},
        "selection": "Largest plus area-stratified spatially separated components per mechanism branch.",
        "target_count": len(manifest),
        "page_count": (len(manifest) + 2) // 3,
        "targets": manifest,
        "limits": [
            "Visual labels are remote-sensing proxy evidence, not field truth.",
            "Stratified samples do not estimate population-wide accuracy.",
            "No component is restored and no DEM volume is calculated.",
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
