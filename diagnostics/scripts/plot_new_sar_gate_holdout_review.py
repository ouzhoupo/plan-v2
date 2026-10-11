#!/usr/bin/env python3
"""Create an independent spatial review for the conservative new-SAR gate.

Targets are selected from connected water objects in the shared-SAR/shared-alpha
baseline.  They must be changed by the conservative gate and spatially separated
from the 16 objects previously used to interpret the mechanism.  This script
only creates review evidence; it does not modify a model output or calculate
downstream volume.
"""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
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
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
SHARED = (
    ROOT / "diagnostics/results/shared_sar_calibration_upstream/20261009_193736_367365"
    / DATE_25 / "shared_sar_shared_alpha_c4_water_mask.tif"
)
EXPERIMENT = ROOT / "diagnostics/results/new_sar_evidence_gate_upstream/20261010_134344_780095"
CONSERVATIVE = EXPERIMENT / "clear_low_no_aux_gate_c4_water_mask.tif"
GATE = EXPERIMENT / "clear_low_no_aux_gate_gate_mask.tif"
PREVIOUS_REVIEW = (
    ROOT / "diagnostics/results/two_date_upstream_object_driver_review/20261009_172614_247553"
    / "review_manifest.csv"
)
S2_24 = ROOT / f"data/s2/s2_stack_{DATE_24}.tif"
S2_25 = ROOT / f"data/s2/s2_stack_{DATE_25}.tif"
S1_25 = ROOT / f"data/s1/s1_stack_{DATE_25}.tif"
LANDUSE = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
OUTPUT_ROOT = ROOT / "diagnostics/results/new_sar_gate_holdout_review"
MAX_WINDOW_PIXELS = 600
MARGIN_PIXELS = 30


def stretch(array: np.ndarray) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    scaled = np.clip((array - low) / max(float(high - low), 1e-6), 0, 1)
    return np.nan_to_num(scaled, nan=0.0).astype("float32") ** 0.75


def aligned_vrt(stack: ExitStack, path: Path, reference, resampling=Resampling.nearest):
    source = stack.enter_context(rasterio.open(path))
    return stack.enter_context(
        WarpedVRT(
            source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=resampling,
        )
    )


def read_rgb(source, window: Window) -> np.ndarray:
    bands = source.read([3, 2, 1], window=window, masked=True).astype("float32").filled(np.nan)
    return np.dstack([stretch(band) for band in bands])


def read_previous_centers() -> list[tuple[float, float]]:
    with PREVIOUS_REVIEW.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 previous review objects, got {len(rows)}")
    return [(float(row["center_x"]), float(row["center_y"])) for row in rows]


def distance_ok(row: dict, points: list[tuple[float, float]], minimum: float) -> bool:
    return all(np.hypot(row["center_x"] - x, row["center_y"] - y) >= minimum for x, y in points)


def choose_spatial(
    pool: list[dict], count: int, previous: list[tuple[float, float]], chosen: list[dict]
) -> list[dict]:
    ordered = sorted(
        pool,
        key=lambda row: (-row["removed_pixels"], -row["area_m2"], row["object_id"]),
    )
    selected: list[dict] = []
    used_ids = {int(row["object_id"]) for row in chosen}
    for previous_distance, within_distance in ((2000.0, 3000.0), (1500.0, 1500.0), (1000.0, 500.0)):
        occupied = [(row["center_x"], row["center_y"]) for row in (*chosen, *selected)]
        for row in ordered:
            if len(selected) >= min(count, len(ordered)):
                break
            if int(row["object_id"]) in used_ids or any(
                int(row["object_id"]) == int(old["object_id"]) for old in selected
            ):
                continue
            if not distance_ok(row, previous, previous_distance):
                continue
            if not distance_ok(row, occupied, within_distance):
                continue
            item = dict(row)
            item["selection_previous_min_distance_m"] = previous_distance
            item["selection_within_min_distance_m"] = within_distance
            selected.append(item)
            occupied.append((item["center_x"], item["center_y"]))
        if len(selected) >= min(count, len(ordered)):
            break
    return selected[:count]


def choose_targets(rows: list[dict], previous: list[tuple[float, float]]) -> list[dict]:
    bands = (
        ("low_removed_5_25pct", 0.05, 0.25),
        ("medium_removed_25_75pct", 0.25, 0.75),
        ("high_removed_ge_75pct", 0.75, 1.01),
    )
    chosen: list[dict] = []
    for name, low, high in bands:
        pool = [
            row for row in rows
            if 500.0 <= row["area_m2"] < 500000.0
            and low <= row["removed_fraction"] < high
        ]
        for row in choose_spatial(pool, 3, previous, chosen):
            chosen.append({**row, "review_group": name})

    large = [row for row in rows if row["area_m2"] >= 500000.0]
    for row in choose_spatial(large, 3, previous, chosen):
        chosen.append({**row, "review_group": "large_object_ge_50ha"})

    if len(chosen) < 12:
        chosen_ids = {int(row["object_id"]) for row in chosen}
        remaining = [row for row in rows if int(row["object_id"]) not in chosen_ids]
        for row in choose_spatial(remaining, 12 - len(chosen), previous, chosen):
            chosen.append({**row, "review_group": "adaptive_fill"})
    if len(chosen) != 12 or len({row["object_id"] for row in chosen}) != 12:
        raise RuntimeError(f"Expected 12 unique holdout targets, got {len(chosen)}")
    return chosen


def display_window(component_slice, local_removed, height: int, width: int) -> tuple[Window, str]:
    row0, row1 = component_slice[0].start, component_slice[0].stop
    col0, col1 = component_slice[1].start, component_slice[1].stop
    mode = "full_object_bbox"
    if row1 - row0 > MAX_WINDOW_PIXELS or col1 - col0 > MAX_WINDOW_PIXELS:
        local_labels, count = ndimage.label(
            local_removed,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
        )
        counts = np.bincount(local_labels.ravel()) if count else np.array([0])
        largest = int(np.argmax(counts[1:]) + 1) if count else 0
        ys, xs = np.nonzero(local_labels == largest) if count else np.nonzero(local_removed)
        center_row = row0 + int(np.median(ys))
        center_col = col0 + int(np.median(xs))
        half = MAX_WINDOW_PIXELS // 2
        row0, row1 = max(0, center_row - half), min(height, center_row + half)
        col0, col1 = max(0, center_col - half), min(width, center_col + half)
        mode = "local_crop_largest_removed_patch"
    else:
        row0, row1 = max(0, row0 - MARGIN_PIXELS), min(height, row1 + MARGIN_PIXELS)
        col0, col1 = max(0, col0 - MARGIN_PIXELS), min(width, col1 + MARGIN_PIXELS)
    return Window(col0, row0, col1 - col0, row1 - row0), mode


def main() -> int:
    previous = read_previous_centers()
    with rasterio.open(SHARED) as reference:
        shared = reference.read(1) > 0
        profile = (reference.shape, reference.crs, reference.transform)
        pixel_area_m2 = abs(float(reference.transform.a * reference.transform.e))
    with rasterio.open(CONSERVATIVE) as source:
        if source.shape != profile[0] or source.crs != profile[1] or not source.transform.almost_equals(profile[2]):
            raise RuntimeError("Conservative gate grid mismatch")
        conservative = source.read(1) > 0

    labels, count = ndimage.label(
        shared,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    slices = ndimage.find_objects(labels)
    object_pixels = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    removed_mask = shared & ~conservative
    removed_pixels = np.bincount(labels[removed_mask], minlength=count + 1)[1:]
    affected_ids = np.flatnonzero(removed_pixels > 0) + 1
    rows: list[dict] = []
    for object_id in affected_ids:
        component_slice = slices[int(object_id) - 1]
        center_row = (component_slice[0].start + component_slice[0].stop - 1) / 2
        center_col = (component_slice[1].start + component_slice[1].stop - 1) / 2
        x, y = rasterio.transform.xy(profile[2], center_row, center_col)
        pixels = int(object_pixels[object_id - 1])
        removed = int(removed_pixels[object_id - 1])
        rows.append({
            "object_id": int(object_id),
            "pixels": pixels,
            "area_m2": pixels * pixel_area_m2,
            "removed_pixels": removed,
            "removed_area_m2": removed * pixel_area_m2,
            "removed_fraction": removed / pixels,
            "center_x": float(x),
            "center_y": float(y),
        })
    targets = choose_targets(rows, previous)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest: list[dict] = []
    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(SHARED))
        gate_source = stack.enter_context(rasterio.open(GATE))
        s2_24 = aligned_vrt(stack, S2_24, reference)
        s2_25 = aligned_vrt(stack, S2_25, reference)
        s1_25 = aligned_vrt(stack, S1_25, reference)
        landuse = aligned_vrt(stack, LANDUSE, reference)

        for page, start in enumerate(range(0, len(targets), 2), start=1):
            subset = targets[start:start + 2]
            fig, axes = plt.subplots(
                len(subset), 6, figsize=(29, 5.4 * len(subset)), squeeze=False,
                constrained_layout=True,
            )
            for row_index, target in enumerate(subset):
                object_id = int(target["object_id"])
                component_slice = slices[object_id - 1]
                local_object = labels[component_slice] == object_id
                local_removed = local_object & removed_mask[component_slice]
                window, crop_mode = display_window(
                    component_slice, local_removed, reference.height, reference.width
                )
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                retained_view = object_view & conservative[row0:row1, col0:col1]
                removed_view = object_view & ~conservative[row0:row1, col0:col1]
                gate_view = gate_source.read(1, window=window) > 0
                rgb24, rgb25 = read_rgb(s2_24, window), read_rgb(s2_25, window)
                vv25 = stretch(s1_25.read(1, window=window, masked=True).astype("float32").filled(np.nan))
                scl = s2_25.read(7, window=window)
                landuse_view = landuse.read(1, window=window)

                for axis, rgb, title in (
                    (axes[row_index, 0], rgb24, f"{DATE_24} S2 RGB"),
                    (axes[row_index, 1], rgb25, f"{DATE_25} S2 RGB"),
                ):
                    axis.imshow(rgb)
                    axis.contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                    axis.set_title(title)

                overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                overlay[retained_view] = (0.0, 0.75, 1.0, 0.65)
                overlay[removed_view] = (1.0, 0.12, 0.08, 0.78)
                axes[row_index, 2].imshow(rgb25)
                axes[row_index, 2].imshow(overlay)
                axes[row_index, 2].set_title("Shared object: cyan retained | red removed")

                axes[row_index, 3].imshow(vv25, cmap="gray", vmin=0, vmax=1)
                axes[row_index, 3].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.8)
                axes[row_index, 3].set_title("2025 S1 VV (local stretch)")

                axes[row_index, 4].imshow(scl, cmap="tab20", vmin=0, vmax=11)
                axes[row_index, 4].contour(removed_view.astype("uint8"), levels=[0.5], colors="red", linewidths=0.9)
                axes[row_index, 4].set_title("2025 SCL | red contour = removed")

                support = np.zeros((*object_view.shape, 3), dtype="float32")
                support[landuse_view == 1] = (0.0, 0.8, 0.3)
                support[gate_view] = (0.75, 0.1, 0.85)
                axes[row_index, 5].imshow(support)
                axes[row_index, 5].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.8)
                axes[row_index, 5].set_title("Green land-use water | purple applied gate")

                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                fig.text(
                    0.01, 1.0 - (row_index + 0.08) / len(subset),
                    f"id={object_id} | {target['review_group']} | area={target['area_m2']/10000:.2f} ha | "
                    f"removed={target['removed_fraction']:.1%} | {crop_mode}",
                    fontsize=10, va="top",
                )
                manifest.append({
                    **target,
                    "page": page,
                    "image": f"review_page_{page:02d}.png",
                    "crop_mode": crop_mode,
                    "displayed_object_pixels": int(object_view.sum()),
                    "displayed_removed_pixels": int(removed_view.sum()),
                    "proxy_label": "unreviewed",
                    "removed_part_interpretation": "unreviewed",
                    "notes": "",
                })
            fig.legend(
                handles=[
                    Patch(color=(0.0, 0.75, 1.0), label="retained by conservative gate"),
                    Patch(color=(1.0, 0.12, 0.08), label="removed by conservative gate"),
                    Patch(color=(0.75, 0.1, 0.85), label="gate applied pixel"),
                ], loc="lower center", ncol=3,
            )
            fig.savefig(output / f"review_page_{page:02d}.png", dpi=140)
            plt.close(fig)
            print(f"Wrote review_page_{page:02d}.png", flush=True)

    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "complete",
        "experiment": "Independent spatial holdout review of conservative new-SAR gate",
        "shared_object_count": int(count),
        "affected_object_count": int(len(affected_ids)),
        "affected_removed_area_km2": float(removed_mask.sum() * pixel_area_m2 / 1e6),
        "target_count": len(manifest),
        "page_count": (len(manifest) + 1) // 2,
        "selection": (
            "Twelve changed shared-baseline objects: three per 5-25%, 25-75%, and >=75% "
            "removed-fraction stratum for 0.05-50 ha objects, plus three >=50 ha controls. "
            "Targets are kept at least 1 km from the centers of the 16 prior review objects."
        ),
        "limits": [
            "This is a mechanism-stratified holdout, not a random accuracy sample.",
            "Spatial center separation reduces direct sample reuse but does not guarantee statistical independence.",
            "Visual labels are GIS proxy evidence rather than field truth or survey membership.",
            "The review covers the 2025 Pihe AOI only and cannot represent Shihe, Hangbuhe, or all Pishihang.",
            "No production model, downstream object result, DEM volume, or parameter is modified.",
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
