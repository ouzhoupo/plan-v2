#!/usr/bin/env python3
"""Create a fixed GIS review of the object groups driving 2025 excess area.

The review compares both dates' optical and SAR views with the frozen C4 masks
and 2025 fused probability.  It does not assign truth labels automatically or
modify any model output.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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
OBJECT_RESULT = (
    ROOT
    / "diagnostics/results/two_date_upstream_object_drivers"
    / "20261009_171738_700797"
)
UPSTREAM = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/two_date_upstream_object_driver_review"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
DRIVER_BANDS = (
    "2000_20000_m2",
    "20000_200000_m2",
    "200000_500000_m2",
    "ge_500000_m2",
)
MAX_WINDOW_PIXELS = 600
MARGIN_PIXELS = 30


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str) -> float:
    return float(row[key])


def stretch(band: np.ndarray) -> np.ndarray:
    values = band[np.isfinite(band)]
    if values.size == 0:
        return np.zeros_like(band, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    result = np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)
    return np.nan_to_num(result, nan=0.0).astype("float32") ** 0.75


def select_spatial(rows: list[dict], count: int) -> list[dict]:
    rows = sorted(
        rows,
        key=lambda row: (
            -number(row, "exclusive_to_this_date_area_m2"),
            int(row["object_id"]),
        ),
    )
    chosen: list[dict] = []
    for minimum_distance in (5000.0, 2500.0, 1000.0, 0.0):
        for row in rows:
            if len(chosen) >= min(count, len(rows)):
                break
            if any(old["object_id"] == row["object_id"] for old in chosen):
                continue
            x, y = number(row, "center_x"), number(row, "center_y")
            if all(
                np.hypot(x - number(old, "center_x"), y - number(old, "center_y"))
                >= minimum_distance
                for old in chosen
            ):
                chosen.append(row)
        if len(chosen) >= min(count, len(rows)):
            break
    return chosen


def choose_targets(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        exclusive_fraction = 1.0 - number(row, "other_date_overlap_fraction")
        if row["area_band"] in DRIVER_BANDS and exclusive_fraction >= 0.75:
            if row["dominant_source_name"] == "dual_source":
                groups[f"driver_dual|{row['area_band']}"] .append(row)
            elif row["dominant_source_name"] == "sar_only":
                groups[f"driver_sar|{row['area_band']}"] .append(row)

    targets: list[dict] = []
    for band in DRIVER_BANDS:
        for row in select_spatial(groups.get(f"driver_dual|{band}", []), 2):
            targets.append({**row, "review_group": f"driver_dual|{band}"})
        for row in select_spatial(groups.get(f"driver_sar|{band}", []), 1):
            targets.append({**row, "review_group": f"driver_sar|{band}"})

    stable_pool = [
        row
        for row in rows
        if row["area_band"] in ("2000_20000_m2", "20000_200000_m2")
        and number(row, "other_date_overlap_fraction") >= 0.8
        and row["dominant_source_name"] in ("dual_source", "sar_only")
    ]
    for source in ("dual_source", "sar_only"):
        pool = [row for row in stable_pool if row["dominant_source_name"] == source]
        for row in select_spatial(pool, 2):
            targets.append({**row, "review_group": f"stable_control|{source}"})

    ids = [row["object_id"] for row in targets]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate review object selected")
    return targets


def display_window(
    component_slice: tuple[slice, slice],
    object_mask: np.ndarray,
    other_water: np.ndarray,
    height: int,
    width: int,
) -> tuple[Window, str]:
    row0, row1 = component_slice[0].start, component_slice[0].stop
    col0, col1 = component_slice[1].start, component_slice[1].stop
    box_height, box_width = row1 - row0, col1 - col0
    mode = "full_object_bbox"
    if box_height > MAX_WINDOW_PIXELS or box_width > MAX_WINDOW_PIXELS:
        local_exclusive = object_mask & ~other_water[row0:row1, col0:col1]
        local_labels, local_count = ndimage.label(
            local_exclusive,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
        )
        if local_count:
            counts = np.bincount(local_labels.ravel())
            largest = int(np.argmax(counts[1:]) + 1)
            ys, xs = np.nonzero(local_labels == largest)
        else:
            ys, xs = np.nonzero(object_mask)
        center_row = row0 + int(np.median(ys))
        center_col = col0 + int(np.median(xs))
        half = MAX_WINDOW_PIXELS // 2
        row0, row1 = max(0, center_row - half), min(height, center_row + half)
        col0, col1 = max(0, center_col - half), min(width, center_col + half)
        mode = "local_crop_largest_exclusive_patch"
    else:
        row0, row1 = max(0, row0 - MARGIN_PIXELS), min(height, row1 + MARGIN_PIXELS)
        col0, col1 = max(0, col0 - MARGIN_PIXELS), min(width, col1 + MARGIN_PIXELS)
    return Window(col0, row0, col1 - col0, row1 - row0), mode


def read_rgb(source, reference, window: Window) -> np.ndarray:
    transform = reference.window_transform(window)
    with WarpedVRT(
        source,
        crs=reference.crs,
        transform=transform,
        width=int(window.width),
        height=int(window.height),
        resampling=Resampling.nearest,
    ) as vrt:
        bands = vrt.read([3, 2, 1], masked=True).astype("float32").filled(np.nan)
    return np.dstack([stretch(band) for band in bands])


def read_sar_vv(source, reference, window: Window) -> np.ndarray:
    transform = reference.window_transform(window)
    with WarpedVRT(
        source,
        crs=reference.crs,
        transform=transform,
        width=int(window.width),
        height=int(window.height),
        resampling=Resampling.nearest,
    ) as vrt:
        vv = vrt.read(1, masked=True).astype("float32").filled(np.nan)
    return stretch(vv)


def main() -> int:
    summary = json.loads((OBJECT_RESULT / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError("Object-driver audit is incomplete")
    rows = read_csv(OBJECT_RESULT / f"objects_{DATE_25}.csv")
    targets = choose_targets(rows)
    if len(targets) != 16:
        raise RuntimeError(f"Expected 16 review targets, got {len(targets)}")

    water25_path = UPSTREAM / DATE_25 / "c4_water_mask.tif"
    water24_path = UPSTREAM / DATE_24 / "c4_water_mask.tif"
    probability25_path = UPSTREAM / DATE_25 / "c2_fused_probability.tif"
    with rasterio.open(water25_path) as water25_source:
        water25 = water25_source.read(1) > 0
    labels, count = ndimage.label(
        water25,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    slices = ndimage.find_objects(labels)
    if count != int(summary["dates"][DATE_25]["objects"]):
        raise RuntimeError("2025 component count mismatch")

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest: list[dict] = []
    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(water25_path))
        water24_source = stack.enter_context(rasterio.open(water24_path))
        probability25_source = stack.enter_context(rasterio.open(probability25_path))
        s2_24 = stack.enter_context(rasterio.open(ROOT / f"data/s2/s2_stack_{DATE_24}.tif"))
        s2_25 = stack.enter_context(rasterio.open(ROOT / f"data/s2/s2_stack_{DATE_25}.tif"))
        s1_24 = stack.enter_context(rasterio.open(ROOT / f"data/s1/s1_stack_{DATE_24}.tif"))
        s1_25 = stack.enter_context(rasterio.open(ROOT / f"data/s1/s1_stack_{DATE_25}.tif"))
        for source in (water24_source, probability25_source):
            if (
                source.shape != reference.shape
                or source.crs != reference.crs
                or not source.transform.almost_equals(reference.transform)
            ):
                raise RuntimeError(f"Grid mismatch: {source.name}")
        water24 = water24_source.read(1) > 0

        for page, start in enumerate(range(0, len(targets), 2), start=1):
            subset = targets[start : start + 2]
            fig, axes = plt.subplots(
                len(subset), 6,
                figsize=(29, 5.2 * len(subset)),
                squeeze=False,
                constrained_layout=True,
            )
            for row_index, target in enumerate(subset):
                object_id = int(target["object_id"])
                component_slice = slices[object_id - 1]
                full_local = labels[component_slice] == object_id
                window, crop_mode = display_window(
                    component_slice,
                    full_local,
                    water24,
                    reference.height,
                    reference.width,
                )
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                water24_view = water24[row0:row1, col0:col1]
                rgb24 = read_rgb(s2_24, reference, window)
                rgb25 = read_rgb(s2_25, reference, window)
                vv24 = read_sar_vv(s1_24, reference, window)
                vv25 = read_sar_vv(s1_25, reference, window)
                probability25 = probability25_source.read(1, window=window)

                exclusive_view = object_view & ~water24_view
                for axis, rgb, title in (
                    (axes[row_index, 0], rgb24, f"{DATE_24} S2 RGB"),
                    (axes[row_index, 1], rgb25, f"{DATE_25} S2 RGB"),
                ):
                    axis.imshow(rgb)
                    axis.contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.8)
                    axis.contour(exclusive_view.astype("uint8"), levels=[0.5], colors="red", linewidths=0.7)
                    axis.set_title(title)

                overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                overlay[water24_view] = (0.0, 0.85, 1.0, 0.65)
                overlay[object_view] = (1.0, 0.15, 0.1, 0.65)
                overlay[water24_view & object_view] = (1.0, 0.85, 0.0, 0.8)
                axes[row_index, 2].imshow(rgb25)
                axes[row_index, 2].imshow(overlay)
                axes[row_index, 2].set_title("Cyan: 2024 water | red: 2025 object | yellow: overlap")

                axes[row_index, 3].imshow(vv24, cmap="gray", vmin=0, vmax=1)
                axes[row_index, 3].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.8)
                axes[row_index, 3].set_title(f"{DATE_24} S1 VV (local stretch)")
                axes[row_index, 4].imshow(vv25, cmap="gray", vmin=0, vmax=1)
                axes[row_index, 4].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.8)
                axes[row_index, 4].set_title(f"{DATE_25} S1 VV (local stretch)")

                image = axes[row_index, 5].imshow(probability25, cmap="viridis", vmin=0, vmax=1)
                axes[row_index, 5].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.8)
                axes[row_index, 5].set_title(
                    f"2025 fused P | mean={number(target, 'probability_mean'):.3f}\n"
                    f"dominant={target['dominant_source_name']}"
                )
                fig.colorbar(image, ax=axes[row_index, 5], shrink=0.65)

                fig.text(
                    0.01,
                    1.0 - (row_index + 0.08) / len(subset),
                    f"id={object_id} | {target['review_group']} | area={number(target, 'area_m2')/10000:.2f} ha | "
                    f"2024 overlap={number(target, 'other_date_overlap_fraction'):.1%} | {crop_mode}",
                    fontsize=10,
                    va="top",
                )
                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                manifest.append({
                    **target,
                    "page": page,
                    "image": f"review_page_{page:02d}.png",
                    "crop_mode": crop_mode,
                    "displayed_object_pixels": int(object_view.sum()),
                    "displayed_exclusive_pixels": int(exclusive_view.sum()),
                    "proxy_label": "unreviewed",
                    "water_evidence_2024": "unreviewed",
                    "water_evidence_2025": "unreviewed",
                    "notes": "",
                })
            fig.legend(
                handles=[
                    Patch(color=(0.0, 0.85, 1.0), label="2024 C4 water"),
                    Patch(color=(1.0, 0.15, 0.1), label="2025 target object"),
                    Patch(color=(1.0, 0.85, 0.0), label="cross-date overlap"),
                ],
                loc="lower center",
                ncol=3,
            )
            fig.savefig(output / f"review_page_{page:02d}.png", dpi=140)
            plt.close(fig)
            print(f"Wrote review_page_{page:02d}.png", flush=True)

    write_fields = list(manifest[0])
    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=write_fields)
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "complete",
        "experiment": "Two-date upstream object-driver visual review",
        "targets": len(manifest),
        "pages": (len(manifest) + 1) // 2,
        "selection": (
            "2025 objects with >=75% date-exclusive support: two dual-source and one SAR-only "
            "object per major area band, plus two high-overlap controls for each source class."
        ),
        "limits": [
            "The sample is mechanism-stratified and cannot estimate population accuracy.",
            "A 2025-exclusive raster object can represent real hydrological change, segmentation change, or classification error.",
            "Large objects use a local crop around the largest exclusive patch; review pages do not always show the full object.",
            "S1 panels use local display stretching and are qualitative, not calibrated backscatter comparisons.",
            "Visual labels are proxy evidence, not field truth or survey membership.",
            "No model input, threshold, mask, object, DEM volume, or production result is modified.",
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
