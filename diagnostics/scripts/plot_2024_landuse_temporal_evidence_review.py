#!/usr/bin/env python3
"""Plot a stratified GIS review of temporal evidence for 2024 land-use rejections."""

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
AUDIT = ROOT / "diagnostics/results/landuse_temporal_evidence_2024/20261010_173122_234024"
AREA_GATE = (
    ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
    / "paired_reference_downstream/2024-06-01/shared_reference_after_area_gate.tif"
)
PREVIOUS_MASK = (
    ROOT / "diagnostics/results/shared_sar_calibration_upstream/20261009_193736_367365"
    / "2024-05-21/shared_sar_shared_alpha_c4_water_mask.tif"
)
FUTURE_MASK = (
    ROOT / "diagnostics/results/new_sar_evidence_gate_upstream/20261010_134344_780095"
    / "clear_low_no_aux_gate_c4_water_mask.tif"
)
S2 = {
    "2024-05-21": ROOT / "data/s2/s2_stack_2024-05-21.tif",
    "2024-06-01": ROOT / "data/s2/s2_stack_2024-06-01.tif",
    "2025-05-21": ROOT / "data/s2/s2_stack_2025-05-21.tif",
}
S1_TARGET = ROOT / "data/s1/s1_stack_2024-06-01.tif"
OUTPUT_ROOT = ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024"
MARGIN = 30
MAX_WINDOW = 550


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "area_m2", "center_x", "center_y", "previous_near", "future_near",
        "scl_target_2024_06", "previous_exact", "future_exact",
    )
    for row in rows:
        row["object_id"] = int(row["object_id"])
        for key in numeric:
            row[key] = float(row[key])
    return rows


def branch(row: dict) -> str | None:
    previous = row["previous_near"]
    future = row["future_near"]
    target_scl = row["scl_target_2024_06"]
    if previous >= 0.25 and future >= 0.25 and target_scl >= 0.25:
        return "triple_strong"
    if previous >= 0.25 and future >= 0.25 and target_scl < 0.01:
        return "model_persistent_no_target_scl"
    if previous >= 0.25 and future < 0.01 and target_scl < 0.01:
        return "previous_only_operational"
    if previous < 0.01 and (future >= 0.25 or target_scl >= 0.25):
        return "other_support_without_previous"
    if previous < 0.01 and future < 0.01 and target_scl < 0.01:
        return "no_strong_temporal_support"
    return None


def choose_spatial(pool: list[dict], count: int, used: list[dict]) -> list[dict]:
    ordered = sorted(pool, key=lambda row: (-row["area_m2"], row["object_id"]))
    if not ordered:
        return []
    positions = sorted(set((0, len(ordered) // 3, 2 * len(ordered) // 3, len(ordered) - 1)))
    candidates = [ordered[index] for index in positions] + ordered
    selected: list[dict] = []
    for minimum in (3000.0, 1500.0, 500.0, 0.0):
        for row in candidates:
            if len(selected) >= min(count, len(pool)):
                break
            if any(row["object_id"] == old["object_id"] for old in (*used, *selected)):
                continue
            if all(
                np.hypot(row["center_x"] - old["center_x"], row["center_y"] - old["center_y"])
                >= minimum
                for old in (*used, *selected)
            ):
                item = dict(row)
                item["selection_min_distance_m"] = minimum
                selected.append(item)
        if len(selected) >= min(count, len(pool)):
            break
    return selected[:count]


def choose_targets(rows: list[dict]) -> list[dict]:
    names = (
        "triple_strong",
        "model_persistent_no_target_scl",
        "previous_only_operational",
        "other_support_without_previous",
        "no_strong_temporal_support",
    )
    pools = {name: [] for name in names}
    for row in rows:
        name = branch(row)
        if name:
            pools[name].append(row)
    targets: list[dict] = []
    for name in names:
        for row in choose_spatial(pools[name], 3, targets):
            targets.append({**row, "review_group": name})
    if len(targets) != 15:
        raise RuntimeError(f"Expected 15 targets, got {len(targets)}")
    return targets


def stretch(array: np.ndarray) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    scaled = np.clip((array - low) / max(float(high - low), 1e-6), 0, 1)
    return np.nan_to_num(scaled, nan=0.0).astype("float32") ** 0.75


def aligned_vrt(stack: ExitStack, path: Path, reference):
    source = stack.enter_context(rasterio.open(path))
    return stack.enter_context(WarpedVRT(
        source, crs=reference.crs, transform=reference.transform,
        width=reference.width, height=reference.height, resampling=Resampling.nearest,
    ))


def read_rgb(source, window: Window) -> np.ndarray:
    bands = source.read([3, 2, 1], window=window, masked=True).astype("float32").filled(np.nan)
    return np.dstack([stretch(band) for band in bands])


def display_window(component_slice, height: int, width: int) -> tuple[Window, str]:
    row0, row1 = component_slice[0].start, component_slice[0].stop
    col0, col1 = component_slice[1].start, component_slice[1].stop
    mode = "full_object_bbox"
    if row1 - row0 > MAX_WINDOW or col1 - col0 > MAX_WINDOW:
        center_row, center_col = (row0 + row1) // 2, (col0 + col1) // 2
        half = MAX_WINDOW // 2
        row0, row1 = max(0, center_row - half), min(height, center_row + half)
        col0, col1 = max(0, center_col - half), min(width, center_col + half)
        mode = "center_crop"
    else:
        row0, row1 = max(0, row0 - MARGIN), min(height, row1 + MARGIN)
        col0, col1 = max(0, col0 - MARGIN), min(width, col1 + MARGIN)
    return Window(col0, row0, col1 - col0, row1 - row0), mode


def main() -> int:
    audit = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("Temporal evidence audit is incomplete")
    rows = read_rows(AUDIT / "rejected_objects_temporal_evidence.csv")
    targets = choose_targets(rows)

    with rasterio.open(AREA_GATE) as source:
        area_gate = source.read(1) > 0
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    if count != audit["checks"]["area_gate_objects"]:
        raise RuntimeError("Area-gate component count mismatch")
    slices = ndimage.find_objects(labels)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest: list[dict] = []
    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(AREA_GATE))
        previous = aligned_vrt(stack, PREVIOUS_MASK, reference)
        future = aligned_vrt(stack, FUTURE_MASK, reference)
        s2 = {date: aligned_vrt(stack, path, reference) for date, path in S2.items()}
        s1 = aligned_vrt(stack, S1_TARGET, reference)

        for page, start in enumerate(range(0, len(targets), 3), start=1):
            subset = targets[start:start + 3]
            fig, axes = plt.subplots(
                len(subset), 6, figsize=(29, 5.1 * len(subset)), squeeze=False,
                constrained_layout=True,
            )
            for row_index, target in enumerate(subset):
                object_id = target["object_id"]
                component_slice = slices[object_id - 1]
                window, crop_mode = display_window(component_slice, reference.height, reference.width)
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                rgb = {date: read_rgb(source, window) for date, source in s2.items()}
                previous_view = previous.read(1, window=window) > 0
                future_view = future.read(1, window=window) > 0
                target_scl = s2["2024-06-01"].read(7, window=window) == 6
                vv = stretch(s1.read(1, window=window, masked=True).astype("float32").filled(np.nan))

                for column, date in enumerate(("2024-05-21", "2024-06-01", "2025-05-21")):
                    axes[row_index, column].imshow(rgb[date])
                    axes[row_index, column].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                    axes[row_index, column].set_title(f"{date} S2 RGB")

                axes[row_index, 3].imshow(vv, cmap="gray", vmin=0, vmax=1)
                axes[row_index, 3].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                axes[row_index, 3].set_title("2024-06-01 S1 VV (local stretch)")

                overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                overlay[previous_view & object_view] = (0.0, 0.8, 1.0, 0.72)
                overlay[target_scl & object_view] = (0.0, 0.9, 0.25, 0.78)
                overlay[previous_view & target_scl & object_view] = (1.0, 0.85, 0.0, 0.85)
                axes[row_index, 4].imshow(rgb["2024-06-01"])
                axes[row_index, 4].imshow(overlay)
                axes[row_index, 4].set_title("Cyan previous model | green target SCL | yellow overlap")

                future_overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                future_overlay[future_view & object_view] = (0.85, 0.05, 0.85, 0.75)
                axes[row_index, 5].imshow(rgb["2025-05-21"])
                axes[row_index, 5].imshow(future_overlay)
                axes[row_index, 5].set_title("Magenta: retrospective 2025 model support")

                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                fig.text(
                    0.01, 1.0 - (row_index + 0.08) / len(subset),
                    f"id={object_id} | {target['review_group']} | area={target['area_m2']/10000:.2f} ha | "
                    f"prev-near={target['previous_near']:.1%} | SCL={target['scl_target_2024_06']:.1%} | "
                    f"future-near={target['future_near']:.1%}",
                    fontsize=9.5, va="top",
                )
                manifest.append({
                    **target,
                    "page": page,
                    "image": f"review_page_{page:02d}.png",
                    "crop_mode": crop_mode,
                    "proxy_label": "unreviewed",
                    "temporal_evidence_interpretation": "unreviewed",
                    "notes": "",
                })
            fig.legend(handles=[
                Patch(color=(0.0, 0.8, 1.0), label="previous-date model support"),
                Patch(color=(0.0, 0.9, 0.25), label="target-date SCL water"),
                Patch(color=(0.85, 0.05, 0.85), label="future model support (diagnostic only)"),
            ], loc="lower center", ncol=3)
            fig.savefig(output / f"review_page_{page:02d}.png", dpi=140)
            plt.close(fig)
            print(f"Wrote review_page_{page:02d}.png", flush=True)

    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "complete",
        "experiment": "2024 land-use rejection temporal-evidence GIS review",
        "targets": len(manifest),
        "pages": (len(manifest) + 2) // 3,
        "selection": (
            "Three spatially separated objects from each of five evidence strata: triple strong; "
            "previous/future model persistence without target SCL; previous-only operational; "
            "other support without previous; and no strong temporal support."
        ),
        "limits": [
            "Samples are mechanism-stratified and cannot estimate population accuracy.",
            "2025 support is retrospective evidence and cannot be used as a 2024 operational input.",
            "Temporal model persistence can repeat systematic false positives and is not field truth.",
            "Local SAR stretching is qualitative; no threshold is inferred from display brightness.",
            "No object is restored and no DEM volume or production output is modified.",
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
