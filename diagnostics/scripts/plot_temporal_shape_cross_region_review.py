#!/usr/bin/env python3
"""Plot held-out Shihe and Hangbuhe samples for temporal-shape transfer review."""

from __future__ import annotations

import csv
import json
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import plot_2024_landuse_temporal_evidence_review as base  # noqa: E402

BRIDGE = (
    ROOT / "diagnostics/results/temporal_shape_cross_region_bridge/20261010_194531_518591"
)
REGIONS = ("shihe", "hangbuhe")
PREVIOUS = "2024-05-21"
TARGET = "2024-06-01"
FUTURE = "2025-05-21"
OUTPUT_ROOT = ROOT / "diagnostics/results/temporal_shape_cross_region_review"
S2 = {
    PREVIOUS: ROOT / f"data/s2/s2_stack_{PREVIOUS}.tif",
    TARGET: ROOT / f"data/s2/s2_stack_{TARGET}.tif",
    FUTURE: ROOT / f"data/s2/s2_stack_{FUTURE}.tif",
}
S1_TARGET = ROOT / f"data/s1/s1_stack_{TARGET}.tif"


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def read_rows(region: str) -> list[dict]:
    path = BRIDGE / region / "landuse_rejected_temporal_shape_bridge.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "area_m2", "previous_near_fraction", "future_near_fraction_retrospective",
        "bbox_fill", "elongation", "compactness",
    )
    for row in rows:
        row["object_id"] = int(row["object_id"])
        for key in numeric:
            row[key] = float(row[key])
        row["selected_loose_candidate"] = truth(row["selected_loose_candidate"])
        row["future_fraction_ge_025_retrospective"] = truth(
            row["future_fraction_ge_025_retrospective"]
        )
    return rows


def choose_area_stratified(pool: list[dict], used: list[dict]) -> list[dict]:
    bins = ((2000, 10000, 5000), (10000, 30000, 20000), (30000, 50000, 40000))
    selected = []
    for low, high, target in bins:
        candidates = [row for row in pool if low <= row["area_m2"] < high]
        candidates.sort(key=lambda row: (abs(row["area_m2"] - target), row["object_id"]))
        chosen = None
        for distance in (5000.0, 3000.0, 1500.0, 0.0):
            for row in candidates:
                if all(
                    np.hypot(row["center_x"] - old["center_x"], row["center_y"] - old["center_y"])
                    >= distance for old in (*used, *selected)
                ):
                    chosen = dict(row)
                    chosen["selection_min_distance_m"] = distance
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"No review sample available in area bin {low}-{high}")
        selected.append(chosen)
    return selected


def region_context(region: str) -> tuple[list[dict], np.ndarray, list, rasterio.DatasetReader]:
    area_path = BRIDGE / region / "original_v75_bridge_after_area_gate.tif"
    reference = rasterio.open(area_path)
    area_gate = reference.read(1) > 0
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    slices = ndimage.find_objects(labels)
    rows = read_rows(region)
    for row in rows:
        component_slice = slices[row["object_id"] - 1]
        center_row = (component_slice[0].start + component_slice[0].stop - 1) / 2
        center_col = (component_slice[1].start + component_slice[1].stop - 1) / 2
        row["center_x"], row["center_y"] = rasterio.transform.xy(
            reference.transform, center_row, center_col
        )
    return rows, labels, slices, reference


def main() -> int:
    summary = json.loads((BRIDGE / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError("Cross-region bridge is incomplete")
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    page = 0

    for region in REGIONS:
        rows, labels, slices, reference = region_context(region)
        selected = [row for row in rows if row["selected_loose_candidate"]]
        used = []
        groups = {
            "future_supported": [
                row for row in selected if row["future_fraction_ge_025_retrospective"]
            ],
            "without_future_support": [
                row for row in selected if not row["future_fraction_ge_025_retrospective"]
            ],
        }
        with ExitStack() as stack:
            previous = base.aligned_vrt(
                stack, ROOT / f"output/20240521/{region}/water_mask_{PREVIOUS}.tif", reference
            )
            future = base.aligned_vrt(
                stack, ROOT / f"output/20250521/{region}/water_mask_{FUTURE}.tif", reference
            )
            s2 = {date: base.aligned_vrt(stack, path, reference) for date, path in S2.items()}
            s1 = base.aligned_vrt(stack, S1_TARGET, reference)

            for group, pool in groups.items():
                targets = choose_area_stratified(pool, used)
                used.extend(targets)
                page += 1
                fig, axes = plt.subplots(3, 6, figsize=(29, 15.3), constrained_layout=True)
                for row_index, target in enumerate(targets):
                    object_id = target["object_id"]
                    window, crop_mode = base.display_window(
                        slices[object_id - 1], reference.height, reference.width
                    )
                    row0, row1 = int(window.row_off), int(window.row_off + window.height)
                    col0, col1 = int(window.col_off), int(window.col_off + window.width)
                    object_view = labels[row0:row1, col0:col1] == object_id
                    rgb = {date: base.read_rgb(source, window) for date, source in s2.items()}
                    previous_view = previous.read(1, window=window) > 0
                    future_view = future.read(1, window=window) > 0
                    target_scl = s2[TARGET].read(7, window=window) == 6
                    vv = base.stretch(
                        s1.read(1, window=window, masked=True).astype("float32").filled(np.nan)
                    )
                    for column, day in enumerate((PREVIOUS, TARGET, FUTURE)):
                        axes[row_index, column].imshow(rgb[day])
                        axes[row_index, column].contour(
                            object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9
                        )
                        axes[row_index, column].set_title(f"{day} S2 RGB")
                    axes[row_index, 3].imshow(vv, cmap="gray", vmin=0, vmax=1)
                    axes[row_index, 3].contour(
                        object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9
                    )
                    axes[row_index, 3].set_title(f"{TARGET} S1 VV (local stretch)")
                    overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                    overlay[previous_view & object_view] = (0.0, 0.8, 1.0, 0.72)
                    overlay[target_scl & object_view] = (0.0, 0.9, 0.25, 0.78)
                    overlay[previous_view & target_scl & object_view] = (1.0, 0.85, 0.0, 0.85)
                    axes[row_index, 4].imshow(rgb[TARGET])
                    axes[row_index, 4].imshow(overlay)
                    axes[row_index, 4].set_title("Cyan previous | green target SCL | yellow overlap")
                    future_overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                    future_overlay[future_view & object_view] = (0.85, 0.05, 0.85, 0.75)
                    axes[row_index, 5].imshow(rgb[FUTURE])
                    axes[row_index, 5].imshow(future_overlay)
                    axes[row_index, 5].set_title("Magenta: 2025 support (diagnostic only)")
                    for axis in axes[row_index]:
                        axis.set_xticks([])
                        axis.set_yticks([])
                    fig.text(
                        0.01, 1.0 - (row_index + 0.08) / 3,
                        f"{region} id={object_id} | {group} | area={target['area_m2']/10000:.2f} ha | "
                        f"previous={target['previous_near_fraction']:.1%} | fill={target['bbox_fill']:.2f} | "
                        f"elong={target['elongation']:.2f} | future={target['future_near_fraction_retrospective']:.1%}",
                        fontsize=9.1, va="top",
                    )
                    manifest.append({
                        **target,
                        "review_group": f"{region}_{group}",
                        "page": page,
                        "image": f"review_page_{page:02d}.png",
                        "crop_mode": crop_mode,
                        "proxy_label": "unreviewed",
                        "target_scope_interpretation": "unreviewed",
                        "confidence": "unreviewed",
                        "notes": "",
                    })
                fig.legend(handles=[
                    Patch(color=(0.0, 0.8, 1.0), label="previous-date model support"),
                    Patch(color=(0.0, 0.9, 0.25), label="target-date SCL water"),
                    Patch(color=(0.85, 0.05, 0.85), label="future support (diagnostic only)"),
                ], loc="lower center", ncol=3)
                fig.savefig(output / f"review_page_{page:02d}.png", dpi=140)
                plt.close(fig)
                print(f"Wrote review_page_{page:02d}.png", flush=True)
        reference.close()

    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "complete",
        "experiment": "Temporal-shape cross-region held-out GIS review",
        "targets": len(manifest),
        "pages": page,
        "selection": (
            "For Shihe and Hangbuhe, three new candidate objects in each of 0.2-1, 1-3, and 3-5 ha "
            "were sampled separately from future-supported and no-future-support groups."
        ),
        "limits": [
            "Mechanism-stratified samples do not estimate population accuracy.",
            "Future absence can reflect drying or sensor/model differences, not only false positives.",
            "Future persistence can repeat stable errors and is not field truth.",
            "No object is restored and no DEM volume or production output is changed."
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
