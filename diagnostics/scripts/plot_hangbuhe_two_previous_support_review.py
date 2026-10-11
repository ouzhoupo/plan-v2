#!/usr/bin/env python3
"""Plot held-out Hangbuhe samples for two-previous-date support review."""

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
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import plot_temporal_shape_cross_region_review as review  # noqa: E402


REGION = "hangbuhe"
DATES = ("2024-05-01", "2024-05-21", "2024-06-01", "2025-05-21")
OBJECTS = (
    ROOT / "diagnostics/results/hangbuhe_two_previous_support"
    / "20261011_163736_703291/hangbuhe_candidates_two_previous_support.csv"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/hangbuhe_two_previous_support_review"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def choose(pool: list[dict], used: list[dict]) -> list[dict]:
    ordered = sorted(pool, key=lambda row: (row["v75_wan_m3"], row["object_id"]))
    if len(ordered) < 3:
        raise RuntimeError(f"Review pool has only {len(ordered)} objects")
    selected = []
    for quantile in (0.25, 0.50, 0.75):
        center = int(round((len(ordered) - 1) * quantile))
        target = ordered[center]["v75_wan_m3"]
        candidates = sorted(ordered, key=lambda row: (abs(row["v75_wan_m3"] - target), row["object_id"]))
        chosen = None
        for distance in (5000.0, 3000.0, 1500.0, 0.0):
            for row in candidates:
                if row in selected:
                    continue
                if all(
                    np.hypot(row["center_x"] - old["center_x"], row["center_y"] - old["center_y"])
                    >= distance for old in (*used, *selected)
                ):
                    chosen = dict(row)
                    chosen["selection_min_distance_m"] = distance
                    chosen["selection_volume_quantile"] = quantile
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("Unable to select review object")
        selected.append(chosen)
    return selected


def main() -> int:
    rows, labels, slices, reference = review.region_context(REGION)
    spatial = {row["object_id"]: row for row in rows}
    reviewed_ids = set()
    for manifest in ROOT.glob("diagnostics/results/*review*/20261010_*/review_manifest.csv"):
        for row in read_csv(manifest):
            if row.get("region") == REGION:
                reviewed_ids.add(int(row["object_id"]))

    objects = []
    for row in read_csv(OBJECTS):
        object_id = int(row["object_id"])
        if object_id in reviewed_ids:
            continue
        location = spatial[object_id]
        objects.append({
            "region": REGION,
            "object_id": object_id,
            "area_m2": float(row["area_m2"]),
            "may01_support": float(row["may01_support"]),
            "may21_support": float(row["may21_support"]),
            "future_proxy": truth(row["future_proxy"]),
            "bbox_fill": float(row["bbox_fill"]),
            "elongation": float(row["elongation"]),
            "v75_wan_m3": float(row["v75_wan_m3"]),
            "center_x": location["center_x"],
            "center_y": location["center_y"],
        })

    groups = {
        "two_previous_and_future": [row for row in objects if row["may01_support"] >= 0.10 and row["future_proxy"]],
        "two_previous_no_future": [row for row in objects if row["may01_support"] >= 0.10 and not row["future_proxy"]],
        "may01_missing_but_future": [row for row in objects if row["may01_support"] < 0.10 and row["future_proxy"]],
        "may01_missing_no_future": [row for row in objects if row["may01_support"] < 0.10 and not row["future_proxy"]],
    }

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    used = []
    page = 0
    s2_paths = {date: ROOT / f"data/s2/s2_stack_{date}.tif" for date in DATES}
    mask_paths = {
        "2024-05-01": ROOT / "output/20240501/hangbuhe/water_mask_2024-05-01.tif",
        "2024-05-21": ROOT / "output/20240521/hangbuhe/water_mask_2024-05-21.tif",
        "2025-05-21": ROOT / "output/20250521/hangbuhe/water_mask_2025-05-21.tif",
    }
    with ExitStack() as stack:
        s2 = {date: review.base.aligned_vrt(stack, path, reference) for date, path in s2_paths.items()}
        masks = {date: review.base.aligned_vrt(stack, path, reference) for date, path in mask_paths.items()}
        s1 = review.base.aligned_vrt(stack, ROOT / "data/s1/s1_stack_2024-06-01.tif", reference)
        for group, pool in groups.items():
            targets = choose(pool, used)
            used.extend(targets)
            page += 1
            fig, axes = plt.subplots(3, 7, figsize=(32, 15.3), constrained_layout=True)
            for row_index, target in enumerate(targets):
                object_id = target["object_id"]
                window, crop_mode = review.base.display_window(
                    slices[object_id - 1], reference.height, reference.width
                )
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                rgb = {date: review.base.read_rgb(source, window) for date, source in s2.items()}
                for column, date in enumerate(DATES):
                    axes[row_index, column].imshow(rgb[date])
                    axes[row_index, column].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                    axes[row_index, column].set_title(f"{date} S2 RGB")
                vv = review.base.stretch(s1.read(1, window=window, masked=True).astype("float32").filled(np.nan))
                axes[row_index, 4].imshow(vv, cmap="gray", vmin=0, vmax=1)
                axes[row_index, 4].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                axes[row_index, 4].set_title("2024-06-01 S1 VV")
                past_overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                may01 = masks["2024-05-01"].read(1, window=window) > 0
                may21 = masks["2024-05-21"].read(1, window=window) > 0
                past_overlay[may01 & object_view] = (1.0, 0.35, 0.0, 0.76)
                past_overlay[may21 & object_view] = (0.0, 0.8, 1.0, 0.76)
                past_overlay[may01 & may21 & object_view] = (1.0, 0.85, 0.0, 0.88)
                axes[row_index, 5].imshow(rgb["2024-06-01"])
                axes[row_index, 5].imshow(past_overlay)
                axes[row_index, 5].set_title("Orange May01 | cyan May21 | yellow both")
                future = masks["2025-05-21"].read(1, window=window) > 0
                future_overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                future_overlay[future & object_view] = (0.85, 0.05, 0.85, 0.76)
                axes[row_index, 6].imshow(rgb["2025-05-21"])
                axes[row_index, 6].imshow(future_overlay)
                axes[row_index, 6].set_title("Magenta: future support (diagnostic)")
                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                fig.text(
                    0.01, 1.0 - (row_index + 0.08) / 3,
                    f"id={object_id} | {group} | area={target['area_m2']/10000:.2f}ha | V75={target['v75_wan_m3']:.2f}万m³ | "
                    f"May01={target['may01_support']:.1%} | May21={target['may21_support']:.1%} | fill={target['bbox_fill']:.2f}",
                    fontsize=8.8, va="top",
                )
                manifest_rows.append({
                    **target,
                    "review_group": group,
                    "page": page,
                    "image": f"review_page_{page:02d}.png",
                    "crop_mode": crop_mode,
                    "proxy_label": "unreviewed",
                    "target_scope_interpretation": "unreviewed",
                    "confidence": "unreviewed",
                    "notes": "",
                })
            fig.legend(handles=[
                Patch(color=(1.0, 0.35, 0.0), label="2024-05-01 support"),
                Patch(color=(0.0, 0.8, 1.0), label="2024-05-21 support"),
                Patch(color=(0.85, 0.05, 0.85), label="2025 support, diagnostic only"),
            ], loc="lower center", ncol=3)
            fig.savefig(output / f"review_page_{page:02d}.png", dpi=140)
            plt.close(fig)
            print(f"Wrote review_page_{page:02d}.png", flush=True)
    reference.close()

    manifest = output / "review_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "status": "complete",
        "experiment": "Hangbuhe two-previous-date held-out GIS review",
        "targets": len(manifest_rows),
        "pages": page,
        "groups": {name: len(pool) for name, pool in groups.items()},
        "selection": "Three held-out objects near low, median, and high V75 rank in each persistence/future group.",
        "limits": [
            "Samples diagnose mechanisms and do not estimate population accuracy.",
            "Future support remains retrospective context rather than a selection input.",
            "No model output or candidate mask is changed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

