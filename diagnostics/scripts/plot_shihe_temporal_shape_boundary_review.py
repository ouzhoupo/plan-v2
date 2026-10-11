#!/usr/bin/env python3
"""Plot held-out Shihe samples at the two sensitivity boundaries."""

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


REGION = "shihe"
PREVIOUS = "2024-05-21"
TARGET = "2024-06-01"
FUTURE = "2025-05-21"
VOLUME = (
    ROOT / "diagnostics/results/shihe_temporal_shape_dem_volume"
    / "20261010_203830_787211/shihe_candidates_with_volume.csv"
)
PRIOR_REVIEW = (
    ROOT / "diagnostics/results/temporal_shape_cross_region_review"
    / "20261010_200047_474082/review_manifest.csv"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/shihe_temporal_shape_boundary_review"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def choose_volume_stratified(pool: list[dict], used: list[dict]) -> list[dict]:
    if len(pool) < 3:
        raise RuntimeError(f"Review pool has only {len(pool)} objects")
    ordered = sorted(pool, key=lambda row: (row["v75_m3"], row["object_id"]))
    targets = (0.25, 0.50, 0.75)
    selected = []
    for quantile in targets:
        center = int(round((len(ordered) - 1) * quantile))
        target_volume = ordered[center]["v75_m3"]
        candidates = sorted(
            ordered,
            key=lambda row: (abs(row["v75_m3"] - target_volume), row["object_id"]),
        )
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
            raise RuntimeError("Unable to select spatially separated review sample")
        selected.append(chosen)
    return selected


def main() -> int:
    rows, labels, slices, reference = review.region_context(REGION)
    volume = {int(row["pond_id"]): row for row in read_csv(VOLUME)}
    prior_ids = {
        int(row["object_id"]) for row in read_csv(PRIOR_REVIEW)
        if row.get("region") == REGION
    }
    candidates = []
    for row in rows:
        if not row["selected_loose_candidate"] or row["object_id"] in prior_ids:
            continue
        volume_row = volume.get(row["object_id"])
        if volume_row is None:
            raise RuntimeError(f"Missing V75 for object {row['object_id']}")
        item = dict(row)
        item["v75_m3"] = float(volume_row["v75_m3"] or 0.0)
        item["qa"] = volume_row.get("qa", "")
        candidates.append(item)

    groups = {
        "small_0p2_1ha_future": [
            row for row in candidates
            if row["area_m2"] < 10000 and row["future_fraction_ge_025_retrospective"]
        ],
        "small_0p2_1ha_no_future": [
            row for row in candidates
            if row["area_m2"] < 10000 and not row["future_fraction_ge_025_retrospective"]
        ],
        "low_fill_0p45_0p55_future": [
            row for row in candidates
            if row["area_m2"] >= 10000 and row["bbox_fill"] < 0.55
            and row["future_fraction_ge_025_retrospective"]
        ],
        "low_fill_0p45_0p55_no_future": [
            row for row in candidates
            if row["area_m2"] >= 10000 and row["bbox_fill"] < 0.55
            and not row["future_fraction_ge_025_retrospective"]
        ],
    }

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    used = []
    page = 0
    with ExitStack() as stack:
        previous = review.base.aligned_vrt(
            stack, ROOT / f"output/20240521/{REGION}/water_mask_{PREVIOUS}.tif", reference
        )
        future = review.base.aligned_vrt(
            stack, ROOT / f"output/20250521/{REGION}/water_mask_{FUTURE}.tif", reference
        )
        s2 = {
            date: review.base.aligned_vrt(stack, path, reference)
            for date, path in review.S2.items()
        }
        s1 = review.base.aligned_vrt(stack, review.S1_TARGET, reference)

        for group, pool in groups.items():
            targets = choose_volume_stratified(pool, used)
            used.extend(targets)
            page += 1
            fig, axes = plt.subplots(3, 6, figsize=(29, 15.3), constrained_layout=True)
            for row_index, target in enumerate(targets):
                object_id = target["object_id"]
                window, crop_mode = review.base.display_window(
                    slices[object_id - 1], reference.height, reference.width
                )
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                rgb = {date: review.base.read_rgb(source, window) for date, source in s2.items()}
                previous_view = previous.read(1, window=window) > 0
                future_view = future.read(1, window=window) > 0
                target_scl = s2[TARGET].read(7, window=window) == 6
                vv = review.base.stretch(
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
                    f"id={object_id} | {group} | area={target['area_m2']/10000:.2f}ha | "
                    f"V75={target['v75_m3']/10000:.2f}万m³ | previous={target['previous_near_fraction']:.1%} | "
                    f"fill={target['bbox_fill']:.2f} | elong={target['elongation']:.2f}",
                    fontsize=9.0, va="top",
                )
                manifest.append({
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
        "experiment": "Shihe temporal-shape sensitivity boundary GIS review",
        "targets": len(manifest),
        "pages": page,
        "groups": {name: len(pool) for name, pool in groups.items()},
        "selection": "Three held-out objects near 25%, 50%, and 75% V75 rank in each boundary group.",
        "limits": [
            "Samples diagnose object composition and do not estimate population accuracy.",
            "Future support is retrospective context rather than field truth.",
            "No object is restored and no production output is changed.",
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
