#!/usr/bin/env python3
"""Plot held-out Shihe samples for target-date SCL secondary rescue."""

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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import plot_temporal_shape_cross_region_review as review  # noqa: E402


REGION = "shihe"
DATES = ("2024-05-01", "2024-05-21", "2024-06-01", "2025-05-21")
OBJECTS = (
    ROOT / "diagnostics/results/shihe_tiered_current_evidence"
    / "20261010_211915_014476/shihe_tiered_current_evidence_objects.csv"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/shihe_scl_secondary_rescue_review"


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
        target_volume = ordered[center]["v75_wan_m3"]
        candidates = sorted(
            ordered,
            key=lambda row: (abs(row["v75_wan_m3"] - target_volume), row["object_id"]),
        )
        chosen = None
        for distance in (5000.0, 3000.0, 1500.0, 0.0):
            for row in candidates:
                if row in selected:
                    continue
                if all(
                    np.hypot(row["center_x"] - old["center_x"], row["center_y"] - old["center_y"])
                    >= distance
                    for old in (*used, *selected)
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
        if object_id in reviewed_ids or float(row["may01_support"]) >= 0.10:
            continue
        location = spatial[object_id]
        objects.append({
            "region": REGION,
            "object_id": object_id,
            "area_m2": float(row["area_m2"]),
            "may01_support": float(row["may01_support"]),
            "may21_support": float(row["may21_support"]),
            "future_proxy": truth(row["future_proxy"]),
            "v75_wan_m3": float(row["v75_wan_m3"]),
            "scl_water_fraction": float(row["scl_water_fraction"]),
            "source_3_both_fraction": float(row["source_3_both_fraction"]),
            "high_probability_fraction": float(row["high_probability_fraction"]),
            "mean_probability": float(row["mean_probability"]),
            "mean_confidence": float(row["mean_confidence"]),
            "center_x": location["center_x"],
            "center_y": location["center_y"],
        })

    groups = {
        "scl_ge_025_future": [r for r in objects if r["scl_water_fraction"] >= 0.25 and r["future_proxy"]],
        "scl_ge_025_no_future": [r for r in objects if r["scl_water_fraction"] >= 0.25 and not r["future_proxy"]],
        "scl_010_025_future": [r for r in objects if 0.10 <= r["scl_water_fraction"] < 0.25 and r["future_proxy"]],
        "scl_010_025_no_future": [r for r in objects if 0.10 <= r["scl_water_fraction"] < 0.25 and not r["future_proxy"]],
    }

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    used = []
    page = 0
    s2_paths = {date: ROOT / f"data/s2/s2_stack_{date}.tif" for date in DATES}
    mask_paths = {
        "2024-05-01": ROOT / "output/20240501/shihe/water_mask_2024-05-01.tif",
        "2024-05-21": ROOT / "output/20240521/shihe/water_mask_2024-05-21.tif",
        "2025-05-21": ROOT / "output/20250521/shihe/water_mask_2025-05-21.tif",
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

                scl = s2["2024-06-01"].read(7, window=window) == 6
                scl_overlay = np.ma.masked_where(~(scl & object_view), np.ones(object_view.shape))
                axes[row_index, 5].imshow(rgb["2024-06-01"])
                axes[row_index, 5].imshow(scl_overlay, cmap=ListedColormap(["lime"]), alpha=0.72, vmin=0, vmax=1)
                axes[row_index, 5].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                axes[row_index, 5].set_title("Green: target SCL water inside object")

                future = masks["2025-05-21"].read(1, window=window) > 0
                future_overlay = np.ma.masked_where(~(future & object_view), np.ones(object_view.shape))
                axes[row_index, 6].imshow(rgb["2025-05-21"])
                axes[row_index, 6].imshow(future_overlay, cmap=ListedColormap(["magenta"]), alpha=0.72, vmin=0, vmax=1)
                axes[row_index, 6].contour(object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9)
                axes[row_index, 6].set_title("Magenta: future support (diagnostic)")

                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                fig.text(
                    0.01,
                    1.0 - (row_index + 0.08) / 3,
                    f"id={object_id} | {group} | area={target['area_m2']/10000:.2f}ha | "
                    f"V75={target['v75_wan_m3']:.2f}万m³ | SCL={target['scl_water_fraction']:.1%} | "
                    f"May01={target['may01_support']:.1%} | May21={target['may21_support']:.1%} | "
                    f"Pmean={target['mean_probability']:.3f}",
                    fontsize=8.6,
                    va="top",
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
            fig.legend(
                handles=[
                    Patch(color="yellow", label="candidate object boundary"),
                    Patch(color="lime", label="target-date SCL water"),
                    Patch(color="magenta", label="2025 support, diagnostic only"),
                ],
                loc="lower center",
                ncol=3,
            )
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
        "experiment": "Shihe SCL secondary-rescue held-out GIS review",
        "targets": len(manifest_rows),
        "pages": page,
        "groups_after_exclusion": {name: len(pool) for name, pool in groups.items()},
        "selection": "Three held-out objects near low, median, and high V75 rank per SCL/future stratum.",
        "purpose": "Compare target SCL thresholds 0.10 and 0.25 for objects lacking second historical support.",
        "limits": [
            "Samples diagnose mechanisms and do not estimate population accuracy.",
            "Future support is retrospective context rather than a selection input.",
            "SCL is model-side evidence and is not field truth.",
            "No production output or parameter is changed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
