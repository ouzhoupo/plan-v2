#!/usr/bin/env python3
"""Plot new GIS samples from the volume-dominant retrospective rescue pool."""

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
from rasterio.windows import Window
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import plot_2024_landuse_temporal_evidence_review as base  # noqa: E402

AUDIT = (
    ROOT / "diagnostics/results/retrospective_rescue_volume_concentration"
    / "20261010_183252_458888"
)
OLD_REVIEW = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024"
    / "20261010_175140_447294"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/retrospective_rescue_contributor_review_2024"


def read_features(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "area_m2", "v75_m3", "previous_near_fraction", "target_scl_fraction",
        "future_near_fraction", "bbox_fill", "compactness", "elongation",
        "center_x", "center_y",
    )
    for row in rows:
        row["object_id"] = int(row["object_id"])
        for key in numeric:
            row[key] = float(row[key]) if row[key] not in ("", None) else float("nan")
    return rows


def select_spatial(pool: list[dict], count: int, used: list[dict]) -> list[dict]:
    ordered = sorted(pool, key=lambda row: (-row["v75_m3"], -row["area_m2"], row["object_id"]))
    selected = []
    for minimum in (5000.0, 3000.0, 1500.0, 500.0, 0.0):
        for row in ordered:
            if len(selected) >= count:
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
        if len(selected) >= count:
            break
    return selected[:count]


def choose_targets(rows: list[dict], excluded: set[int]) -> list[dict]:
    available = [row for row in rows if row["object_id"] not in excluded]
    pools = {
        "top_v75_contributors": available,
        "compact_02_1ha_zero_scl": [
            row for row in available
            if 2000 <= row["area_m2"] < 10000
            and row["target_scl_fraction"] == 0
            and row["bbox_fill"] >= 0.45 and row["elongation"] <= 3
        ],
        "compact_1_5ha_zero_scl": [
            row for row in available
            if 10000 <= row["area_m2"] < 50000
            and row["target_scl_fraction"] == 0
            and row["bbox_fill"] >= 0.45 and row["elongation"] <= 3
        ],
        "elongated_02_5ha_zero_scl": [
            row for row in available
            if 2000 <= row["area_m2"] < 50000
            and row["target_scl_fraction"] == 0
            and row["elongation"] >= 4
        ],
        "weak_target_scl_below_025": [
            row for row in available if 0 < row["target_scl_fraction"] < 0.25
        ],
    }
    targets = []
    for name, pool in pools.items():
        chosen = select_spatial(pool, 3, targets)
        if len(chosen) != 3:
            raise RuntimeError(f"Review pool {name} supplied only {len(chosen)} objects")
        targets.extend({**row, "review_group": name} for row in chosen)
    if len({row["object_id"] for row in targets}) != 15:
        raise RuntimeError("Review selection contains duplicate objects")
    return targets


def main() -> int:
    audit = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("Volume concentration audit is incomplete")
    rows = read_features(AUDIT / "retrospective_objects_volume_features.csv")
    old_labels = base.read_rows(OLD_REVIEW / "review_manifest.csv")
    excluded = {row["object_id"] for row in old_labels}
    targets = choose_targets(rows, excluded)

    with rasterio.open(base.AREA_GATE) as source:
        area_gate = source.read(1) > 0
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    if count != 28078:
        raise RuntimeError(f"Area-gate component count changed: {count}")
    slices = ndimage.find_objects(labels)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(base.AREA_GATE))
        previous = base.aligned_vrt(stack, base.PREVIOUS_MASK, reference)
        future = base.aligned_vrt(stack, base.FUTURE_MASK, reference)
        s2 = {date: base.aligned_vrt(stack, path, reference) for date, path in base.S2.items()}
        s1 = base.aligned_vrt(stack, base.S1_TARGET, reference)

        for page, start in enumerate(range(0, len(targets), 3), start=1):
            subset = targets[start:start + 3]
            fig, axes = plt.subplots(
                len(subset), 6, figsize=(29, 5.1 * len(subset)), squeeze=False,
                constrained_layout=True,
            )
            for row_index, target in enumerate(subset):
                object_id = target["object_id"]
                component_slice = slices[object_id - 1]
                window, crop_mode = base.display_window(
                    component_slice, reference.height, reference.width
                )
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                rgb = {date: base.read_rgb(source, window) for date, source in s2.items()}
                previous_view = previous.read(1, window=window) > 0
                future_view = future.read(1, window=window) > 0
                target_scl = s2["2024-06-01"].read(7, window=window) == 6
                vv = base.stretch(
                    s1.read(1, window=window, masked=True).astype("float32").filled(np.nan)
                )

                for column, date in enumerate(("2024-05-21", "2024-06-01", "2025-05-21")):
                    axes[row_index, column].imshow(rgb[date])
                    axes[row_index, column].contour(
                        object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9
                    )
                    axes[row_index, column].set_title(f"{date} S2 RGB")
                axes[row_index, 3].imshow(vv, cmap="gray", vmin=0, vmax=1)
                axes[row_index, 3].contour(
                    object_view.astype("uint8"), levels=[0.5], colors="yellow", linewidths=0.9
                )
                axes[row_index, 3].set_title("2024-06-01 S1 VV (local stretch)")

                overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                overlay[previous_view & object_view] = (0.0, 0.8, 1.0, 0.72)
                overlay[target_scl & object_view] = (0.0, 0.9, 0.25, 0.78)
                overlay[previous_view & target_scl & object_view] = (1.0, 0.85, 0.0, 0.85)
                axes[row_index, 4].imshow(rgb["2024-06-01"])
                axes[row_index, 4].imshow(overlay)
                axes[row_index, 4].set_title("Cyan previous | green target SCL | yellow overlap")

                future_overlay = np.zeros((*object_view.shape, 4), dtype="float32")
                future_overlay[future_view & object_view] = (0.85, 0.05, 0.85, 0.75)
                axes[row_index, 5].imshow(rgb["2025-05-21"])
                axes[row_index, 5].imshow(future_overlay)
                axes[row_index, 5].set_title("Magenta: 2025 model support (diagnostic only)")
                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                fig.text(
                    0.01, 1.0 - (row_index + 0.08) / len(subset),
                    f"id={object_id} | {target['review_group']} | area={target['area_m2']/10000:.2f} ha | "
                    f"V75={target['v75_m3']/10000:.2f} wan m3 | fill={target['bbox_fill']:.2f} | "
                    f"elong={target['elongation']:.2f} | SCL={target['target_scl_fraction']:.1%}",
                    fontsize=9.2, va="top",
                )
                manifest.append({
                    **target,
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
        "experiment": "Retrospective rescue volume-contributor GIS review",
        "targets": len(manifest),
        "pages": 5,
        "excluded_previous_review_objects": len(excluded),
        "selection": (
            "Three unseen objects from each of five volume/shape strata: top V75; compact 0.2-1 ha "
            "with zero target SCL; compact 1-5 ha with zero target SCL; elongated 0.2-5 ha with zero "
            "target SCL; and weak nonzero target SCL below 0.25."
        ),
        "limits": [
            "The sample is mechanism-stratified and cannot estimate population accuracy.",
            "Top-volume selection intentionally overrepresents influential objects.",
            "2025 support remains retrospective and cannot be used as a 2024 inference input.",
            "Local SAR stretching is qualitative and does not define a threshold.",
            "No object is restored and no model output is modified.",
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
