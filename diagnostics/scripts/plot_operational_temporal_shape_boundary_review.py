#!/usr/bin/env python3
"""Plot new GIS samples from four sensitive boundaries of the loose candidate."""

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
import plot_2024_landuse_temporal_evidence_review as evidence  # noqa: E402
import plot_operational_temporal_shape_review as prior_review  # noqa: E402

AUDIT = (
    ROOT / "diagnostics/results/operational_temporal_shape_candidate/20261010_184856_285062"
)
VOLUME_RESULT = (
    ROOT / "diagnostics/results/operational_temporal_shape_dem_volume/20261010_192503_120178"
)
RETROSPECTIVE_FEATURES = (
    ROOT / "diagnostics/results/retrospective_rescue_volume_concentration"
    / "20261010_183252_458888/retrospective_objects_volume_features.csv"
)
OLD_REVIEWS = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024/20261010_175140_447294",
    ROOT / "diagnostics/results/retrospective_rescue_contributor_review_2024/20261010_183734_965492",
    ROOT / "diagnostics/results/operational_temporal_shape_review/20261010_190316_163013",
)
OUTPUT_ROOT = ROOT / "diagnostics/results/operational_temporal_shape_boundary_review"


def read_exact_volumes() -> dict[int, float]:
    volumes = {}
    with RETROSPECTIVE_FEATURES.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("v75_m3") not in (None, ""):
                volumes[int(row["object_id"])] = float(row["v75_m3"])
    with (
        VOLUME_RESULT / "selected_without_future_support_with_volume.csv"
    ).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            volumes[int(row["pond_id"])] = float(row.get("v75_m3") or 0.0)
    return volumes


def choose_targets(rows: list[dict], excluded: set[int]) -> list[dict]:
    available = [
        row for row in rows
        if row["selected_compact_02_5ha"] and row["object_id"] not in excluded
    ]
    pools = {
        "previous_support_025_to_040": [
            row for row in available if 0.25 <= row["previous_near_fraction"] < 0.40
        ],
        "small_area_020_to_030ha": [
            row for row in available if 2000 <= row["area_m2"] < 3000
        ],
        "bbox_fill_045_to_050": [
            row for row in available if 0.45 <= row["bbox_fill"] < 0.50
        ],
        "elongation_25_to_30": [
            row for row in available if 2.5 < row["elongation"] <= 3.0
        ],
    }
    targets = []
    for name, pool in pools.items():
        chosen = prior_review.choose_spatial(pool, 3, targets)
        if len(chosen) != 3:
            raise RuntimeError(f"Boundary pool {name} supplied only {len(chosen)} objects")
        targets.extend({**row, "review_group": name} for row in chosen)
    if len({row["object_id"] for row in targets}) != 12:
        raise RuntimeError("Boundary review contains duplicate IDs")
    return targets


def main() -> int:
    rows = prior_review.read_rows(AUDIT / "previous_strong_shape_candidates.csv")
    exact_volumes = read_exact_volumes()
    for row in rows:
        row["retrospective_v75_m3"] = exact_volumes.get(row["object_id"])

    excluded = set()
    for review in OLD_REVIEWS:
        with (review / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            excluded.update(int(row["object_id"]) for row in csv.DictReader(stream))
    targets = choose_targets(rows, excluded)

    with rasterio.open(evidence.AREA_GATE) as source:
        area_gate = source.read(1) > 0
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    if count != 28078:
        raise RuntimeError("Area-gate component count changed")
    slices = ndimage.find_objects(labels)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(evidence.AREA_GATE))
        previous = evidence.aligned_vrt(stack, evidence.PREVIOUS_MASK, reference)
        future = evidence.aligned_vrt(stack, evidence.FUTURE_MASK, reference)
        s2 = {
            date: evidence.aligned_vrt(stack, path, reference)
            for date, path in evidence.S2.items()
        }
        s1 = evidence.aligned_vrt(stack, evidence.S1_TARGET, reference)

        for page, start in enumerate(range(0, len(targets), 3), start=1):
            subset = targets[start:start + 3]
            fig, axes = plt.subplots(
                3, 6, figsize=(29, 15.3), squeeze=False, constrained_layout=True
            )
            for row_index, target in enumerate(subset):
                object_id = target["object_id"]
                window, crop_mode = evidence.display_window(
                    slices[object_id - 1], reference.height, reference.width
                )
                row0, row1 = int(window.row_off), int(window.row_off + window.height)
                col0, col1 = int(window.col_off), int(window.col_off + window.width)
                object_view = labels[row0:row1, col0:col1] == object_id
                rgb = {date: evidence.read_rgb(source, window) for date, source in s2.items()}
                previous_view = previous.read(1, window=window) > 0
                future_view = future.read(1, window=window) > 0
                target_scl = s2["2024-06-01"].read(7, window=window) == 6
                vv = evidence.stretch(
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
                axes[row_index, 5].set_title("Magenta: 2025 support (diagnostic only)")
                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])

                v75 = exact_volumes.get(object_id, 0.0)
                fig.text(
                    0.01, 1.0 - (row_index + 0.08) / 3,
                    f"id={object_id} | {target['review_group']} | area={target['area_m2']/10000:.2f} ha | "
                    f"V75={v75/10000:.2f} wan m3 | previous={target['previous_near_fraction']:.1%} | "
                    f"fill={target['bbox_fill']:.2f} | elong={target['elongation']:.2f} | "
                    f"future={target['future_near_fraction']:.1%}",
                    fontsize=9.1, va="top",
                )
                manifest.append({
                    **target,
                    "exact_v75_m3": v75,
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

    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "complete",
        "experiment": "Operational temporal-shape sensitive-boundary GIS review",
        "targets": len(manifest),
        "pages": 4,
        "excluded_previous_review_objects": len(excluded),
        "selection": (
            "Three unseen objects from each capacity-sensitive boundary: previous support 0.25-0.40, "
            "area 0.2-0.3 ha, bbox fill 0.45-0.50, and elongation 2.5-3.0."
        ),
        "limits": [
            "Mechanism-stratified images do not estimate population accuracy.",
            "Future support is diagnostic only and may repeat systematic error.",
            "The four pools can overlap in the population; selected review IDs are unique.",
            "No threshold, mask, model output, or DEM result is changed.",
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
