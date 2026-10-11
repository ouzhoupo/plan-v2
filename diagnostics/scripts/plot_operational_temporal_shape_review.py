#!/usr/bin/env python3
"""Plot held-out GIS evidence for operational temporal-shape candidates."""

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

AUDIT = (
    ROOT / "diagnostics/results/operational_temporal_shape_candidate"
    / "20261010_184856_285062"
)
OLD_REVIEWS = (
    ROOT / "diagnostics/results/landuse_temporal_evidence_review_2024/20261010_175140_447294",
    ROOT / "diagnostics/results/retrospective_rescue_contributor_review_2024/20261010_183734_965492",
)
OUTPUT_ROOT = ROOT / "diagnostics/results/operational_temporal_shape_review"


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "area_m2", "previous_near_fraction", "future_near_fraction",
        "target_scl_fraction", "bbox_fill", "elongation", "compactness",
        "center_x", "center_y",
    )
    for row in rows:
        row["object_id"] = int(row["object_id"])
        for key in numeric:
            row[key] = float(row[key])
        for key in (
            "selected_compact_02_5ha", "selected_compact_02_5ha_strict",
            "retrospective_future_persistent",
        ):
            row[key] = truth(row[key])
        row["retrospective_v75_m3"] = (
            float(row["retrospective_v75_m3"])
            if row["retrospective_v75_m3"] not in ("", None) else None
        )
    return rows


def choose_spatial(pool: list[dict], count: int, used: list[dict]) -> list[dict]:
    ordered = sorted(
        pool,
        key=lambda row: (
            -(row["retrospective_v75_m3"] or 0.0), -row["area_m2"], row["object_id"]
        ),
    )
    selected = []
    for minimum in (5000.0, 3000.0, 1500.0, 500.0, 0.0):
        for row in ordered:
            if len(selected) >= count:
                break
            if any(row["object_id"] == old["object_id"] for old in (*used, *selected)):
                continue
            if all(
                np.hypot(row["center_x"] - old["center_x"], row["center_y"] - old["center_y"])
                >= minimum for old in (*used, *selected)
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
        "strict_selected_future_persistent": [
            row for row in available
            if row["selected_compact_02_5ha_strict"]
            and row["retrospective_future_persistent"]
        ],
        "loose_only_future_persistent": [
            row for row in available
            if row["selected_compact_02_5ha"]
            and not row["selected_compact_02_5ha_strict"]
            and row["retrospective_future_persistent"]
        ],
        "loose_selected_without_future": [
            row for row in available
            if row["selected_compact_02_5ha"]
            and not row["retrospective_future_persistent"]
        ],
        "future_persistent_rejected_elongated": [
            row for row in available
            if row["retrospective_future_persistent"]
            and 2000 <= row["area_m2"] < 50000
            and row["elongation"] > 3.0
        ],
        "future_persistent_rejected_large": [
            row for row in available
            if row["retrospective_future_persistent"] and row["area_m2"] >= 50000
        ],
    }
    targets = []
    for name, pool in pools.items():
        chosen = choose_spatial(pool, 3, targets)
        if len(chosen) != 3:
            raise RuntimeError(f"Review pool {name} supplied only {len(chosen)} objects")
        targets.extend({**row, "review_group": name} for row in chosen)
    if len({row["object_id"] for row in targets}) != 15:
        raise RuntimeError("Review selection contains duplicate IDs")
    return targets


def main() -> int:
    audit = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("Operational shape audit is incomplete")
    rows = read_rows(AUDIT / "previous_strong_shape_candidates.csv")
    excluded = set()
    for review in OLD_REVIEWS:
        with (review / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            excluded.update(int(row["object_id"]) for row in csv.DictReader(stream))
    targets = choose_targets(rows, excluded)

    with rasterio.open(base.AREA_GATE) as source:
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
                window, crop_mode = base.display_window(
                    slices[object_id - 1], reference.height, reference.width
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
                v75 = target["retrospective_v75_m3"]
                v75_text = f"{v75/10000:.2f}" if v75 is not None else "n/a"
                fig.text(
                    0.01, 1.0 - (row_index + 0.08) / len(subset),
                    f"id={object_id} | {target['review_group']} | area={target['area_m2']/10000:.2f} ha | "
                    f"V75={v75_text} wan m3 | fill={target['bbox_fill']:.2f} | "
                    f"elong={target['elongation']:.2f} | future={target['future_near_fraction']:.1%}",
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
        "experiment": "Operational temporal-shape candidate held-out GIS review",
        "targets": len(manifest),
        "pages": 5,
        "excluded_previous_review_objects": len(excluded),
        "selection": (
            "Three unseen objects from strict selected/future persistent, loose-only/future persistent, "
            "loose selected/no future support, future persistent rejected as elongated, and future "
            "persistent rejected as >=5 ha."
        ),
        "limits": [
            "This is a mechanism-stratified review and cannot estimate population accuracy.",
            "Future absence may indicate drying rather than a false positive.",
            "Future persistence may repeat a systematic error rather than prove water.",
            "No candidate is restored and no model output is changed.",
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
