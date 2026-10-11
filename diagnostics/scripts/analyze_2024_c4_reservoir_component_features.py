#!/usr/bin/env python3
"""Add probability/source evidence to 2024-06 reservoir-affected C4 components.

This is a feature audit only. It does not select a rescue threshold, modify a
mask, or calculate DEM volume.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
COMPONENT_RESULT = (
    ROOT
    / "diagnostics/results/second_round_pre_deduction_components"
    / "20260926_212547_194707"
)
REVIEW_RESULT = (
    ROOT
    / "diagnostics/results/second_round_pre_deduction_component_review"
    / "20260926_220001_993492"
)
SPLIT_RESULT = (
    ROOT
    / "diagnostics/results/second_round_deduction_morphology_split"
    / "20260926_203928_115516"
)
UPSTREAM_DATE = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712/2024-06-01"
)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    return float(np.quantile(values, q)) if values.size else None


def main() -> int:
    component_rows = read_csv(COMPONENT_RESULT / "all_pre_deduction_components.csv")
    labels = {
        int(row["component_id"]): row
        for row in read_csv(REVIEW_RESULT / "assistant_visual_labels.csv")
    }

    pre_path = SPLIT_RESULT / "c4_pre_deduction_no_morph.tif"
    probability_path = UPSTREAM_DATE / "c2_fused_probability.tif"
    source_path = UPSTREAM_DATE / "c2_source_flag.tif"
    with (
        rasterio.open(pre_path) as pre_src,
        rasterio.open(probability_path) as prob_src,
        rasterio.open(source_path) as source_src,
    ):
        for other in (prob_src, source_src):
            if (
                pre_src.width != other.width
                or pre_src.height != other.height
                or pre_src.crs != other.crs
                or not pre_src.transform.almost_equals(other.transform)
            ):
                raise RuntimeError(f"Grid mismatch: {other.name}")
        pre = pre_src.read(1) > 0
        probability = prob_src.read(1).astype("float32")
        source_flag = source_src.read(1)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8")
    components, component_count = ndimage.label(pre, structure=structure)
    if component_count != len(component_rows):
        raise RuntimeError("Component reconstruction count mismatch")
    slices = ndimage.find_objects(components)

    scope = []
    for row in component_rows:
        area = float(row["area_m2"])
        fully = row["fully_removed"].lower() == "true"
        partially = row["partially_removed"].lower() == "true"
        reservoir_pixels = int(row["reservoir_involved_pixels"])
        dominant = row["dominant_deduction_group"]
        if not (500 <= area <= 500_000 and reservoir_pixels > 0):
            continue
        if not (fully or (partially and dominant.startswith("reservoir"))):
            continue

        component_id = int(row["component_id"])
        bbox = slices[component_id - 1]
        local = components[bbox] == component_id
        probs = probability[bbox][local]
        probs = probs[np.isfinite(probs)]
        sources = source_flag[bbox][local]
        source_counts = Counter(int(value) for value in sources)
        pixels = int(row["pixels"])
        label = labels.get(component_id, {})
        scope.append(
            {
                **row,
                "probability_valid_fraction": probs.size / pixels,
                "probability_mean": float(np.mean(probs)) if probs.size else None,
                "probability_p10": quantile(probs, 0.10),
                "probability_p50": quantile(probs, 0.50),
                "probability_p90": quantile(probs, 0.90),
                "source_0_fraction": source_counts[0] / pixels,
                "source_1_fraction": source_counts[1] / pixels,
                "source_2_fraction": source_counts[2] / pixels,
                "source_3_fraction": source_counts[3] / pixels,
                "source_4_fraction": source_counts[4] / pixels,
                "proxy_label": label.get("proxy_label", "unreviewed"),
                "proxy_confidence": label.get("confidence", ""),
                "proxy_notes": label.get("notes", ""),
            }
        )

    reservoir_label_ids = {
        component_id
        for component_id, row in labels.items()
        if row["review_branch"] != "partially_cut_linear_target"
    }
    found_labels = {int(row["component_id"]) for row in scope if row["proxy_label"] != "unreviewed"}
    missing_labels = sorted(reservoir_label_ids - found_labels)
    if missing_labels:
        raise RuntimeError(f"Labeled reservoir-scope components missing: {missing_labels}")

    labeled = [row for row in scope if row["proxy_label"] != "unreviewed"]
    label_summary = defaultdict(lambda: {"objects": 0, "area_m2": 0.0, "removed_area_m2": 0.0})
    for row in labeled:
        key = f"{row['review_branch'] if 'review_branch' in row else labels[int(row['component_id'])]['review_branch']}|{row['proxy_label']}"
        label_summary[key]["objects"] += 1
        label_summary[key]["area_m2"] += float(row["area_m2"])
        label_summary[key]["removed_area_m2"] += float(row["removed_area_m2"])

    output = (
        ROOT
        / "diagnostics/results/second_round_reservoir_component_features"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "reservoir_affected_component_features.csv", scope)
    write_csv(output / "labeled_component_features.csv", labeled)
    write_csv(
        output / "visual_label_summary.csv",
        [
            {
                "branch_and_label": key,
                "objects": values["objects"],
                "area_km2": values["area_m2"] / 1_000_000.0,
                "removed_area_km2": values["removed_area_m2"] / 1_000_000.0,
            }
            for key, values in sorted(label_summary.items())
        ],
    )

    validations = {
        "all_18_reservoir_visual_labels_joined": len(found_labels) == 18,
        "all_scoped_components_have_finite_probability": all(
            float(row["probability_valid_fraction"]) > 0 for row in scope
        ),
    }
    status = "complete" if all(validations.values()) else "validation_failed"
    report = {
        "status": status,
        "experiment": "Second-round reservoir-affected component feature audit",
        "scope": (
            "2024-06 pre-deduction components of 500-500000 m2 that are either "
            "fully removed with reservoir involvement or partially removed with "
            "a reservoir-dominant deduction group"
        ),
        "scope_objects": len(scope),
        "scope_original_area_km2": sum(float(row["area_m2"]) for row in scope) / 1_000_000.0,
        "scope_removed_area_km2": sum(float(row["removed_area_m2"]) for row in scope) / 1_000_000.0,
        "visual_labels_joined": len(found_labels),
        "validations": validations,
        "outputs": {
            "features_csv": str(output / "reservoir_affected_component_features.csv"),
            "labeled_features_csv": str(output / "labeled_component_features.csv"),
            "label_summary_csv": str(output / "visual_label_summary.csv"),
        },
        "limits": [
            "The 18 visual labels in this scope are proxy evidence, not field truth.",
            "The review sample is stratified and cannot estimate population accuracy.",
            "No threshold is selected and no candidate is restored in this experiment.",
            "Source flags describe upstream evidence paths and are not land-cover classes.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {status}", flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
