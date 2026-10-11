#!/usr/bin/env python3
"""Separate SAR and optical probability drift on fixed cross-date pixels.

This replays the frozen paired C2 calculation used by the combination experiment
and summarizes its two branches over fixed scopes.  It changes no parameter and
writes no candidate mask.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import experiment_a3_dual_pol_upstream as a3  # noqa: E402
import run_combined_upstream_c0_c4 as combined  # noqa: E402


UPSTREAM = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/two_date_branch_probabilities"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"
THRESHOLDS = (0.20, 0.50, 0.62, 0.68, 0.78, 0.80, 0.93)


def new_accumulator() -> dict:
    return {
        "scope_pixels": 0,
        "valid_pixels": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "ge": {str(value): 0 for value in THRESHOLDS},
    }


def update(acc: dict, scope: np.ndarray, values: np.ndarray) -> None:
    acc["scope_pixels"] += int(np.count_nonzero(scope))
    valid = scope & np.isfinite(values)
    selected = values[valid].astype("float64")
    count = int(selected.size)
    acc["valid_pixels"] += count
    if not count:
        return
    acc["sum"] += float(selected.sum())
    acc["sumsq"] += float(np.square(selected).sum())
    for threshold in THRESHOLDS:
        acc["ge"][str(threshold)] += int(np.count_nonzero(selected >= threshold))


def finish(acc: dict, pixel_area: float) -> dict:
    scope_count = acc["scope_pixels"]
    valid_count = acc["valid_pixels"]
    mean = acc["sum"] / valid_count if valid_count else None
    variance = (
        max(acc["sumsq"] / valid_count - mean * mean, 0.0)
        if valid_count else None
    )
    return {
        "scope_pixels": scope_count,
        "scope_area_km2": scope_count * pixel_area / 1e6,
        "valid_pixels": valid_count,
        "valid_area_km2": valid_count * pixel_area / 1e6,
        "valid_fraction": valid_count / scope_count if scope_count else None,
        "mean": mean,
        "sd": variance ** 0.5 if variance is not None else None,
        **{
            f"ge_{threshold}_fraction_of_valid": acc["ge"][str(threshold)] / valid_count
            if valid_count else None
            for threshold in THRESHOLDS
        },
    }


def close_context(context: dict) -> None:
    for reader in context["readers"].values():
        reader.close()
    a3.m.close_scenes(
        context["s1"],
        context["s2"],
        extras=[context["dem"]] if context["dem"] is not None else [],
    )


def main() -> int:
    print(f"Preparing exact paired C2 contexts: {DATE_24}", flush=True)
    context24, captured24 = a3.capture_setup(DATE_24)
    alternate24, training24 = a3.replay_component_split(context24, captured24)
    print(f"Preparing exact paired C2 contexts: {DATE_25}", flush=True)
    context25, captured25 = a3.capture_setup(DATE_25)
    alternate25, training25 = a3.replay_component_split(context25, captured25)

    try:
        grid24, grid25 = context24["grid"], context25["grid"]
        if (
            grid24.width != grid25.width
            or grid24.height != grid25.height
            or grid24.crs != grid25.crs
            or not grid24.transform.almost_equals(grid25.transform)
        ):
            raise RuntimeError("Cross-date runtime grid mismatch")

        paths = {
            "water24": UPSTREAM / DATE_24 / "c4_water_mask.tif",
            "water25": UPSTREAM / DATE_25 / "c4_water_mask.tif",
            "source24": UPSTREAM / DATE_24 / "c2_source_flag.tif",
            "source25": UPSTREAM / DATE_25 / "c2_source_flag.tif",
            "prob24": UPSTREAM / DATE_24 / "c2_fused_probability.tif",
            "prob25": UPSTREAM / DATE_25 / "c2_fused_probability.tif",
        }
        accumulators: dict[tuple[str, str, str], dict] = defaultdict(new_accumulator)
        validation = {
            "probability_pixels": 0,
            "probability_abs_sum_2024": 0.0,
            "probability_abs_sum_2025": 0.0,
            "probability_abs_max_2024": 0.0,
            "probability_abs_max_2025": 0.0,
            "source_mismatch_2024": 0,
            "source_mismatch_2025": 0,
            "water_2025_only_pixels": 0,
        }
        with ExitStack() as stack:
            datasets = {
                name: stack.enter_context(rasterio.open(path))
                for name, path in paths.items()
            }
            reference = datasets["water24"]
            pixel_area = abs(float(reference.transform.a * reference.transform.e))
            for name, dataset in datasets.items():
                if (
                    dataset.shape != reference.shape
                    or dataset.crs != reference.crs
                    or not dataset.transform.almost_equals(reference.transform)
                ):
                    raise RuntimeError(f"Saved-grid mismatch: {name}")

            block_size = int(context24["fcfg"].get("block_size", 512))
            for index, window in enumerate(
                a3.m.iter_windows(grid24.width, grid24.height, block_size), start=1
            ):
                water24 = datasets["water24"].read(1, window=window) > 0
                water25 = datasets["water25"].read(1, window=window) > 0
                source24 = datasets["source24"].read(1, window=window)
                source25 = datasets["source25"].read(1, window=window)
                scope_masks = {
                    "water_2025_only": ~water24 & water25,
                    "common_water": water24 & water25,
                    "water_2024_only": water24 & ~water25,
                }
                if not any(np.any(mask) for mask in scope_masks.values()):
                    continue

                values24 = combined.a4_refuse(
                    a3.alternate_values(a3.calc(context24, window), context24, alternate24),
                    context24,
                )
                values25 = combined.a4_refuse(
                    a3.alternate_values(a3.calc(context25, window), context25, alternate25),
                    context25,
                )
                saved_prob24 = datasets["prob24"].read(1, window=window)
                saved_prob25 = datasets["prob25"].read(1, window=window)
                transform = a3.m.window_transform(grid24.transform, window)
                aoi = a3.m.rasterize_geom_mask(
                    grid24.aoi_geom, water24.shape, transform
                )
                finite = (
                    aoi
                    & np.isfinite(values24[4]) & np.isfinite(values25[4])
                    & np.isfinite(saved_prob24) & np.isfinite(saved_prob25)
                )
                difference24 = np.abs(values24[4][finite] - saved_prob24[finite])
                difference25 = np.abs(values25[4][finite] - saved_prob25[finite])
                validation["probability_pixels"] += int(np.count_nonzero(finite))
                validation["probability_abs_sum_2024"] += float(difference24.sum(dtype=np.float64))
                validation["probability_abs_sum_2025"] += float(difference25.sum(dtype=np.float64))
                if difference24.size:
                    validation["probability_abs_max_2024"] = max(
                        validation["probability_abs_max_2024"], float(difference24.max())
                    )
                    validation["probability_abs_max_2025"] = max(
                        validation["probability_abs_max_2025"], float(difference25.max())
                    )
                validation["source_mismatch_2024"] += int(
                    np.count_nonzero(aoi & (values24[5] != source24))
                )
                validation["source_mismatch_2025"] += int(
                    np.count_nonzero(aoi & (values25[5] != source25))
                )
                validation["water_2025_only_pixels"] += int(
                    np.count_nonzero(scope_masks["water_2025_only"])
                )

                groups = {
                    "all": np.ones(water24.shape, dtype=bool),
                    "optical_to_dual": (source24 == 2) & (source25 == 3),
                    "dual_to_dual": (source24 == 3) & (source25 == 3),
                    "optical_to_sar": (source24 == 2) & (source25 == 1),
                    "prior_to_sar_or_dual": (source24 == 4) & np.isin(source25, (1, 3)),
                }
                branch_arrays = {
                    DATE_24: {
                        "sar_probability": values24[0]["sar_prob"],
                        "optical_base_probability": values24[1]["opt_prob_base"],
                        "optical_model_probability": values24[2],
                        "optical_probability": values24[3],
                        "fused_probability": values24[4],
                        "sar_confidence": values24[0]["sar_conf"],
                        "optical_confidence": values24[1]["opt_conf"],
                    },
                    DATE_25: {
                        "sar_probability": values25[0]["sar_prob"],
                        "optical_base_probability": values25[1]["opt_prob_base"],
                        "optical_model_probability": values25[2],
                        "optical_probability": values25[3],
                        "fused_probability": values25[4],
                        "sar_confidence": values25[0]["sar_conf"],
                        "optical_confidence": values25[1]["opt_conf"],
                    },
                }
                for scope_name, scope_mask in scope_masks.items():
                    for group_name, group_mask in groups.items():
                        selection = scope_mask & group_mask
                        if not np.any(selection):
                            continue
                        for date, arrays in branch_arrays.items():
                            for branch, array in arrays.items():
                                update(accumulators[(scope_name, group_name, date, branch)], selection, array)
                if index % 100 == 0:
                    print(f"Processed {index} blocks", flush=True)

        rows = []
        for key in sorted(accumulators):
            scope_name, group_name, date, branch = key
            rows.append({
                "scope": scope_name,
                "source_transition_group": group_name,
                "date": date,
                "branch": branch,
                **finish(accumulators[key], pixel_area),
            })

        probability_pixels = validation["probability_pixels"]
        validation.update({
            "probability_mae_2024": validation["probability_abs_sum_2024"] / probability_pixels,
            "probability_mae_2025": validation["probability_abs_sum_2025"] / probability_pixels,
            "water_2025_only_matches_previous_audit": validation["water_2025_only_pixels"] == 9_455_724,
        })
        validations_pass = (
            validation["source_mismatch_2024"] == 0
            and validation["source_mismatch_2025"] == 0
            and validation["probability_mae_2024"] < 1e-6
            and validation["probability_mae_2025"] < 1e-6
            and validation["water_2025_only_matches_previous_audit"]
        )
        status = "complete" if validations_pass else "validation_failed"

        scene24 = context24["s1"][0]
        scene25 = context25["s1"][0]
        output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "branch_probability_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "status": status,
            "experiment": "Two-date fixed-pixel SAR/optical branch probability audit v2",
            "paired_variant_replayed": "C2 = A4 finite-SAR validity + A3 dual-pol optical training",
            "dates": {
                DATE_24: {
                    "model_quality": context24["quality"],
                    "optical_blend": context24["blend"],
                    "sar_parameters": scene24.params,
                    "training_effect": training24,
                },
                DATE_25: {
                    "model_quality": context25["quality"],
                    "optical_blend": context25["blend"],
                    "sar_parameters": scene25.params,
                    "training_effect": training25,
                },
            },
            "validation": validation,
            "outputs": {"branch_summary": str(output / "branch_probability_summary.csv")},
            "limits": [
                "Branch probability differences locate calibration drift but do not by themselves establish which date is correct.",
                "The optical trained probability is derived from SAR pseudo-labels and is not independent SAR evidence.",
                "Optical base, trained-model and blended probabilities are emitted separately to test pseudo-label amplification.",
                "Confidence and probability are summarized only on fixed C4 change scopes; this is not an AOI accuracy estimate.",
                "No threshold, fusion weight, training sample, mask, object, DEM volume, or production output is modified.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"RESULT_FOLDER: {output}", flush=True)
        print(f"STATUS: {status}", flush=True)
        return 0 if status == "complete" else 2
    finally:
        close_context(context24)
        close_context(context25)


if __name__ == "__main__":
    raise SystemExit(main())
