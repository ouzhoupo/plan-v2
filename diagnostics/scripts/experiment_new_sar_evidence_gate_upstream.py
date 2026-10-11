#!/usr/bin/env python3
"""Selective 2025 new-SAR-coverage evidence gate, upstream only.

Unlike the common-support bound, valid 2025 SAR is retained unless it lies
outside the frozen 2024-06 SAR footprint, lacks SCL/land-use/previous-water
support, and has low (or unavailable, in the inclusive variant) base optical
evidence.  The model structure and shared SAR/shared optical calibration are
otherwise fixed.
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import experiment_a3_dual_pol_upstream as a3  # noqa: E402
import experiment_shared_sar_calibration_upstream as sar_exp  # noqa: E402
import run_combined_upstream_c0_c4 as combined  # noqa: E402


DATE = "2025-05-21"
REFERENCE_AREA_2024_KM2 = 207.6423
BASE_LOW = 0.20
SUPPORT_SOURCE = (
    sar_exp.FROZEN_UPSTREAM / "2024-06-01/c2_source_flag.tif"
)
PREVIOUS_WATER = (
    ROOT
    / "diagnostics/results/shared_sar_calibration_upstream"
    / "20261009_193736_367365"
    / "2024-06-01"
    / "shared_sar_shared_alpha_c4_water_mask.tif"
)
SHARED_BASELINE = (
    ROOT
    / "diagnostics/results/shared_sar_calibration_upstream"
    / "20261009_193736_367365"
    / DATE
    / "shared_sar_shared_alpha_c4_water_mask.tif"
)
COMMON_BOUND = (
    ROOT
    / "diagnostics/results/common_sar_support_upstream"
    / "20261009_232419_505438"
    / DATE
    / "shared_sar_shared_alpha_c4_water_mask.tif"
)
LANDUSE = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
S2 = ROOT / "data/s2/s2_stack_2025-05-21.tif"
OUTPUT_ROOT = ROOT / "diagnostics/results/new_sar_evidence_gate_upstream"
VARIANTS = (
    "clear_low_no_aux_gate",
    "unclear_or_low_no_aux_gate",
)


def aligned_vrt(stack: ExitStack, path: Path, reference):
    source = stack.enter_context(rasterio.open(path))
    return stack.enter_context(
        WarpedVRT(
            source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=Resampling.nearest,
        )
    )


def gate_sar(values, gate: np.ndarray):
    sar = dict(values[0])
    probability = sar["sar_prob"].copy()
    confidence = sar["sar_conf"].copy()
    valid = sar["sar_valid"] & np.isfinite(probability)
    applied = gate & valid
    valid[applied] = False
    probability[applied] = np.nan
    confidence[applied] = 0.0
    sar["sar_valid"] = valid
    sar["sar_prob"] = probability
    sar["sar_conf"] = confidence
    return (sar, *values[1:]), applied


def main() -> int:
    print("Preparing selective new-SAR evidence gate: 2025-05-21", flush=True)
    context, captured = a3.capture_setup(DATE)
    alternate, training_effect = a3.replay_component_split(context, captured)
    if len(context["s1"]) != 1:
        sar_exp.close_context(context)
        raise RuntimeError("Selective gate expects one S1 scene")
    scene = context["s1"][0]
    original_params = dict(scene.params)
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    try:
        scene.params = {**original_params, **sar_exp.REFERENCE_PARAMS}
        reference = context["readers"]["mask"]
        pixel_area = abs(float(reference.transform.a * reference.transform.e))
        mask_paths = {name: output / f"{name}_c4_water_mask.tif" for name in VARIANTS}
        gate_paths = {name: output / f"{name}_gate_mask.tif" for name in VARIANTS}
        counts = {
            name: {
                "gate_pixels": 0,
                "gate_on_shared_water_pixels": 0,
                "water_pixels": 0,
                "added_vs_shared_pixels": 0,
                "removed_vs_shared_pixels": 0,
                "added_vs_current_pixels": 0,
                "removed_vs_current_pixels": 0,
                "added_vs_common_bound_pixels": 0,
                "removed_vs_common_bound_pixels": 0,
            }
            for name in VARIANTS
        }
        validation = {"shared_baseline_mismatch_pixels": 0}

        with ExitStack() as stack:
            support_reader = stack.enter_context(rasterio.open(SUPPORT_SOURCE))
            previous_reader = stack.enter_context(rasterio.open(PREVIOUS_WATER))
            shared_reader = stack.enter_context(rasterio.open(SHARED_BASELINE))
            common_reader = stack.enter_context(rasterio.open(COMMON_BOUND))
            current_reader = stack.enter_context(
                rasterio.open(sar_exp.FROZEN_UPSTREAM / DATE / "c4_water_mask.tif")
            )
            landuse_reader = aligned_vrt(stack, LANDUSE, reference)
            s2_reader = aligned_vrt(stack, S2, reference)
            for source in (
                support_reader, previous_reader, shared_reader, common_reader, current_reader,
            ):
                if (
                    source.shape != reference.shape
                    or source.crs != reference.crs
                    or not source.transform.almost_equals(reference.transform)
                ):
                    raise RuntimeError(f"Grid mismatch: {source.name}")
            mask_writers = {
                name: stack.enter_context(combined.writer(path, reference))
                for name, path in mask_paths.items()
            }
            gate_writers = {
                name: stack.enter_context(combined.writer(path, reference))
                for name, path in gate_paths.items()
            }

            block_size = int(context["fcfg"].get("block_size", 512))
            for index, window in enumerate(
                a3.m.iter_windows(context["grid"].width, context["grid"].height, block_size),
                start=1,
            ):
                base = a3.calc(context, window)
                support_source = support_reader.read(1, window=window)
                reference_support = (support_source == 1) | (support_source == 3)
                previous_water = previous_reader.read(1, window=window) > 0
                scl_water = s2_reader.read(7, window=window) == 6
                landuse_water = landuse_reader.read(1, window=window) == 1
                auxiliary = previous_water | scl_water | landuse_water
                sar_valid = base[0]["sar_valid"] & np.isfinite(base[0]["sar_prob"])
                new_coverage = sar_valid & ~reference_support
                optical = base[1]
                clear = optical["opt_clear"] & np.isfinite(optical["opt_prob_base"])
                low = clear & (optical["opt_prob_base"] < BASE_LOW)
                gates = {
                    "clear_low_no_aux_gate": new_coverage & ~auxiliary & low,
                    "unclear_or_low_no_aux_gate": (
                        new_coverage & ~auxiliary & (~clear | low)
                    ),
                }

                shared_saved = shared_reader.read(1, window=window) > 0
                common_saved = common_reader.read(1, window=window) > 0
                current_saved = current_reader.read(1, window=window) > 0
                baseline_trained = a3.alternate_values(base, context, alternate)
                baseline_values = sar_exp.remix(
                    baseline_trained, context, sar_exp.REFERENCE_BLEND
                )
                baseline_mask, _ = combined.mask_from_values(
                    baseline_values, context, window, a1=True
                )
                validation["shared_baseline_mismatch_pixels"] += int(
                    np.count_nonzero((baseline_mask > 0) != shared_saved)
                )

                for name, proposed_gate in gates.items():
                    gated_base, applied = gate_sar(base, proposed_gate)
                    trained = a3.alternate_values(gated_base, context, alternate)
                    values = sar_exp.remix(trained, context, sar_exp.REFERENCE_BLEND)
                    mask, _ = combined.mask_from_values(values, context, window, a1=True)
                    water = mask > 0
                    counts[name]["gate_pixels"] += int(np.count_nonzero(applied))
                    counts[name]["gate_on_shared_water_pixels"] += int(
                        np.count_nonzero(applied & shared_saved)
                    )
                    counts[name]["water_pixels"] += int(np.count_nonzero(water))
                    counts[name]["added_vs_shared_pixels"] += int(
                        np.count_nonzero(~shared_saved & water)
                    )
                    counts[name]["removed_vs_shared_pixels"] += int(
                        np.count_nonzero(shared_saved & ~water)
                    )
                    counts[name]["added_vs_current_pixels"] += int(
                        np.count_nonzero(~current_saved & water)
                    )
                    counts[name]["removed_vs_current_pixels"] += int(
                        np.count_nonzero(current_saved & ~water)
                    )
                    counts[name]["added_vs_common_bound_pixels"] += int(
                        np.count_nonzero(~common_saved & water)
                    )
                    counts[name]["removed_vs_common_bound_pixels"] += int(
                        np.count_nonzero(common_saved & ~water)
                    )
                    mask_writers[name].write(mask.astype("uint8"), 1, window=window)
                    gate_writers[name].write(applied.astype("uint8"), 1, window=window)

                if index % 100 == 0:
                    print(f"  processed {index} blocks", flush=True)

        variants = {}
        for name in VARIANTS:
            metrics = {
                **counts[name],
                **{
                    key.replace("pixels", "km2"): value * pixel_area / 1e6
                    for key, value in counts[name].items()
                },
            }
            metrics["ratio_2025_to_2024_reference"] = (
                metrics["water_km2"] / REFERENCE_AREA_2024_KM2
            )
            metrics["water_mask"] = str(mask_paths[name])
            metrics["gate_mask"] = str(gate_paths[name])
            variants[name] = metrics

        status = (
            "complete"
            if validation["shared_baseline_mismatch_pixels"] == 0
            else "validation_failed"
        )
        report = {
            "status": status,
            "experiment": "Selective gate for unsupported new 2025 SAR coverage",
            "fixed_2024_reference_area_km2": REFERENCE_AREA_2024_KM2,
            "base_low_bin": BASE_LOW,
            "training_effect": training_effect,
            "validation": validation,
            "variants": variants,
            "limits": [
                "The 0.20 base-optical cut is a coarse mechanism bin, not a selected production threshold.",
                "SCL and base probability share Sentinel-2 input; land-use and previous masks have known limitations.",
                "The clear-only variant leaves optically unclear unsupported new coverage untouched; the inclusive variant gates it.",
                "This is a 2025 Pihe upstream experiment only; no downstream object, volume, or cross-region result is produced.",
                "No production code or production output is modified.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"RESULT_FOLDER: {output}", flush=True)
        print(f"STATUS: {status}", flush=True)
        return 0 if status == "complete" else 2
    finally:
        scene.params = original_params
        sar_exp.close_context(context)


if __name__ == "__main__":
    raise SystemExit(main())
