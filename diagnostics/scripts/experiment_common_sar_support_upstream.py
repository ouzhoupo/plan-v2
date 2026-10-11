#!/usr/bin/env python3
"""Diagnose the capacity of cross-date SAR coverage change.

The 2025 SAR branch is restricted to pixels that had finite SAR support in the
frozen 2024-06 C2 result.  The 2024 dates remain unchanged from the shared-SAR
calibration experiment.  This is a common-observation diagnostic bound, not a
proposal to discard valid 2025 observations in production.
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


DATES = ("2024-05-21", "2024-06-01", "2025-05-21")
FROZEN_UPSTREAM = sar_exp.FROZEN_UPSTREAM
SHARED_SAR_RESULT = (
    ROOT
    / "diagnostics/results/shared_sar_calibration_upstream"
    / "20261009_193736_367365"
)
SUPPORT_SOURCE = FROZEN_UPSTREAM / "2024-06-01/c2_source_flag.tif"
OUTPUT_ROOT = ROOT / "diagnostics/results/common_sar_support_upstream"
VARIANTS = sar_exp.VARIANTS


def restrict_sar(values, support: np.ndarray):
    sar = dict(values[0])
    valid_before = sar["sar_valid"] & np.isfinite(sar["sar_prob"])
    valid_after = valid_before & support
    probability = sar["sar_prob"].copy()
    confidence = sar["sar_conf"].copy()
    probability[~valid_after] = np.nan
    confidence[~valid_after] = 0.0
    sar["sar_valid"] = valid_after
    sar["sar_prob"] = probability
    sar["sar_conf"] = confidence
    return (sar, *values[1:]), valid_before, valid_after


def process_date(
    date: str,
    output: Path,
    previous_c3_paths: dict[str, Path] | None,
) -> tuple[dict, dict[str, Path]]:
    print(f"Common-SAR-support upstream counterfactual: {date}", flush=True)
    context, captured = a3.capture_setup(date)
    alternate, training_effect = a3.replay_component_split(context, captured)
    if len(context["s1"]) != 1:
        sar_exp.close_context(context)
        raise RuntimeError("Common-support audit expects one S1 scene per date")
    scene = context["s1"][0]
    original_params = dict(scene.params)
    try:
        scene.params = {**original_params, **sar_exp.REFERENCE_PARAMS}
        date_dir = output / date
        date_dir.mkdir(parents=True, exist_ok=False)
        reference = context["readers"]["mask"]
        pixel_area = abs(float(reference.transform.a * reference.transform.e))
        c3_paths = {name: date_dir / f"{name}_c3_water_mask.tif" for name in VARIANTS}
        c4_paths = {name: date_dir / f"{name}_c4_water_mask.tif" for name in VARIANTS}
        counts = {
            name: {
                "c3_water_pixels": 0,
                "c4_water_pixels": 0,
                "c3_added_vs_shared_sar_pixels": 0,
                "c3_removed_vs_shared_sar_pixels": 0,
                "c4_added_vs_shared_sar_pixels": 0,
                "c4_removed_vs_shared_sar_pixels": 0,
                "c4_added_vs_current_pixels": 0,
                "c4_removed_vs_current_pixels": 0,
            }
            for name in VARIANTS
        }
        support_counts = {
            "reference_support_pixels": 0,
            "target_sar_valid_before_pixels": 0,
            "target_sar_valid_after_pixels": 0,
            "target_sar_valid_removed_pixels": 0,
        }

        with ExitStack() as stack:
            support_reader = stack.enter_context(rasterio.open(SUPPORT_SOURCE))
            current_c4_reader = stack.enter_context(
                rasterio.open(FROZEN_UPSTREAM / date / "c4_water_mask.tif")
            )
            shared_c3_readers = {
                name: stack.enter_context(
                    rasterio.open(SHARED_SAR_RESULT / date / f"{name}_c3_water_mask.tif")
                )
                for name in VARIANTS
            }
            shared_c4_readers = {
                name: stack.enter_context(
                    rasterio.open(SHARED_SAR_RESULT / date / f"{name}_c4_water_mask.tif")
                )
                for name in VARIANTS
            }
            c3_writers = {
                name: stack.enter_context(combined.writer(path, reference))
                for name, path in c3_paths.items()
            }
            c4_writers = {
                name: stack.enter_context(combined.writer(path, reference))
                for name, path in c4_paths.items()
            }
            previous_readers = {}
            scl_vrt = None
            if date == "2024-06-01":
                if previous_c3_paths is None:
                    raise RuntimeError("2024-06 requires candidate 2024-05 C3 masks")
                previous_readers = {
                    name: stack.enter_context(rasterio.open(path))
                    for name, path in previous_c3_paths.items()
                }
                s2_source = stack.enter_context(
                    rasterio.open(ROOT / "data/s2/s2_stack_2024-06-01.tif")
                )
                scl_vrt = stack.enter_context(
                    WarpedVRT(
                        s2_source,
                        crs=reference.crs,
                        transform=reference.transform,
                        width=reference.width,
                        height=reference.height,
                        resampling=Resampling.nearest,
                    )
                )

            block_size = int(context["fcfg"].get("block_size", 512))
            for index, window in enumerate(
                a3.m.iter_windows(context["grid"].width, context["grid"].height, block_size),
                start=1,
            ):
                support_source = support_reader.read(1, window=window)
                reference_support = (support_source == 1) | (support_source == 3)
                support_counts["reference_support_pixels"] += int(
                    np.count_nonzero(reference_support)
                )

                shared_base = a3.calc(context, window)
                transform = a3.m.window_transform(context["grid"].transform, window)
                aoi = a3.m.rasterize_geom_mask(
                    context["grid"].aoi_geom,
                    shared_base[0]["sar_valid"].shape,
                    transform,
                )
                if date == "2025-05-21":
                    shared_base, valid_before, valid_after = restrict_sar(
                        shared_base, reference_support
                    )
                else:
                    valid_before = shared_base[0]["sar_valid"] & np.isfinite(
                        shared_base[0]["sar_prob"]
                    )
                    valid_after = valid_before
                support_counts["target_sar_valid_before_pixels"] += int(
                    np.count_nonzero(valid_before & aoi)
                )
                support_counts["target_sar_valid_after_pixels"] += int(
                    np.count_nonzero(valid_after & aoi)
                )
                support_counts["target_sar_valid_removed_pixels"] += int(
                    np.count_nonzero(valid_before & ~valid_after & aoi)
                )

                trained = a3.alternate_values(shared_base, context, alternate)
                current_c4 = current_c4_reader.read(1, window=window) > 0
                for name, configured_blend in VARIANTS.items():
                    blend = context["blend"] if configured_blend is None else configured_blend
                    values = sar_exp.remix(trained, context, blend)
                    c3, aoi = combined.mask_from_values(values, context, window, a1=True)
                    c4 = c3.copy()
                    if date == "2024-06-01":
                        prior_water = previous_readers[name].read(1, window=window) > 0
                        scl = scl_vrt.read(7, window=window)
                        eligible = aoi & (values[5] == 4) & prior_water & (scl == 6)
                        c4[eligible] = 1

                    c3_bool = c3 > 0
                    c4_bool = c4 > 0
                    shared_c3 = shared_c3_readers[name].read(1, window=window) > 0
                    shared_c4 = shared_c4_readers[name].read(1, window=window) > 0
                    counts[name]["c3_water_pixels"] += int(np.count_nonzero(c3_bool))
                    counts[name]["c4_water_pixels"] += int(np.count_nonzero(c4_bool))
                    counts[name]["c3_added_vs_shared_sar_pixels"] += int(
                        np.count_nonzero(~shared_c3 & c3_bool)
                    )
                    counts[name]["c3_removed_vs_shared_sar_pixels"] += int(
                        np.count_nonzero(shared_c3 & ~c3_bool)
                    )
                    counts[name]["c4_added_vs_shared_sar_pixels"] += int(
                        np.count_nonzero(~shared_c4 & c4_bool)
                    )
                    counts[name]["c4_removed_vs_shared_sar_pixels"] += int(
                        np.count_nonzero(shared_c4 & ~c4_bool)
                    )
                    counts[name]["c4_added_vs_current_pixels"] += int(
                        np.count_nonzero(~current_c4 & c4_bool)
                    )
                    counts[name]["c4_removed_vs_current_pixels"] += int(
                        np.count_nonzero(current_c4 & ~c4_bool)
                    )
                    c3_writers[name].write(c3.astype("uint8"), 1, window=window)
                    c4_writers[name].write(c4.astype("uint8"), 1, window=window)

                if index % 100 == 0:
                    print(f"  {date}: processed {index} blocks", flush=True)

        variants = {}
        for name, configured_blend in VARIANTS.items():
            variants[name] = {
                "optical_blend": context["blend"] if configured_blend is None else configured_blend,
                **counts[name],
                **{
                    key.replace("pixels", "km2"): value * pixel_area / 1e6
                    for key, value in counts[name].items()
                },
                "c3_mask": str(c3_paths[name]),
                "c4_mask": str(c4_paths[name]),
            }
        return ({
            "date": date,
            "support_restricted": date == "2025-05-21",
            "original_sar_parameters": original_params,
            "shared_sar_parameters": sar_exp.REFERENCE_PARAMS,
            "support_counts": {
                **support_counts,
                **{
                    key.replace("pixels", "km2"): value * pixel_area / 1e6
                    for key, value in support_counts.items()
                },
            },
            "training_effect": training_effect,
            "variants": variants,
        }, c3_paths)
    finally:
        scene.params = original_params
        sar_exp.close_context(context)


def main() -> int:
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "experiment": "2025 SAR common-observation-support upstream bound",
        "reference_support": str(SUPPORT_SOURCE),
        "dates": {},
        "limits": [
            "Only 2025 SAR validity is restricted to the frozen 2024-06 finite-SAR footprint.",
            "The common footprint is an attribution bound, not a production rule and not a water truth mask.",
            "2024 dates reproduce the preceding shared-SAR calibration candidates for paired comparison.",
            "Date-trained optical classifiers remain frozen; no pseudo-label retraining is performed.",
            "No downstream object, land-use, DEM volume, survey metric, production code or production output is modified.",
        ],
    }
    previous = None
    for date in DATES:
        date_report, c3_paths = process_date(date, output, previous)
        report["dates"][date] = date_report
        if date == "2024-05-21":
            previous = c3_paths
        (output / "summary.partial.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    unchanged = all(
        report["dates"][date]["variants"][name]["c4_added_vs_shared_sar_pixels"] == 0
        and report["dates"][date]["variants"][name]["c4_removed_vs_shared_sar_pixels"] == 0
        for date in ("2024-05-21", "2024-06-01")
        for name in VARIANTS
    )
    report["validation_2024_matches_shared_sar"] = unchanged
    report["status"] = "complete" if unchanged else "validation_failed"
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {report['status']}", flush=True)
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
