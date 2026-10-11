#!/usr/bin/env python3
"""Audit independent/auxiliary evidence in the 2025 coverage-effect scope.

The scope is the shared-SAR/shared-alpha C4 water removed by the common-support
diagnostic.  Evidence is summarized without selecting a rescue or rejection
rule and without changing any model output.
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
import pihe_component_helpers as helper  # noqa: E402


DATE = "2025-05-21"
SHARED = (
    ROOT
    / "diagnostics/results/shared_sar_calibration_upstream"
    / "20261009_193736_367365"
    / DATE
    / "shared_sar_shared_alpha_c4_water_mask.tif"
)
COMMON = (
    ROOT
    / "diagnostics/results/common_sar_support_upstream"
    / "20261009_232419_505438"
    / DATE
    / "shared_sar_shared_alpha_c4_water_mask.tif"
)
PREVIOUS = (
    ROOT
    / "diagnostics/results/common_sar_support_upstream"
    / "20261009_232419_505438"
    / "2024-06-01"
    / "shared_sar_shared_alpha_c4_water_mask.tif"
)
HISTORICAL = ROOT / "output/source_water_masks/water_mask_2025-04-11_pihe.tif"
LANDUSE = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
S2 = ROOT / "data/s2/s2_stack_2025-05-21.tif"
OUTPUT_ROOT = ROOT / "diagnostics/results/new_sar_coverage_evidence"
BASE_BINS = (0.20, 0.50, 0.70)


def aligned_vrt(stack: ExitStack, path: Path, reference, resampling=Resampling.nearest):
    source = stack.enter_context(rasterio.open(path))
    return stack.enter_context(
        WarpedVRT(
            source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=resampling,
        )
    )


def new_accumulator(historical_available: bool) -> dict:
    return {
        "pixels": 0,
        "optical_clear": 0,
        "base_valid": 0,
        "base_values": [],
        "historical_available": historical_available,
        "base_ge": {str(value): 0 for value in BASE_BINS},
        "scl_water": 0,
        "landuse_water": 0,
        "historical_water": 0,
        "previous_water": 0,
        "any_auxiliary_support": 0,
        "base_ge_050_or_auxiliary": 0,
        "base_ge_070_or_auxiliary": 0,
        "no_auxiliary_and_base_lt_020": 0,
    }


def update(acc: dict, scope: np.ndarray, optical: dict, scl, landuse, historical, previous):
    count = int(np.count_nonzero(scope))
    acc["pixels"] += count
    if not count:
        return
    clear = optical["opt_clear"]
    base = optical["opt_prob_base"]
    valid = scope & clear & np.isfinite(base)
    values = base[valid].astype("float32")
    acc["optical_clear"] += int(np.count_nonzero(scope & clear))
    acc["base_valid"] += int(values.size)
    if values.size:
        acc["base_values"].append(values)
    for threshold in BASE_BINS:
        acc["base_ge"][str(threshold)] += int(
            np.count_nonzero(valid & (base >= threshold))
        )

    scl_water = scope & (scl == 6)
    landuse_water = scope & (landuse == 1)
    historical_water = (
        scope & (historical > 0)
        if historical is not None
        else np.zeros(scope.shape, dtype=bool)
    )
    previous_water = scope & (previous > 0)
    auxiliary = scl_water | landuse_water | historical_water | previous_water
    base_050 = valid & (base >= 0.50)
    base_070 = valid & (base >= 0.70)
    base_lt_020 = (~valid) | (base < 0.20)
    acc["scl_water"] += int(np.count_nonzero(scl_water))
    acc["landuse_water"] += int(np.count_nonzero(landuse_water))
    acc["historical_water"] += int(np.count_nonzero(historical_water))
    acc["previous_water"] += int(np.count_nonzero(previous_water))
    acc["any_auxiliary_support"] += int(np.count_nonzero(scope & auxiliary))
    acc["base_ge_050_or_auxiliary"] += int(
        np.count_nonzero(scope & (base_050 | auxiliary))
    )
    acc["base_ge_070_or_auxiliary"] += int(
        np.count_nonzero(scope & (base_070 | auxiliary))
    )
    acc["no_auxiliary_and_base_lt_020"] += int(
        np.count_nonzero(scope & ~auxiliary & base_lt_020)
    )


def finish(acc: dict, pixel_area: float) -> dict:
    pixels = acc["pixels"]
    values = (
        np.concatenate(acc["base_values"])
        if acc["base_values"]
        else np.array([], dtype="float32")
    )
    result = {
        "pixels": pixels,
        "area_km2": pixels * pixel_area / 1e6,
        "optical_clear_fraction": acc["optical_clear"] / pixels if pixels else None,
        "base_valid_fraction": acc["base_valid"] / pixels if pixels else None,
        "base_probability_mean": float(values.mean(dtype=np.float64)) if values.size else None,
        "base_probability_quantiles": (
            {
                str(q): float(value)
                for q, value in zip(
                    (0.10, 0.25, 0.50, 0.75, 0.90),
                    np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90)),
                )
            }
            if values.size else {}
        ),
    }
    for threshold in BASE_BINS:
        result[f"base_ge_{threshold:.2f}_fraction_of_scope"] = (
            acc["base_ge"][str(threshold)] / pixels if pixels else None
        )
    for key in (
        "scl_water",
        "landuse_water",
        "previous_water",
        "any_auxiliary_support",
        "base_ge_050_or_auxiliary",
        "base_ge_070_or_auxiliary",
        "no_auxiliary_and_base_lt_020",
    ):
        result[f"{key}_fraction"] = acc[key] / pixels if pixels else None
        result[f"{key}_area_km2"] = acc[key] * pixel_area / 1e6
    if acc["historical_available"]:
        result["historical_water_fraction"] = (
            acc["historical_water"] / pixels if pixels else None
        )
        result["historical_water_area_km2"] = (
            acc["historical_water"] * pixel_area / 1e6
        )
    else:
        result["historical_water_fraction"] = None
        result["historical_water_area_km2"] = None
    return result


def main() -> int:
    print("Preparing 2025 optical context", flush=True)
    context = helper.setup(DATE)
    historical_available = HISTORICAL.exists()
    accumulators = {
        "coverage_removed": new_accumulator(historical_available),
        "coverage_retained": new_accumulator(historical_available),
        "coverage_added": new_accumulator(historical_available),
    }
    try:
        with ExitStack() as stack:
            shared = stack.enter_context(rasterio.open(SHARED))
            common = stack.enter_context(rasterio.open(COMMON))
            previous = stack.enter_context(rasterio.open(PREVIOUS))
            for source in (common, previous):
                if (
                    source.shape != shared.shape
                    or source.crs != shared.crs
                    or not source.transform.almost_equals(shared.transform)
                ):
                    raise RuntimeError(f"Grid mismatch: {source.name}")
            historical = (
                aligned_vrt(stack, HISTORICAL, shared)
                if historical_available else None
            )
            landuse = aligned_vrt(stack, LANDUSE, shared)
            s2 = aligned_vrt(stack, S2, shared)
            pixel_area = abs(float(shared.transform.a * shared.transform.e))
            block_size = int(context["fcfg"].get("block_size", 512))
            for index, window in enumerate(
                helper.m.iter_windows(context["grid"].width, context["grid"].height, block_size),
                start=1,
            ):
                before = shared.read(1, window=window) > 0
                after = common.read(1, window=window) > 0
                scopes = {
                    "coverage_removed": before & ~after,
                    "coverage_retained": before & after,
                    "coverage_added": ~before & after,
                }
                if not any(np.any(scope) for scope in scopes.values()):
                    continue
                optical = helper.calc(context, window)[1]
                scl = s2.read(7, window=window)
                landuse_values = landuse.read(1, window=window)
                historical_values = (
                    historical.read(1, window=window)
                    if historical is not None else None
                )
                previous_values = previous.read(1, window=window)
                for name, scope in scopes.items():
                    update(
                        accumulators[name], scope, optical, scl, landuse_values,
                        historical_values, previous_values,
                    )
                if index % 100 == 0:
                    print(f"  processed {index} blocks", flush=True)

        scopes = {
            name: finish(accumulator, pixel_area)
            for name, accumulator in accumulators.items()
        }
        if scopes["coverage_removed"]["pixels"] != 1_931_250:
            raise RuntimeError("Coverage-removed scope no longer matches paired experiment")
        output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "complete",
            "experiment": "Auxiliary evidence in 2025 common-support change scopes",
            "evidence_availability": {
                "historical_water_2025_04_11": historical_available,
                "historical_water_path": str(HISTORICAL),
                "historical_water_note": (
                    "Available and included."
                    if historical_available
                    else "Configured path is absent; excluded rather than substituted with a target-date result."
                ),
            },
            "scopes": scopes,
            "limits": [
                "All probability cut points are descriptive bins, not selected rules.",
                "SCL and base optical probability share Sentinel-2 input and are not independent of each other.",
                "Historical and previous-date masks are model outputs, not field truth.",
                "The configured 2025-04-11 historical mask is optional here; if absent, its metrics are null and it is excluded from combined evidence.",
                "The land-use water layer has known spatial and semantic limitations.",
                "The audit sizes evidence combinations but does not restore or delete pixels, run downstream modules, or calculate volume.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"RESULT_FOLDER: {output}", flush=True)
        print("STATUS: complete", flush=True)
        return 0
    finally:
        for reader in context["readers"].values():
            reader.close()
        helper.m.close_scenes(
            context["s1"], context["s2"],
            extras=[context["dem"]] if context["dem"] is not None else [],
        )


if __name__ == "__main__":
    raise SystemExit(main())
