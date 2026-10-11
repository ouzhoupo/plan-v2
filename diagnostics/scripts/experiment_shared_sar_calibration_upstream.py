#!/usr/bin/env python3
"""Coarse upstream counterfactuals for cross-date SAR calibration.

The original model structure is retained.  The frozen 2024-06 SAR scene
parameters are used as a common diagnostic reference for all dates.  The
date-trained A3 optical model is kept fixed, so the first variant isolates the
direct SAR-to-fusion path.  Additional variants combine the shared SAR scale
with already-audited optical blend bounds to test total mechanism capacity.
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
import run_combined_upstream_c0_c4 as combined  # noqa: E402


DATES = ("2024-05-21", "2024-06-01", "2025-05-21")
FROZEN_UPSTREAM = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/shared_sar_calibration_upstream"
REFERENCE_PARAMS = {
    "th_vv": -13.535,
    "th_vh": -21.5,
    "sigma_vv": 1.2,
    "sigma_vh": 1.23,
}
REFERENCE_BLEND = 0.6142669675465995
VARIANTS = {
    "shared_sar_current_alpha": None,
    "shared_sar_shared_alpha": REFERENCE_BLEND,
    "shared_sar_alpha_035": 0.35,
    "shared_sar_base_only_bound": 0.0,
}


def close_context(context: dict) -> None:
    for reader in context["readers"].values():
        reader.close()
    a3.m.close_scenes(
        context["s1"], context["s2"],
        extras=[context["dem"]] if context["dem"] is not None else [],
    )


def remix(values, context: dict, blend: float):
    sar, optical, model_probability, _, _, _, _, prior = values
    optical_probability = optical["opt_prob_base"].copy()
    good = optical["opt_clear"] & np.isfinite(model_probability)
    optical_probability[good] = np.clip(
        blend * model_probability[good]
        + (1.0 - blend) * optical["opt_prob_base"][good],
        0.0,
        1.0,
    )
    candidate = (
        sar, optical, model_probability, optical_probability,
        values[4], values[5], values[6], prior,
    )
    return combined.a4_refuse(candidate, context)


def process_date(
    date: str,
    output: Path,
    previous_c3_paths: dict[str, Path] | None,
) -> tuple[dict, dict[str, Path]]:
    print(f"Shared-SAR upstream counterfactual: {date}", flush=True)
    context, captured = a3.capture_setup(date)
    alternate, training_effect = a3.replay_component_split(context, captured)
    if len(context["s1"]) != 1:
        close_context(context)
        raise RuntimeError("Shared-SAR audit expects one S1 scene per date")
    scene = context["s1"][0]
    original_params = dict(scene.params)
    try:
        if date == "2024-06-01":
            for key, value in REFERENCE_PARAMS.items():
                if not np.isclose(original_params[key], value, rtol=0.0, atol=1e-12):
                    raise RuntimeError("Reference SAR parameters no longer match 2024-06")
        scene.params = {**original_params, **REFERENCE_PARAMS}

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
                "c3_added_vs_current_pixels": 0,
                "c3_removed_vs_current_pixels": 0,
                "c4_added_vs_current_pixels": 0,
                "c4_removed_vs_current_pixels": 0,
            }
            for name in VARIANTS
        }
        current_counts = {"c3_water_pixels": 0, "c4_water_pixels": 0}

        with ExitStack() as stack:
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

            saved_c3 = stack.enter_context(
                rasterio.open(FROZEN_UPSTREAM / date / "c3_water_mask.tif")
            )
            saved_c4 = stack.enter_context(
                rasterio.open(FROZEN_UPSTREAM / date / "c4_water_mask.tif")
            )
            block_size = int(context["fcfg"].get("block_size", 512))
            for index, window in enumerate(
                a3.m.iter_windows(context["grid"].width, context["grid"].height, block_size),
                start=1,
            ):
                shared_base = a3.calc(context, window)
                shared_trained = a3.alternate_values(shared_base, context, alternate)
                current_c3 = saved_c3.read(1, window=window) > 0
                current_c4 = saved_c4.read(1, window=window) > 0
                current_counts["c3_water_pixels"] += int(np.count_nonzero(current_c3))
                current_counts["c4_water_pixels"] += int(np.count_nonzero(current_c4))

                for name, configured_blend in VARIANTS.items():
                    blend = context["blend"] if configured_blend is None else configured_blend
                    values = remix(shared_trained, context, blend)
                    c3, aoi = combined.mask_from_values(values, context, window, a1=True)
                    c4 = c3.copy()
                    if date == "2024-06-01":
                        prior_water = previous_readers[name].read(1, window=window) > 0
                        scl = scl_vrt.read(7, window=window)
                        eligible = aoi & (values[5] == 4) & prior_water & (scl == 6)
                        c4[eligible] = 1

                    c3_bool = c3 > 0
                    c4_bool = c4 > 0
                    counts[name]["c3_water_pixels"] += int(np.count_nonzero(c3_bool))
                    counts[name]["c4_water_pixels"] += int(np.count_nonzero(c4_bool))
                    counts[name]["c3_added_vs_current_pixels"] += int(
                        np.count_nonzero(~current_c3 & c3_bool)
                    )
                    counts[name]["c3_removed_vs_current_pixels"] += int(
                        np.count_nonzero(current_c3 & ~c3_bool)
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
            "original_sar_parameters": original_params,
            "shared_sar_parameters": REFERENCE_PARAMS,
            "current_optical_blend": context["blend"],
            "model_quality": context["quality"],
            "training_effect": training_effect,
            "current_reference": {
                **current_counts,
                **{
                    key.replace("pixels", "km2"): value * pixel_area / 1e6
                    for key, value in current_counts.items()
                },
            },
            "variants": variants,
        }, c3_paths)
    finally:
        scene.params = original_params
        close_context(context)


def main() -> int:
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "experiment": "Shared SAR calibration upstream counterfactual",
        "reference_parameters": REFERENCE_PARAMS,
        "variants": {
            "shared_sar_current_alpha": "Only the direct SAR path uses the common reference; date-trained optical model and alpha stay fixed.",
            "shared_sar_shared_alpha": "Common SAR reference plus the frozen 2024-06 optical alpha.",
            "shared_sar_alpha_035": "Common SAR reference plus the existing optical formula floor.",
            "shared_sar_base_only_bound": "Common SAR reference plus base optical only; diagnostic lower bound.",
        },
        "dates": {},
        "limits": [
            "The 2024-06 SAR parameters are a diagnostic common scale, not selected production constants.",
            "The date-trained A3 optical classifier is frozen; SAR pseudo-label training is not retrained in this audit.",
            "The first variant isolates direct SAR fusion capacity; combined variants test coarse total mechanism capacity.",
            "Fusion logic, A1, causal A2, hysteresis, morphology and all downstream modules remain structurally unchanged.",
            "No production code or production output is modified.",
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

    report["status"] = "complete"
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {report['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
