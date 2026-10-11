#!/usr/bin/env python3
"""Coarse upstream counterfactuals for the optical internal blend only.

The model structure, A3-trained optical model, SAR probability, fusion weights,
hysteresis, morphology, A1 confidence handling and causal A2 fill are retained.
Only alpha in optical_p = alpha*model_p + (1-alpha)*base_optical_p is varied.
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
OUTPUT_ROOT = ROOT / "diagnostics/results/shared_optical_blend_upstream"
SHARED_CONSERVATIVE_BLEND = 0.6142669675465995
VARIANTS = {
    "current_date_specific": None,
    "shared_conservative": SHARED_CONSERVATIVE_BLEND,
    "balanced_050": 0.50,
    "formula_floor_035": 0.35,
    "base_only_bound": 0.0,
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
    print(f"Preparing optical-blend upstream counterfactual: {date}", flush=True)
    context, captured = a3.capture_setup(date)
    alternate, training_effect = a3.replay_component_split(context, captured)
    try:
        if date == "2024-06-01" and not np.isclose(
            context["blend"], SHARED_CONSERVATIVE_BLEND, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("Frozen shared blend no longer matches 2024-06 reference")

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
        validations = {
            "current_c3_mismatch_pixels": 0,
            "current_c4_mismatch_pixels": 0,
            "a2_current_added_pixels": 0,
        }

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

            saved_c3 = rasterio.open(FROZEN_UPSTREAM / date / "c3_water_mask.tif")
            saved_c4 = rasterio.open(FROZEN_UPSTREAM / date / "c4_water_mask.tif")
            stack.callback(saved_c3.close)
            stack.callback(saved_c4.close)
            block_size = int(context["fcfg"].get("block_size", 512))
            for index, window in enumerate(
                a3.m.iter_windows(context["grid"].width, context["grid"].height, block_size),
                start=1,
            ):
                base = a3.calc(context, window)
                trained = a3.alternate_values(base, context, alternate)
                c3_masks = {}
                c4_masks = {}
                sources = {}
                for name, configured_blend in VARIANTS.items():
                    blend = context["blend"] if configured_blend is None else configured_blend
                    values = remix(trained, context, blend)
                    c3, aoi = combined.mask_from_values(values, context, window, a1=True)
                    c4 = c3.copy()
                    if date == "2024-06-01":
                        prior_water = previous_readers[name].read(1, window=window) > 0
                        scl = scl_vrt.read(7, window=window)
                        eligible = aoi & (values[5] == 4) & prior_water & (scl == 6)
                        c4[eligible] = 1
                        if name == "current_date_specific":
                            validations["a2_current_added_pixels"] += int(np.count_nonzero(eligible))
                    c3_masks[name] = c3
                    c4_masks[name] = c4
                    sources[name] = values[5]

                current_c3 = c3_masks["current_date_specific"] > 0
                current_c4 = c4_masks["current_date_specific"] > 0
                for name in VARIANTS:
                    c3 = c3_masks[name] > 0
                    c4 = c4_masks[name] > 0
                    counts[name]["c3_water_pixels"] += int(np.count_nonzero(c3))
                    counts[name]["c4_water_pixels"] += int(np.count_nonzero(c4))
                    counts[name]["c3_added_vs_current_pixels"] += int(np.count_nonzero(~current_c3 & c3))
                    counts[name]["c3_removed_vs_current_pixels"] += int(np.count_nonzero(current_c3 & ~c3))
                    counts[name]["c4_added_vs_current_pixels"] += int(np.count_nonzero(~current_c4 & c4))
                    counts[name]["c4_removed_vs_current_pixels"] += int(np.count_nonzero(current_c4 & ~c4))
                    c3_writers[name].write(c3.astype("uint8"), 1, window=window)
                    c4_writers[name].write(c4.astype("uint8"), 1, window=window)

                validations["current_c3_mismatch_pixels"] += int(
                    np.count_nonzero(current_c3 != (saved_c3.read(1, window=window) > 0))
                )
                validations["current_c4_mismatch_pixels"] += int(
                    np.count_nonzero(current_c4 != (saved_c4.read(1, window=window) > 0))
                )
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
            "current_optical_blend": context["blend"],
            "model_quality": context["quality"],
            "training_effect": training_effect,
            "variants": variants,
            "validations": validations,
        }, c3_paths)
    finally:
        close_context(context)


def main() -> int:
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "experiment": "Shared optical internal blend upstream counterfactual",
        "single_changed_quantity": (
            "alpha in optical_p = alpha*SAR-pseudolabel-trained model_p + "
            "(1-alpha)*base_optical_p"
        ),
        "variants": {
            "current_date_specific": "Frozen date-specific alpha reference.",
            "shared_conservative": "Use the frozen 2024-06 alpha on all dates.",
            "balanced_050": "Equal trained-model and base-optical contribution.",
            "formula_floor_035": "Existing formula's lower alpha bound.",
            "base_only_bound": "Diagnostic lower bound; no trained-model contribution.",
        },
        "dates": {},
        "limits": [
            "This is a coarse sensitivity set, not a selected production alpha.",
            "The base-only row is a diagnostic bound, not a proposed final model.",
            "All SAR calculations, A3 trained models, fusion weights, thresholds, A1, A2, hysteresis and morphology remain fixed.",
            "Only upstream masks are produced; no vector deduction, object gates, land-use gate, DEM volume or survey evaluation is run here.",
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

    validations_pass = all(
        date_report["validations"]["current_c3_mismatch_pixels"] == 0
        and date_report["validations"]["current_c4_mismatch_pixels"] == 0
        for date_report in report["dates"].values()
    )
    report["status"] = "complete" if validations_pass else "validation_failed"
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print(f"STATUS: {report['status']}", flush=True)
    return 0 if validations_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
