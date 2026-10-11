#!/usr/bin/env python3
"""Audit current-date evidence for Shihe single-history rescue objects.

Two-history objects form the strong tier.  Objects missing 2024-05-01 support
are tested with existing target-date evidence (SCL water, dual-source fusion,
probability, and confidence).  This is diagnostic only and does not change the
production mask or thresholds.
"""

from __future__ import annotations

import csv
import json
import math
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RESULT = (
    ROOT / "diagnostics/results/shihe_two_previous_support"
    / "20261010_210728_559984/shihe_candidates_two_previous_support.csv"
)
BRIDGE_DIR = (
    ROOT / "diagnostics/results/temporal_shape_cross_region_bridge"
    / "20261010_194531_518591/shihe"
)
AREA_GATE = BRIDGE_DIR / "original_v75_bridge_after_area_gate.tif"
PROBABILITY = ROOT / "output/20240601/shihe/water_prob_2024-06-01.tif"
CONFIDENCE = ROOT / "output/20240601/shihe/confidence_2024-06-01.tif"
SOURCE_FLAG = ROOT / "output/20240601/shihe/source_flag_2024-06-01.tif"
S2 = ROOT / "data/s2/s2_stack_2024-06-01.tif"
BASELINE = (
    ROOT / "diagnostics/results/cross_region_baseline_trends"
    / "20261010_202616_329605/summary.json"
)
MANUAL = (
    ROOT / "diagnostics/results/cross_region_manual_totals"
    / "20261010_201437_787088/summary.json"
)
OUTPUT_ROOT = ROOT / "diagnostics/results/shihe_tiered_current_evidence"
DATE_24 = "2024-06-01"
DATE_25 = "2025-05-21"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def evaluate(model_24: float, model_25: float, manual_24: float, manual_25: float) -> dict:
    ratio = model_25 / model_24
    manual_ratio = manual_25 / manual_24
    errors = (abs(model_24 - manual_24), abs(model_25 - manual_25))
    return {
        "model_ratio": ratio,
        "manual_ratio": manual_ratio,
        "direction_matches": (ratio < 1) == (manual_ratio < 1),
        "trend_log_error": abs(math.log(ratio / manual_ratio)),
        "total_absolute_error_wan_m3": sum(errors),
        "mape_percent": 50.0 * (errors[0] / manual_24 + errors[1] / manual_25),
    }


def aligned(stack: ExitStack, path: Path, reference, resampling: Resampling):
    source = stack.enter_context(rasterio.open(path))
    return stack.enter_context(WarpedVRT(
        source,
        crs=reference.crs,
        transform=reference.transform,
        width=reference.width,
        height=reference.height,
        resampling=resampling,
    ))


def main() -> int:
    source_rows = {int(row["object_id"]): row for row in read_csv(SOURCE_RESULT)}
    with rasterio.open(AREA_GATE) as reference:
        area_gate = reference.read(1) > 0
        labels, count = ndimage.label(
            area_gate,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
        )
        with ExitStack() as stack:
            probability = aligned(stack, PROBABILITY, reference, Resampling.bilinear).read(1).astype("float32")
            confidence = aligned(stack, CONFIDENCE, reference, Resampling.bilinear).read(1).astype("float32")
            source_flag = aligned(stack, SOURCE_FLAG, reference, Resampling.nearest).read(1).astype("uint8")
            s2 = aligned(stack, S2, reference, Resampling.nearest)
            if s2.count < 7:
                raise RuntimeError("Target S2 stack is missing SCL band 7")
            scl = s2.read(7)

    pixels = np.bincount(labels.ravel(), minlength=count + 1).astype("float64")
    scl_water = np.bincount(labels[scl == 6], minlength=count + 1).astype("float64")
    source_counts = {
        code: np.bincount(labels[source_flag == code], minlength=count + 1).astype("float64")
        for code in (1, 2, 3, 4)
    }
    high_prob = np.bincount(
        labels[np.isfinite(probability) & (probability >= 0.93)], minlength=count + 1
    ).astype("float64")
    finite_prob = np.isfinite(probability)
    finite_conf = np.isfinite(confidence)
    prob_sum = np.bincount(labels[finite_prob], weights=probability[finite_prob], minlength=count + 1)
    prob_n = np.bincount(labels[finite_prob], minlength=count + 1)
    conf_sum = np.bincount(labels[finite_conf], weights=confidence[finite_conf], minlength=count + 1)
    conf_n = np.bincount(labels[finite_conf], minlength=count + 1)

    objects = []
    for object_id, row in source_rows.items():
        n = pixels[object_id]
        if n <= 0:
            raise RuntimeError(f"Missing label {object_id}")
        item = {
            "object_id": object_id,
            "area_m2": float(row["area_m2"]),
            "may01_support": float(row["may01_support"]),
            "may21_support": float(row["may21_support"]),
            "future_proxy": truth(row["future_proxy"]),
            "v75_wan_m3": float(row["v75_wan_m3"]),
            "scl_water_fraction": scl_water[object_id] / n,
            "source_1_sar_only_fraction": source_counts[1][object_id] / n,
            "source_2_optical_only_fraction": source_counts[2][object_id] / n,
            "source_3_both_fraction": source_counts[3][object_id] / n,
            "source_4_prior_only_fraction": source_counts[4][object_id] / n,
            "high_probability_fraction": high_prob[object_id] / n,
            "mean_probability": prob_sum[object_id] / prob_n[object_id] if prob_n[object_id] else float("nan"),
            "mean_confidence": conf_sum[object_id] / conf_n[object_id] if conf_n[object_id] else float("nan"),
        }
        objects.append(item)

    def strong(row):
        return row["may01_support"] >= 0.10

    variants = [
        ("strong_two_history_only", lambda row: strong(row)),
        ("plus_single_scl_ge_010", lambda row: strong(row) or row["scl_water_fraction"] >= 0.10),
        ("plus_single_scl_ge_025", lambda row: strong(row) or row["scl_water_fraction"] >= 0.25),
        ("plus_single_both_source_ge_025", lambda row: strong(row) or row["source_3_both_fraction"] >= 0.25),
        ("plus_single_scl010_or_both025", lambda row: strong(row) or row["scl_water_fraction"] >= 0.10 or row["source_3_both_fraction"] >= 0.25),
        ("plus_single_high_prob_ge_025", lambda row: strong(row) or row["high_probability_fraction"] >= 0.25),
    ]

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["regional"]["shihe"]["model"]
    manual = json.loads(MANUAL.read_text(encoding="utf-8"))["regional_trends"]["shihe"]
    model_24 = float(baseline[DATE_24]["v75_wan_m3"])
    model_25 = float(baseline[DATE_25]["v75_wan_m3"])
    manual_scenarios = {
        "source_total": (float(manual["source_total"][DATE_24]), float(manual["source_total"][DATE_25])),
        "component_sum": (float(manual["component_sum"][DATE_24]), float(manual["component_sum"][DATE_25])),
    }
    before = {name: evaluate(model_24, model_25, *values) for name, values in manual_scenarios.items()}
    total_volume = sum(row["v75_wan_m3"] for row in objects)
    results = []
    for name, predicate in variants:
        selected = [row for row in objects if predicate(row)]
        increment = sum(row["v75_wan_m3"] for row in selected)
        future = [row for row in selected if row["future_proxy"]]
        result = {
            "variant": name,
            "objects": len(selected),
            "area_km2": sum(row["area_m2"] for row in selected) / 1e6,
            "v75_increment_wan_m3": increment,
            "retained_volume_fraction": increment / total_volume,
            "future_proxy_object_fraction": len(future) / len(selected) if selected else 0.0,
            "future_proxy_volume_fraction": sum(row["v75_wan_m3"] for row in future) / increment if increment else 0.0,
            "evaluation": {},
        }
        for scenario, manual_values in manual_scenarios.items():
            after = evaluate(model_24 + increment, model_25, *manual_values)
            after["trend_log_error_reduction"] = before[scenario]["trend_log_error"] - after["trend_log_error"]
            after["total_absolute_error_reduction_wan_m3"] = before[scenario]["total_absolute_error_wan_m3"] - after["total_absolute_error_wan_m3"]
            result["evaluation"][scenario] = after
        results.append(result)

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    object_table = output / "shihe_tiered_current_evidence_objects.csv"
    with object_table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    flat_rows = []
    for result in results:
        flat = {key: value for key, value in result.items() if key != "evaluation"}
        for scenario, values in result["evaluation"].items():
            for key, value in values.items():
                flat[f"{scenario}_{key}"] = value
        flat_rows.append(flat)
    variant_table = output / "tiered_current_evidence_variants.csv"
    with variant_table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    report = {
        "status": "complete",
        "experiment": "Shihe tiered historical and target-date evidence audit",
        "source_flag_meaning": {"1": "SAR only", "2": "optical only", "3": "both", "4": "prior only"},
        "strong_tier": "2024-05-01 support >=0.10 in addition to existing 2024-05-21 support >=0.25",
        "variants": results,
        "outputs": {"objects": str(object_table), "variants": str(variant_table)},
        "interpretation": [
            "Single-history objects are added only through target-date evidence already present in the model.",
            "Thresholds are diagnostic boundaries based on existing model gates, not fitted production parameters.",
            "Future persistence and manual totals evaluate tradeoffs but never select objects.",
        ],
        "limits": [
            "SCL and source flags are model-side evidence rather than field truth.",
            "Probability and confidence may inherit calibration error.",
            "No new DEM calculation or production modification is performed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_FOLDER: {output}")
    print("STATUS: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
