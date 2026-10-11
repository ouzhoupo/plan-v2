#!/usr/bin/env python3
"""Describe which upstream objects drive the 2024/2025 area reversal.

This is a read-only attribution audit.  It labels the frozen paired C4 upstream
water masks with four-neighbour connectivity, joins probability, confidence and
source-path evidence, and measures pixel-support overlap with the other date.
It does not classify pond identity, change a threshold, or calculate volume.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = (
    ROOT
    / "diagnostics/results/combined_upstream_c0_c4"
    / "20260922_180153_112712"
)
RESULTS = ROOT / "diagnostics/results/two_date_upstream_object_drivers"
DATES = ("2024-06-01", "2025-05-21")
EXPECTED_AREA_KM2 = {"2024-06-01": 207.7158, "2025-05-21": 1102.0679}
SOURCE_NAMES = {
    0: "none_or_unassigned",
    1: "sar_only",
    2: "optical_only",
    3: "dual_source",
    4: "prior_only",
}


def area_band(area_m2: float) -> str:
    if area_m2 < 500:
        return "lt_500_m2"
    if area_m2 < 2_000:
        return "500_2000_m2"
    if area_m2 < 20_000:
        return "2000_20000_m2"
    if area_m2 < 200_000:
        return "20000_200000_m2"
    if area_m2 < 500_000:
        return "200000_500000_m2"
    return "ge_500000_m2"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_same_grid(reference, other) -> None:
    if (
        reference.width != other.width
        or reference.height != other.height
        or reference.crs != other.crs
        or not reference.transform.almost_equals(other.transform)
    ):
        raise RuntimeError(f"Grid mismatch: {other.name}")


def read_date(date: str) -> dict:
    folder = UPSTREAM / date
    paths = {
        "water": folder / "c4_water_mask.tif",
        "probability": folder / "c2_fused_probability.tif",
        "confidence": folder / "c2_confidence.tif",
        "source": folder / "c2_source_flag.tif",
    }
    with (
        rasterio.open(paths["water"]) as water_src,
        rasterio.open(paths["probability"]) as probability_src,
        rasterio.open(paths["confidence"]) as confidence_src,
        rasterio.open(paths["source"]) as source_src,
    ):
        for other in (probability_src, confidence_src, source_src):
            assert_same_grid(water_src, other)
        water = water_src.read(1) > 0
        probability = probability_src.read(1).astype("float32")
        confidence = confidence_src.read(1).astype("float32")
        # Historical saved confidence rasters may use either 0..1 floats or
        # 0..100 percentage encoding.  Normalize only the latter.
        confidence_inside = confidence[water & np.isfinite(confidence)]
        confidence_encoding = "unit_interval"
        if confidence_inside.size and float(np.max(confidence_inside)) > 1.5:
            confidence /= 100.0
            confidence_encoding = "integer_percentage_divided_by_100"
        source = source_src.read(1).astype("uint8")
        transform = water_src.transform
        crs = str(water_src.crs)
        pixel_area_m2 = abs(float(transform.a * transform.e))
    return {
        "water": water,
        "probability": probability,
        "confidence": confidence,
        "source": source,
        "transform": transform,
        "crs": crs,
        "pixel_area_m2": pixel_area_m2,
        "confidence_encoding": confidence_encoding,
    }


def object_rows(date: str, data: dict, other_water: np.ndarray) -> tuple[list[dict], dict]:
    water = data["water"]
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8")
    labels, count = ndimage.label(water, structure=structure)
    slices = ndimage.find_objects(labels)
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    overlap_counts = np.bincount(
        labels[other_water].ravel(), minlength=count + 1
    )

    valid_probability = np.isfinite(data["probability"]) & water
    valid_confidence = np.isfinite(data["confidence"]) & water
    probability_counts = np.bincount(
        labels[valid_probability].ravel(), minlength=count + 1
    )
    confidence_counts = np.bincount(
        labels[valid_confidence].ravel(), minlength=count + 1
    )
    probability_sums = np.bincount(
        labels[valid_probability].ravel(),
        weights=data["probability"][valid_probability],
        minlength=count + 1,
    )
    confidence_sums = np.bincount(
        labels[valid_confidence].ravel(),
        weights=data["confidence"][valid_confidence],
        minlength=count + 1,
    )
    source_counts = {
        code: np.bincount(
            labels[water & (data["source"] == code)].ravel(),
            minlength=count + 1,
        )
        for code in SOURCE_NAMES
    }

    eroded = ndimage.binary_erosion(water, structure=structure, border_value=0)
    boundary = water & ~eroded
    boundary_counts = np.bincount(labels[boundary].ravel(), minlength=count + 1)
    transform = data["transform"]
    pixel_area = data["pixel_area_m2"]
    pixel_size = math.sqrt(pixel_area)

    rows: list[dict] = []
    for object_id in range(1, count + 1):
        pixels = int(counts[object_id])
        if pixels == 0:
            continue
        window = slices[object_id - 1]
        height = window[0].stop - window[0].start
        width = window[1].stop - window[1].start
        center_row = (window[0].start + window[0].stop - 1) / 2.0
        center_col = (window[1].start + window[1].stop - 1) / 2.0
        center_x, center_y = transform * (center_col + 0.5, center_row + 0.5)
        area_m2 = pixels * pixel_area
        perimeter_m = float(boundary_counts[object_id]) * pixel_size
        compactness = (
            4.0 * math.pi * area_m2 / (perimeter_m * perimeter_m)
            if perimeter_m > 0
            else None
        )
        fractions = {
            code: float(source_counts[code][object_id] / pixels)
            for code in SOURCE_NAMES
        }
        dominant_code = max(fractions, key=fractions.get)
        other_overlap_pixels = int(overlap_counts[object_id])
        row = {
            "date": date,
            "object_id": object_id,
            "pixels": pixels,
            "area_m2": area_m2,
            "area_band": area_band(area_m2),
            "bbox_height_pixels": height,
            "bbox_width_pixels": width,
            "bbox_fill_fraction": pixels / (height * width),
            "bbox_elongation": max(height, width) / min(height, width),
            "boundary_pixels": int(boundary_counts[object_id]),
            "pixel_compactness_proxy": compactness,
            "center_x": center_x,
            "center_y": center_y,
            "other_date_overlap_pixels": other_overlap_pixels,
            "other_date_overlap_fraction": other_overlap_pixels / pixels,
            "exclusive_to_this_date_pixels": pixels - other_overlap_pixels,
            "exclusive_to_this_date_area_m2": (pixels - other_overlap_pixels) * pixel_area,
            "probability_valid_fraction": probability_counts[object_id] / pixels,
            "probability_mean": (
                probability_sums[object_id] / probability_counts[object_id]
                if probability_counts[object_id]
                else None
            ),
            "confidence_valid_fraction": confidence_counts[object_id] / pixels,
            "confidence_mean": (
                confidence_sums[object_id] / confidence_counts[object_id]
                if confidence_counts[object_id]
                else None
            ),
            "dominant_source_code": dominant_code,
            "dominant_source_name": SOURCE_NAMES[dominant_code],
        }
        for code, name in SOURCE_NAMES.items():
            row[f"source_{code}_{name}_fraction"] = fractions[code]
        rows.append(row)

    total_pixels = int(water.sum())
    validation = {
        "objects": count,
        "water_pixels": total_pixels,
        "object_pixel_sum": sum(int(row["pixels"]) for row in rows),
        "area_km2": total_pixels * pixel_area / 1e6,
        "other_date_overlap_pixels": int(np.count_nonzero(water & other_water)),
        "exclusive_pixels": int(np.count_nonzero(water & ~other_water)),
        "confidence_encoding": data["confidence_encoding"],
    }
    del labels, eroded, boundary
    return rows, validation


def grouped_rows(all_rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    groupings = (
        ("area_band",),
        ("dominant_source_name",),
        ("area_band", "dominant_source_name"),
    )
    for date in DATES:
        date_rows = [row for row in all_rows if row["date"] == date]
        for fields in groupings:
            keys = sorted({tuple(row[field] for field in fields) for row in date_rows})
            for key in keys:
                rows = [
                    row for row in date_rows
                    if tuple(row[field] for field in fields) == key
                ]
                total_area = sum(float(row["area_m2"]) for row in rows)
                exclusive_area = sum(
                    float(row["exclusive_to_this_date_area_m2"]) for row in rows
                )
                result.append({
                    "date": date,
                    "grouping": "+".join(fields),
                    "group": "|".join(str(value) for value in key),
                    "objects": len(rows),
                    "area_km2": total_area / 1e6,
                    "exclusive_to_this_date_area_km2": exclusive_area / 1e6,
                    "exclusive_fraction_of_group_area": exclusive_area / total_area,
                })
    return result


def main() -> int:
    data = {date: read_date(date) for date in DATES}
    if data[DATES[0]]["crs"] != data[DATES[1]]["crs"]:
        raise RuntimeError("Cross-date CRS mismatch")
    if data[DATES[0]]["water"].shape != data[DATES[1]]["water"].shape:
        raise RuntimeError("Cross-date shape mismatch")
    if not data[DATES[0]]["transform"].almost_equals(data[DATES[1]]["transform"]):
        raise RuntimeError("Cross-date transform mismatch")

    rows: list[dict] = []
    validations: dict[str, dict] = {}
    for date, other_date in ((DATES[0], DATES[1]), (DATES[1], DATES[0])):
        date_rows, validation = object_rows(date, data[date], data[other_date]["water"])
        rows.extend(date_rows)
        validations[date] = validation

    for date in DATES:
        observed = validations[date]["area_km2"]
        expected = EXPECTED_AREA_KM2[date]
        validations[date]["matches_stage_area"] = math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=0.00005
        )
        validations[date]["object_pixels_close"] = (
            validations[date]["water_pixels"] == validations[date]["object_pixel_sum"]
        )

    overlap_24 = validations[DATES[0]]["other_date_overlap_pixels"]
    overlap_25 = validations[DATES[1]]["other_date_overlap_pixels"]
    cross_date_overlap_symmetric = overlap_24 == overlap_25
    status = "complete" if (
        cross_date_overlap_symmetric
        and all(v["matches_stage_area"] and v["object_pixels_close"] for v in validations.values())
    ) else "validation_failed"

    out = RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    for date in DATES:
        date_rows = [row for row in rows if row["date"] == date]
        write_csv(out / f"objects_{date}.csv", date_rows)
        top_area = sorted(date_rows, key=lambda row: float(row["area_m2"]), reverse=True)[:100]
        top_exclusive = sorted(
            date_rows,
            key=lambda row: float(row["exclusive_to_this_date_area_m2"]),
            reverse=True,
        )[:100]
        write_csv(out / f"top100_area_{date}.csv", top_area)
        write_csv(out / f"top100_exclusive_area_{date}.csv", top_exclusive)
    write_csv(out / "grouped_object_area.csv", grouped_rows(rows))

    summary = {
        "status": status,
        "experiment": "Two-date upstream object-driver audit",
        "stage": "paired C4 upstream water before vector deduction and downstream gates",
        "dates": validations,
        "cross_date_overlap_symmetric": cross_date_overlap_symmetric,
        "area_ratio_2025_to_2024": (
            validations[DATES[1]]["area_km2"] / validations[DATES[0]]["area_km2"]
        ),
        "source_code_meaning": SOURCE_NAMES,
        "outputs": {
            "all_object_tables": [str(out / f"objects_{date}.csv") for date in DATES],
            "grouped_table": str(out / "grouped_object_area.csv"),
            "top_area_tables": [str(out / f"top100_area_{date}.csv") for date in DATES],
            "top_exclusive_area_tables": [
                str(out / f"top100_exclusive_area_{date}.csv") for date in DATES
            ],
        },
        "limits": [
            "A connected component is a raster object, not a confirmed physical pond.",
            "Cross-date pixel overlap is common spatial support, not proof of physical identity or unchanged water.",
            "Exclusive area can contain real hydrological change, observation differences, segmentation effects, or false classification.",
            "Source code and model probability describe the evidence path; neither is an independent truth label.",
            "Area bands and shape metrics are descriptive bins, not proposed screening thresholds.",
            "No model threshold, mask, object output, DEM volume, or production result is modified.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {out}", flush=True)
    print(f"STATUS: {status}", flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
