#!/usr/bin/env python3
"""Audit temporal water evidence for 2024 objects rejected by land use.

The audit uses the paired 2024 shared-SAR/shared-alpha downstream result.  It
does not restore any object.  Previous-date evidence is potentially usable in
an operational prior; 2025 evidence is reported separately as retrospective
mechanism evidence and must not be treated as an available 2024 input.
"""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DATE = "2024-06-01"
SOURCE_RESULT = (
    ROOT / "diagnostics/results/new_sar_gate_dem_volume/20261010_145544_878831"
    / "paired_reference_downstream" / DATE
)
AREA_GATE = SOURCE_RESULT / "shared_reference_after_area_gate.tif"
FINAL = SOURCE_RESULT / "shared_reference_after_landuse.tif"
PREVIOUS_MASK = (
    ROOT / "diagnostics/results/shared_sar_calibration_upstream/20261009_193736_367365"
    / "2024-05-21/shared_sar_shared_alpha_c4_water_mask.tif"
)
FUTURE_MASK = (
    ROOT / "diagnostics/results/new_sar_evidence_gate_upstream/20261010_134344_780095"
    / "clear_low_no_aux_gate_c4_water_mask.tif"
)
S2_PATHS = {
    "previous_2024_05": ROOT / "data/s2/s2_stack_2024-05-21.tif",
    "target_2024_06": ROOT / "data/s2/s2_stack_2024-06-01.tif",
    "future_2025_05": ROOT / "data/s2/s2_stack_2025-05-21.tif",
}
OUTPUT_ROOT = ROOT / "diagnostics/results/landuse_temporal_evidence_2024"


def aligned_read(stack: ExitStack, path: Path, reference, band: int = 1) -> np.ndarray:
    source = stack.enter_context(rasterio.open(path))
    with WarpedVRT(
        source,
        crs=reference.crs,
        transform=reference.transform,
        width=reference.width,
        height=reference.height,
        resampling=Resampling.nearest,
    ) as vrt:
        return vrt.read(band)


def component_counts(labels: np.ndarray, evidence: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(labels[evidence], minlength=count + 1)[1:]


def aggregate(rows: list[dict], key: str) -> dict:
    selected = [row for row in rows if row[key]]
    return {
        "objects": len(selected),
        "area_km2": sum(row["area_m2"] for row in selected) / 1e6,
        "fraction_of_rejected_area": (
            sum(row["area_m2"] for row in selected)
            / sum(row["area_m2"] for row in rows)
            if rows else 0.0
        ),
    }


def main() -> int:
    for path in (AREA_GATE, FINAL, PREVIOUS_MASK, FUTURE_MASK, *S2_PATHS.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    with rasterio.open(AREA_GATE) as reference:
        area_gate = reference.read(1) > 0
        pixel_area = abs(float(reference.transform.a * reference.transform.e))
        grid = (reference.shape, reference.crs, reference.transform)
    labels, count = ndimage.label(
        area_gate,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8"),
    )
    slices = ndimage.find_objects(labels)
    pixels = np.bincount(labels.ravel(), minlength=count + 1)[1:]

    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(AREA_GATE))
        final = aligned_read(stack, FINAL, reference) > 0
        previous = aligned_read(stack, PREVIOUS_MASK, reference) > 0
        future = aligned_read(stack, FUTURE_MASK, reference) > 0
        scl = {
            name: aligned_read(stack, path, reference, band=7) == 6
            for name, path in S2_PATHS.items()
        }
    if (
        reference.shape != grid[0]
        or reference.crs != grid[1]
        or not reference.transform.almost_equals(grid[2])
    ):
        raise RuntimeError("Reference grid changed while reading")

    kept_pixels = component_counts(labels, final, count)
    rejected_ids = np.flatnonzero(kept_pixels == 0) + 1
    previous_near = ndimage.binary_dilation(previous, iterations=1)
    future_near = ndimage.binary_dilation(future, iterations=1)
    evidence_counts = {
        "previous_exact": component_counts(labels, previous, count),
        "previous_near": component_counts(labels, previous_near, count),
        "future_exact": component_counts(labels, future, count),
        "future_near": component_counts(labels, future_near, count),
        **{
            f"scl_{name}": component_counts(labels, mask, count)
            for name, mask in scl.items()
        },
    }

    rows: list[dict] = []
    for object_id in rejected_ids:
        index = int(object_id) - 1
        component_slice = slices[index]
        center_row = (component_slice[0].start + component_slice[0].stop - 1) / 2
        center_col = (component_slice[1].start + component_slice[1].stop - 1) / 2
        center_x, center_y = rasterio.transform.xy(grid[2], center_row, center_col)
        n = int(pixels[index])
        fractions = {key: int(values[index]) / n for key, values in evidence_counts.items()}
        previous_any = fractions["previous_near"] > 0
        future_any = fractions["future_near"] > 0
        target_scl_any = fractions["scl_target_2024_06"] > 0
        strong_previous = fractions["previous_near"] >= 0.25
        strong_future = fractions["future_near"] >= 0.25
        strong_target_scl = fractions["scl_target_2024_06"] >= 0.25
        evidence_sources = sum((previous_any, future_any, target_scl_any))
        strong_sources = sum((strong_previous, strong_future, strong_target_scl))
        rows.append({
            "object_id": int(object_id),
            "pixels": n,
            "area_m2": n * pixel_area,
            "center_x": float(center_x),
            "center_y": float(center_y),
            **fractions,
            "previous_any": previous_any,
            "future_any_retrospective": future_any,
            "target_scl_any": target_scl_any,
            "previous_fraction_ge_025": strong_previous,
            "future_fraction_ge_025_retrospective": strong_future,
            "target_scl_fraction_ge_025": strong_target_scl,
            "at_least_two_sources_any": evidence_sources >= 2,
            "at_least_two_sources_ge_025": strong_sources >= 2,
        })

    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    table = output / "rejected_objects_temporal_evidence.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checks = {
        "area_gate_objects": int(count),
        "kept_objects": int(np.count_nonzero(kept_pixels > 0)),
        "rejected_objects": len(rows),
        "area_gate_area_km2": float(area_gate.sum() * pixel_area / 1e6),
        "final_area_km2": float(final.sum() * pixel_area / 1e6),
        "rejected_object_area_km2": sum(row["area_m2"] for row in rows) / 1e6,
    }
    report = {
        "status": "complete",
        "experiment": "2024 land-use rejection temporal-evidence capacity audit",
        "date": DATE,
        "region": "pihe",
        "checks": checks,
        "evidence_capacity": {
            key: aggregate(rows, key)
            for key in (
                "previous_any",
                "future_any_retrospective",
                "target_scl_any",
                "previous_fraction_ge_025",
                "future_fraction_ge_025_retrospective",
                "target_scl_fraction_ge_025",
                "at_least_two_sources_any",
                "at_least_two_sources_ge_025",
            )
        },
        "output_table": str(table),
        "interpretation": [
            "Previous-date mask/SCL support is a candidate operational temporal prior, not truth.",
            "2025 support is retrospective diagnostic evidence and cannot be used as a real-time 2024 input.",
            "One-pixel dilation only tests 10 m shoreline/alignment tolerance and is not object restoration.",
            "The 0.25 fraction is a descriptive capacity bin, not a selected threshold.",
            "No rejected object is restored and no DEM volume is calculated.",
        ],
        "limits": [
            "All temporal masks remain remote-sensing model evidence and share sensor/systematic errors.",
            "SCL water is not independent field truth and may miss turbid, vegetated, shadowed, or cloud-covered water.",
            "Survey spatial-scope equivalence remains unresolved because subunit geometries are unavailable.",
            "Results apply only to the 2024-06 Pihe area-gate objects in this paired chain.",
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
