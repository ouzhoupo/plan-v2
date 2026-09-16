#!/usr/bin/env python3
"""Build small GIS-ready samples for Pihe stage-error diagnosis.

This script reads existing model outputs. It does not rerun or modify the model.
It creates diagnostic rasters, sampled polygons, and CSV summaries under
``diagnostics/results/pihe_gis_samples`` by default.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fiona
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask, geometry_window, shapes
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


DEFAULT_DATES = ("2024-06-01", "2025-05-21")


def date_key(date: str) -> str:
    return date.replace("-", "")


def paths_for(root: Path, date: str) -> dict[str, Path]:
    day = root / "output" / date_key(date)
    return {
        "water_prob": day / "pihe" / f"water_prob_{date}.tif",
        "water_mask": day / "pihe" / f"water_mask_{date}.tif",
        "source_flag": day / "pihe" / f"source_flag_{date}.tif",
        "confidence": day / "pihe" / f"confidence_{date}.tif",
        "candidate": day / "final" / "pihe" / f"pond_water_mask_{date}.tif",
        "final_gpkg": day / "final" / "pihe" / f"pond_storage_{date}.gpkg",
    }


def require_files(items: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in items.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.crs == b.crs
        and a.transform.almost_equals(b.transform)
        and a.width == b.width
        and a.height == b.height
    )


def transformed_geom(geom: Any, source_crs: Any, target_crs: Any) -> Any:
    src = CRS.from_user_input(source_crs)
    dst = CRS.from_user_input(target_crs)
    if src == dst:
        return geom
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def clipped_window(dataset: rasterio.io.DatasetReader, geom: Any):
    try:
        win = geometry_window(dataset, [mapping(geom)], pad_x=1, pad_y=1)
    except Exception:
        return None
    full = rasterio.windows.Window(0, 0, dataset.width, dataset.height)
    try:
        return win.intersection(full).round_offsets().round_lengths()
    except Exception:
        return None


def pixels_inside(dataset: rasterio.io.DatasetReader, geom: Any):
    win = clipped_window(dataset, geom)
    if win is None or win.width <= 0 or win.height <= 0:
        return None, None, None
    arr = dataset.read(1, window=win, masked=True)
    inside = geometry_mask(
        [mapping(geom)],
        out_shape=(int(win.height), int(win.width)),
        transform=dataset.window_transform(win),
        invert=True,
        all_touched=True,
    )
    return arr, inside, win


def raster_stats(dataset: rasterio.io.DatasetReader, geom: Any) -> dict[str, float | None]:
    arr, inside, _ = pixels_inside(dataset, geom)
    if arr is None:
        return {"mean": None, "positive_fraction": None}
    valid = inside & ~np.ma.getmaskarray(arr)
    values = np.asarray(arr)[valid]
    if values.size == 0:
        return {"mean": None, "positive_fraction": None}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "positive_fraction": None}
    return {
        "mean": float(finite.mean()),
        "positive_fraction": float(np.count_nonzero(finite > 0) / finite.size),
    }


def binary_fraction(dataset: rasterio.io.DatasetReader, geom: Any) -> float | None:
    """Fraction of geometry-covered pixels above zero, retaining background zeros."""
    win = clipped_window(dataset, geom)
    if win is None or win.width <= 0 or win.height <= 0:
        return None
    values = dataset.read(1, window=win, masked=False)
    inside = geometry_mask(
        [mapping(geom)],
        out_shape=(int(win.height), int(win.width)),
        transform=dataset.window_transform(win),
        invert=True,
        all_touched=True,
    )
    denominator = int(np.count_nonzero(inside))
    if denominator == 0:
        return None
    return float(np.count_nonzero((values > 0) & inside) / denominator)


def source_fractions(dataset: rasterio.io.DatasetReader, geom: Any) -> dict[str, float | None]:
    win = clipped_window(dataset, geom)
    if win is None or win.width <= 0 or win.height <= 0:
        return {f"source_{i}_fraction": None for i in range(5)}
    arr = dataset.read(1, window=win, masked=False)
    inside = geometry_mask(
        [mapping(geom)],
        out_shape=(int(win.height), int(win.width)),
        transform=dataset.window_transform(win),
        invert=True,
        all_touched=True,
    )
    values = arr[inside]
    if values.size == 0:
        return {f"source_{i}_fraction": None for i in range(5)}
    return {f"source_{i}_fraction": float(np.count_nonzero(values == i) / values.size) for i in range(5)}


def landuse_evidence(dataset: rasterio.io.DatasetReader, geom: Any) -> dict[str, Any]:
    """Return both the current bbox-any rule and a true polygon-pixel overlap."""
    win = clipped_window(dataset, geom)
    if win is None:
        return {"landuse_bbox_hit": False, "landuse_exact_hit": False, "landuse_exact_fraction": 0.0}
    arr = dataset.read(1, window=win, masked=False)
    water = arr == 1
    bbox_hit = bool(np.any(water))
    inside = geometry_mask(
        [mapping(geom)],
        out_shape=(int(win.height), int(win.width)),
        transform=dataset.window_transform(win),
        invert=True,
        all_touched=True,
    )
    denominator = int(np.count_nonzero(inside))
    overlap = int(np.count_nonzero(water & inside))
    return {
        "landuse_bbox_hit": bbox_hit,
        "landuse_exact_hit": overlap > 0,
        "landuse_exact_fraction": overlap / denominator if denominator else 0.0,
    }


@dataclass
class TopSamples:
    limit: int
    heap: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)
    serial: int = 0

    def add(self, score: float, record: dict[str, Any]) -> None:
        self.serial += 1
        item = (float(score), self.serial, record)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        elif item[0] > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def records(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heap, reverse=True)]


def read_final_ids(gpkg: Path) -> set[int]:
    layers = fiona.listlayers(gpkg)
    if not layers:
        raise RuntimeError(f"No vector layer in {gpkg}")
    ids: set[int] = set()
    with fiona.open(gpkg, layer=layers[0]) as src:
        for feature in src:
            value = feature["properties"].get("pond_id")
            if value is not None:
                ids.add(int(value))
    return ids


def scan_final_landuse(
    gpkg: Path,
    target_crs: Any,
    landuse: rasterio.io.DatasetReader,
    sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Find final objects retained only because water exists in their padded bbox."""
    sampler = TopSamples(sample_count)
    counts = {"final_total": 0, "final_exact_landuse_hit": 0, "final_bbox_only": 0, "final_no_bbox_hit": 0}
    layer = fiona.listlayers(gpkg)[0]
    with fiona.open(gpkg, layer=layer) as src:
        source_crs = src.crs_wkt or src.crs
        for feature in src:
            if feature["geometry"] is None:
                continue
            counts["final_total"] += 1
            geom = shape(feature["geometry"])
            landuse_geom = transformed_geom(geom, source_crs, landuse.crs)
            evidence = landuse_evidence(landuse, landuse_geom)
            if evidence["landuse_exact_hit"]:
                counts["final_exact_landuse_hit"] += 1
                continue
            if evidence["landuse_bbox_hit"]:
                counts["final_bbox_only"] += 1
                output_geom = transformed_geom(geom, source_crs, target_crs)
                sampler.add(
                    float(output_geom.area),
                    {
                        "geometry": output_geom,
                        "area_m2": float(output_geom.area),
                        "pond_id": int(feature["properties"].get("pond_id")),
                        "category": "final_bbox_only",
                        "crs_wkt": CRS.from_user_input(target_crs).to_wkt(),
                    },
                )
            else:
                counts["final_no_bbox_hit"] += 1
    return sampler.records(), counts


def polygon_samples(
    candidate_path: Path,
    final_ids: set[int],
    min_area: float,
    max_area: float,
    sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samplers = {
        "too_small_near_gate": TopSamples(sample_count),
        "too_large_near_gate": TopSamples(sample_count),
        "landuse_rejected_large": TopSamples(sample_count),
        "final_kept_large": TopSamples(sample_count),
    }
    counts = {key: 0 for key in samplers}
    area_gate_id = 0
    total_components = 0
    with rasterio.open(candidate_path) as src:
        crs_wkt = src.crs.to_wkt()
        for geom_mapping, value in shapes(rasterio.band(src, 1), transform=src.transform, connectivity=4):
            if int(value) <= 0:
                continue
            total_components += 1
            geom = shape(geom_mapping)
            area = float(geom.area)
            base = {"geometry": geom, "area_m2": area, "pond_id": None, "crs_wkt": crs_wkt}
            if area < min_area:
                category = "too_small_near_gate"
                counts[category] += 1
                samplers[category].add(area, {**base, "category": category})
                continue
            if area > max_area:
                category = "too_large_near_gate"
                counts[category] += 1
                samplers[category].add(-abs(area - max_area), {**base, "category": category})
                continue
            area_gate_id += 1
            category = "final_kept_large" if area_gate_id in final_ids else "landuse_rejected_large"
            counts[category] += 1
            samplers[category].add(area, {**base, "category": category, "pond_id": area_gate_id})
    records = [record for sampler in samplers.values() for record in sampler.records()]
    return records, {
        "candidate_components": total_components,
        "area_gate_components": area_gate_id,
        "category_counts": counts,
    }


def make_difference_raster(reference_path: Path, other_path: Path, output_path: Path) -> dict[str, Any]:
    """Write 1=reference only, 2=both, 3=other only on the reference grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"reference_only_pixels": 0, "both_pixels": 0, "other_only_pixels": 0}
    with rasterio.open(reference_path) as ref, rasterio.open(other_path) as other:
        profile = ref.profile.copy()
        profile.update(dtype="uint8", nodata=0, count=1, compress="deflate", BIGTIFF="IF_SAFER")
        vrt_options = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "resampling": Resampling.nearest,
            "nodata": 0,
        }
        other_view = other if same_grid(ref, other) else WarpedVRT(other, **vrt_options)
        try:
            with rasterio.open(output_path, "w", **profile) as dst:
                for _, win in ref.block_windows(1):
                    a = ref.read(1, window=win) > 0
                    b = other_view.read(1, window=win) > 0
                    out = np.zeros(a.shape, dtype="uint8")
                    out[a & ~b] = 1
                    out[a & b] = 2
                    out[~a & b] = 3
                    counts["reference_only_pixels"] += int(np.count_nonzero(out == 1))
                    counts["both_pixels"] += int(np.count_nonzero(out == 2))
                    counts["other_only_pixels"] += int(np.count_nonzero(out == 3))
                    dst.write(out, 1, window=win)
        finally:
            if other_view is not other:
                other_view.close()
        pixel_area = abs(ref.transform.a * ref.transform.e)
        return {**counts, "pixel_area_m2": pixel_area}


def make_removed_raster(water_path: Path, candidate_path: Path, output_path: Path) -> dict[str, Any]:
    """Write pixels present in upstream water mask but absent from candidate mask."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_pixels = 0
    with rasterio.open(water_path) as water, rasterio.open(candidate_path) as candidate:
        if not same_grid(water, candidate):
            raise RuntimeError(f"Grid mismatch: {water_path} vs {candidate_path}")
        profile = water.profile.copy()
        profile.update(dtype="uint8", nodata=0, count=1, compress="deflate", BIGTIFF="IF_SAFER")
        with rasterio.open(output_path, "w", **profile) as dst:
            for _, win in water.block_windows(1):
                a = water.read(1, window=win) > 0
                b = candidate.read(1, window=win) > 0
                out = (a & ~b).astype("uint8")
                removed_pixels += int(out.sum())
                dst.write(out, 1, window=win)
        return {"removed_pixels": removed_pixels, "removed_area_m2": removed_pixels * abs(water.transform.a * water.transform.e)}


def largest_mask_polygons(mask_path: Path, category: str, sample_count: int) -> list[dict[str, Any]]:
    sampler = TopSamples(sample_count)
    with rasterio.open(mask_path) as src:
        crs_wkt = src.crs.to_wkt()
        for geom_mapping, value in shapes(rasterio.band(src, 1), transform=src.transform, connectivity=4):
            if int(value) <= 0:
                continue
            geom = shape(geom_mapping)
            area = float(geom.area)
            sampler.add(area, {"geometry": geom, "area_m2": area, "pond_id": None, "category": category, "crs_wkt": crs_wkt})
    return sampler.records()


def enrich_records(
    records: Iterable[dict[str, Any]],
    rasters: dict[str, rasterio.io.DatasetReader],
    landuse: rasterio.io.DatasetReader,
    other_water: rasterio.io.DatasetReader,
) -> list[dict[str, Any]]:
    enriched = []
    rank_by_category: dict[str, int] = {}
    for record in records:
        category = record["category"]
        rank_by_category[category] = rank_by_category.get(category, 0) + 1
        source_crs = record.pop("crs_wkt")
        geom = record["geometry"]
        prob_geom = transformed_geom(geom, source_crs, rasters["water_prob"].crs)
        landuse_geom = transformed_geom(geom, source_crs, landuse.crs)
        other_geom = transformed_geom(geom, source_crs, other_water.crs)
        prob = raster_stats(rasters["water_prob"], prob_geom)
        confidence = raster_stats(rasters["confidence"], prob_geom)
        other_fraction = binary_fraction(other_water, other_geom)
        confidence_mean = confidence["mean"]
        if confidence_mean is not None and confidence_mean > 1.0:
            confidence_mean /= 100.0
        record.update(
            {
                "sample_rank": rank_by_category[category],
                "water_prob_mean": prob["mean"],
                "confidence_mean": confidence_mean,
                "other_date_water_fraction": other_fraction,
                **source_fractions(rasters["source_flag"], prob_geom),
                **landuse_evidence(landuse, landuse_geom),
            }
        )
        enriched.append(record)
    return enriched


def write_gpkg(path: Path, records: list[dict[str, Any]], crs_wkt: str) -> None:
    if path.exists():
        path.unlink()
    schema = {
        "geometry": "Unknown",
        "properties": {
            "date": "str:10",
            "category": "str:40",
            "rank": "int",
            "pond_id": "int",
            "area_m2": "float",
            "prob_mean": "float",
            "conf_mean": "float",
            "other_wtr": "float",
            "src0": "float",
            "src1": "float",
            "src2": "float",
            "src3": "float",
            "src4": "float",
            "lu_bbox": "int",
            "lu_exact": "int",
            "lu_frac": "float",
        },
    }
    with fiona.open(path, "w", driver="GPKG", layer="samples", schema=schema, crs_wkt=crs_wkt) as dst:
        for record in records:
            dst.write(
                {
                    "geometry": mapping(record["geometry"]),
                    "properties": {
                        "date": record["date"],
                        "category": record["category"],
                        "rank": record["sample_rank"],
                        "pond_id": record.get("pond_id"),
                        "area_m2": record["area_m2"],
                        "prob_mean": record.get("water_prob_mean"),
                        "conf_mean": record.get("confidence_mean"),
                        "other_wtr": record.get("other_date_water_fraction"),
                        "src0": record.get("source_0_fraction"),
                        "src1": record.get("source_1_fraction"),
                        "src2": record.get("source_2_fraction"),
                        "src3": record.get("source_3_fraction"),
                        "src4": record.get("source_4_fraction"),
                        "lu_bbox": int(bool(record.get("landuse_bbox_hit"))),
                        "lu_exact": int(bool(record.get("landuse_exact_hit"))),
                        "lu_frac": record.get("landuse_exact_fraction"),
                    },
                }
            )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "date", "category", "sample_rank", "pond_id", "area_m2", "water_prob_mean",
        "confidence_mean", "other_date_water_fraction", "source_0_fraction", "source_1_fraction",
        "source_2_fraction", "source_3_fraction", "source_4_fraction", "landuse_bbox_hit",
        "landuse_exact_hit", "landuse_exact_fraction", "centroid_x", "centroid_y",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            row["centroid_x"] = record["geometry"].centroid.x
            row["centroid_y"] = record["geometry"].centroid.y
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dates", nargs=2, default=DEFAULT_DATES, metavar=("EARLY", "LATE"))
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    dates = tuple(args.dates)
    out_dir = (args.out_dir or root / "diagnostics" / "results" / "pihe_gis_samples").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    landuse_path = root / "data" / "landuse_water" / "PSH_土地利用水体.tif"
    if not landuse_path.exists():
        raise FileNotFoundError(f"Missing land-use water raster: {landuse_path}")

    all_paths = {date: paths_for(root, date) for date in dates}
    for date in dates:
        require_files(all_paths[date])

    summary: dict[str, Any] = {"dates": dates, "sample_count_per_category": args.sample_count, "per_date": {}}
    change_path = out_dir / f"water_mask_change_{date_key(dates[0])}_{date_key(dates[1])}.tif"
    summary["water_mask_change"] = make_difference_raster(
        all_paths[dates[0]]["water_mask"], all_paths[dates[1]]["water_mask"], change_path
    )

    for index, date in enumerate(dates):
        other_date = dates[1 - index]
        items = all_paths[date]
        final_ids = read_final_ids(items["final_gpkg"])
        records, component_summary = polygon_samples(
            items["candidate"], final_ids, min_area=500.0, max_area=500000.0, sample_count=args.sample_count
        )
        with rasterio.open(items["candidate"]) as candidate, rasterio.open(landuse_path) as landuse:
            bbox_only_records, final_landuse_summary = scan_final_landuse(
                items["final_gpkg"], candidate.crs, landuse, args.sample_count
            )
        records.extend(bbox_only_records)
        removed_path = out_dir / f"water_removed_before_candidate_{date_key(date)}.tif"
        removed_summary = make_removed_raster(items["water_mask"], items["candidate"], removed_path)
        records.extend(largest_mask_polygons(removed_path, "removed_before_candidate", args.sample_count))

        with rasterio.open(items["candidate"]) as candidate, rasterio.open(items["water_prob"]) as prob, rasterio.open(
            items["confidence"]
        ) as confidence, rasterio.open(items["source_flag"]) as source_flag, rasterio.open(
            landuse_path
        ) as landuse, rasterio.open(all_paths[other_date]["water_mask"]) as other_water:
            rasters = {"water_prob": prob, "confidence": confidence, "source_flag": source_flag}
            for record in records:
                record["date"] = date
            records = enrich_records(records, rasters, landuse, other_water)
            gpkg_path = out_dir / f"pihe_stage_samples_{date_key(date)}.gpkg"
            write_gpkg(gpkg_path, records, candidate.crs.to_wkt())
        csv_path = out_dir / f"pihe_stage_samples_{date_key(date)}.csv"
        write_csv(csv_path, records)
        summary["per_date"][date] = {
            **component_summary,
            **final_landuse_summary,
            **removed_summary,
            "final_ids_in_gpkg": len(final_ids),
            "sample_records": len(records),
            "gpkg": str(gpkg_path),
            "csv": str(csv_path),
            "removed_raster": str(removed_path),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
