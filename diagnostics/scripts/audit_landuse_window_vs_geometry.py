"""Quantify current land-use window hits versus true object-intersection hits."""
from datetime import datetime
from pathlib import Path
import json
import sys

import rasterio
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import pond_storage_monitor as monitor  # noqa: E402
from sample_landuse_kept_controls import landuse_counts  # noqa: E402


SOURCE = (
    ROOT
    / "diagnostics/results/released_samples_downstream"
    / "20260913_143146_669833"
)
LANDUSE = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
OUT = (
    ROOT
    / "diagnostics/results/landuse_window_geometry_audit"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
WINDOWS = [
    ("2024-06-01", 14660),
    ("2025-05-21", 45219),
    ("2025-05-21", 46608),
]


def main():
    report = {
        "scope": (
            "All area-qualified experiment candidates in three local windows. Current "
            "filter uses any value=1 in a padded bounding window; strict count uses "
            "all_touched pixels within the true object geometry. Local audit only."
        ),
        "windows": [],
    }
    with rasterio.open(LANDUSE) as lu:
        for date, window_id in WINDOWS:
            candidate = SOURCE / f"{date}_{window_id}_experiment_candidate.tif"
            cfg = yaml.safe_load(
                (
                    ROOT
                    / f"output/{date.replace('-', '')}/final/pihe/resolved_monitor_cfg_{date}.yaml"
                ).read_text(encoding="utf-8")
            )
            with rasterio.open(candidate) as src:
                crs = src.crs
            objects = monitor.polygonize_water_mask(
                str(candidate),
                min_area_m2=float(cfg["min_water_area_m2"]),
                max_area_m2=float(cfg["max_water_area_m2"]),
                simplify_tolerance=float(cfg.get("simplify_tolerance_m", 0)),
                watershed_split_enable=False,
            )
            rows = []
            for obj in objects:
                window_hits, object_hits = landuse_counts(obj.geom, crs, lu)
                if window_hits == 0:
                    continue
                rows.append(
                    {
                        "local_id": obj.obj_id,
                        "area_m2": obj.area_m2,
                        "window_value1_pixels": window_hits,
                        "object_all_touched_value1_pixels": object_hits,
                        "external_only": object_hits == 0,
                    }
                )
            external = [row for row in rows if row["external_only"]]
            strict = [row for row in rows if not row["external_only"]]
            report["windows"].append(
                {
                    "date": date,
                    "window_id": window_id,
                    "area_qualified_objects": len(objects),
                    "current_window_rule_kept_objects": len(rows),
                    "current_window_rule_kept_area_m2": float(
                        sum(row["area_m2"] for row in rows)
                    ),
                    "strict_geometry_hit_objects": len(strict),
                    "strict_geometry_hit_area_m2": float(
                        sum(row["area_m2"] for row in strict)
                    ),
                    "external_only_objects": len(external),
                    "external_only_area_m2": float(
                        sum(row["area_m2"] for row in external)
                    ),
                    "external_only_fraction_of_current_count": (
                        len(external) / len(rows) if rows else None
                    ),
                    "external_only_fraction_of_current_area": (
                        sum(row["area_m2"] for row in external)
                        / sum(row["area_m2"] for row in rows)
                        if rows else None
                    ),
                    "external_only_candidates": external,
                }
            )
            print(date, window_id, "complete", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT:", path)


if __name__ == "__main__":
    main()
