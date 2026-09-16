"""Audit land-use evidence and source experiment tiles for three downstream traces."""
from datetime import datetime
from pathlib import Path
import json

import rasterio
from rasterio.features import geometry_window
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform


ROOT = Path(__file__).resolve().parents[2]
TRACE = (
    ROOT
    / "diagnostics/results/counterexample_downstream"
    / "20260913_134900_764047/trace.json"
)
SOURCE = (
    ROOT
    / "diagnostics/results/positive_label_experiment"
    / "20260913_125712_794605"
)
LANDUSE = ROOT / "data/landuse_water/PSH_土地利用水体.tif"
OUT = (
    ROOT
    / "diagnostics/results/counterexample_landuse_audit"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)


def audit_geometry(geom, object_crs, lu):
    transformer = Transformer.from_crs(object_crs, lu.crs, always_xy=True)
    lu_geom = shapely_transform(transformer.transform, geom)
    win = geometry_window(lu, [lu_geom.__geo_interface__], pad_x=1, pad_y=1)
    data = lu.read(1, window=win)
    mask = lu.read_masks(1, window=win)
    return {
        "window_value1_pixels": int((data == 1).sum()),
        "window_nodata_pixels": int((mask == 0).sum()),
        "window_valid_pixels": int((mask > 0).sum()),
        "window_total_pixels": int(data.size),
    }


def intersecting_tiles(date, geom):
    names = []
    for path in sorted(SOURCE.glob(f"{date}_*_masks.tif")):
        with rasterio.open(path) as src:
            if box(*src.bounds).intersects(geom):
                names.append(path.name)
    return names


def main():
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    report = {
        "trace": str(TRACE),
        "experiment_source": str(SOURCE),
        "landuse": str(LANDUSE),
        "objects": [],
    }
    with rasterio.open(LANDUSE) as lu:
        for item in trace["objects"]:
            day = ROOT / "output" / item["date"].replace("-", "")
            candidate = day / f"final/pihe/pond_water_mask_{item['date']}.tif"
            with rasterio.open(candidate) as candidate_src:
                object_crs = candidate_src.crs
            row = {
                "date": item["date"],
                "pond_id": item["pond_id"],
                "baseline_candidate_mismatch_pixels": item[
                    "baseline_candidate_mismatch_pixels"
                ],
                "variants": {},
            }
            for variant in ("baseline", "experiment"):
                audited = []
                for obj in item["variants"][variant]:
                    geom = shape(obj["geometry"])
                    audited.append(
                        {**obj, **audit_geometry(geom, object_crs, lu)}
                    )
                row["variants"][variant] = audited
            sample_geom = shape(item["variants"]["experiment"][0]["geometry"])
            row["intersecting_experiment_tiles"] = intersecting_tiles(
                item["date"], sample_geom
            )
            report["objects"].append(row)
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
