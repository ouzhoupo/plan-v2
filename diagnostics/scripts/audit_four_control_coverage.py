"""Check four previously reviewed controls against actual patched tile footprints."""
from pathlib import Path
from datetime import datetime
import json
import rasterio
from rasterio.features import geometry_window, geometry_mask
from shapely.geometry import shape, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'diagnostics/results'
SOURCE = BASE / 'positive_label_experiment/20260913_125712_794605'
DOWN = BASE / 'released_samples_downstream/20260913_143146_669833'
REVIEW = BASE / 'landuse_kept_control_review/20260913_143432_440490/review.json'

def main():
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    trace = json.loads((DOWN/'trace.json').read_text(encoding='utf-8'))
    rows = []
    for group in review['windows']:
        date, wid = group['date'], group['window_id']
        targets = {45219: {1427, 625}, 46608: {1723, 1114}}.get(wid, set())
        if date != '2025-05-21' or not targets:
            continue
        used = next(g['used_experiment_tiles'] for g in trace['groups']
                    if g['date'] == date and g['window_id'] == wid)
        footprints = []
        with rasterio.open(DOWN/f'{date}_{wid}_experiment_water.tif') as ref:
            for name in used:
                with rasterio.open(SOURCE/name) as tile:
                    assert tile.crs == ref.crs
                    footprints.append(box(*tile.bounds))
            coverage = unary_union(footprints).intersection(box(*ref.bounds))
        for item in group['samples']:
            if item['local_id'] not in targets:
                continue
            geom = shape(item['geometry'])
            row = dict(date=date, window_id=wid, local_id=item['local_id'],
                       area_m2=geom.area, patched_overlap_m2=geom.intersection(coverage).area,
                       patched_fraction=geom.intersection(coverage).area/geom.area,
                       distance_to_patch_m=geom.distance(coverage), used_tiles=used)
            for stage in ('water', 'candidate'):
                arrays = []
                for variant in ('baseline', 'experiment'):
                    with rasterio.open(DOWN/f'{date}_{wid}_{variant}_{stage}.tif') as src:
                        win = geometry_window(src, [geom.__geo_interface__])
                        arr = src.read(1, window=win) > 0
                        inside = geometry_mask([geom.__geo_interface__], out_shape=arr.shape,
                                               transform=src.window_transform(win), invert=True)
                        arrays.append(arr & inside)
                row[stage] = dict(baseline_pixels=int(arrays[0].sum()),
                                  experiment_pixels=int(arrays[1].sum()),
                                  changed_pixels=int((arrays[0] != arrays[1]).sum()))
            rows.append(row)
    assert len(rows) == 4
    out = BASE/'four_control_coverage'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(review=str(REVIEW), downstream=str(DOWN), objects=rows,
                  scope='Geometric patch coverage and saved raw-mask comparison only; no training.')
    (out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    print('RESULT:', out/'audit.json')

if __name__ == '__main__':
    main()
