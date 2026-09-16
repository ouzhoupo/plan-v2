"""User-run, bounded area-gate check. No land-use or DEM processing."""
import sys
sys.dont_write_bytecode = True
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import pond_storage_monitor as model

SOURCE = ROOT / 'diagnostics/results/fixed_site_closing/20260914_170521'
SITES = ROOT / 'diagnostics/results/deduction_fixed_sites/20260914_164745/review.json'
DATE = '2025-05-21'


def main():
    cfg = yaml.safe_load((ROOT / f'output/20250521/final/pihe/resolved_monitor_cfg_{DATE}.yaml').read_text(encoding='utf-8'))
    if cfg.get('watershed_split_enable') or cfg.get('simplify_tolerance_m', 0):
        raise RuntimeError('This bounded check requires watershed and simplification disabled.')
    minimum, maximum = float(cfg['min_water_area_m2']), float(cfg['max_water_area_m2'])
    old_sites = {s['site']: s for s in json.loads(SITES.read_text(encoding='utf-8'))['sites']}
    out = ROOT / 'diagnostics/results/closing_area_gate' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'minimum_m2': minimum, 'maximum_m2': maximum,
              'scope': 'Two fixed sites, 2025-05-21 only. Area gate only. Boundary components unresolved; no final pond counts or volumes.', 'sites': []}
    rows = []
    for site in ['small_reservoir', 'main_canal']:
        arrays = []
        paths = [SOURCE / f'{site}_{DATE}_close{k}.tif' for k in (0, 1)]
        for k, path in enumerate(paths):
            with rasterio.open(path) as src:
                arrays.append(src.read(1) > 0)
                if k == 0:
                    trans, crs, bounds = src.transform, src.crs, src.bounds
                elif not src.transform.almost_equals(trans) or src.crs != crs or arrays[0].shape != arrays[1].shape:
                    raise RuntimeError('Local input grid mismatch')
        with rasterio.open(ROOT / f'output/20250521/final/pihe/pond_water_mask_{DATE}.tif') as saved:
            win = from_bounds(*bounds, saved.transform).round_offsets().round_lengths()
            if not saved.window_transform(win).almost_equals(trans):
                raise RuntimeError('Historical grid mismatch')
            historical = saved.read(1, window=win) > 0
        if historical.shape != arrays[1].shape or not np.array_equal(historical, arrays[1]):
            raise RuntimeError('Baseline differs from historical candidate; stop interpretation')
        cw = from_bounds(*old_sites[site]['dates'][DATE]['bounds'], trans).round_offsets().round_lengths()
        core = np.zeros(arrays[0].shape, bool)
        y, x, h, w = map(int, [cw.row_off, cw.col_off, cw.height, cw.width])
        if min(x, y) < 0 or x+w > core.shape[1] or y+h > core.shape[0]:
            raise RuntimeError('Core window outside patch')
        core[y:y+h, x:x+w] = True
        changed = (arrays[0] != arrays[1]) & core
        added = arrays[1] & ~arrays[0] & core
        removed = arrays[0] & ~arrays[1] & core
        pixel_area = abs(trans.a * trans.e - trans.b * trans.d)
        variants = []
        for k, (path, mask) in enumerate(zip(paths, arrays)):
            labels, _ = ndimage.label(mask)  # Four-neighbour, matching polygonization.
            counts = np.bincount(labels.ravel())
            areas = counts * pixel_area
            edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            boundary = np.zeros(len(counts), bool)
            boundary[edge_ids] = True
            passing = (areas >= minimum) & (areas <= maximum)
            passing[0] = False
            # Verify the area rule against the actual production function on the same patch.
            objects = model.polygonize_water_mask(str(path), min_area_m2=minimum, max_area_m2=maximum,
                                                 simplify_tolerance=0, watershed_split_enable=False)
            actual = model.rasterize_geoms_for_window([o.geom for o in objects], mask.shape, trans,
                                                     all_touched=False) > 0
            mismatch = int(np.count_nonzero(actual != passing[labels]))
            if mismatch:
                raise RuntimeError(f'Production area-gate mismatch: {site}, close{k}, {mismatch}')
            affected = np.unique(labels[changed & mask])
            for lid in affected:
                lid = int(lid)
                state = 'boundary_unresolved' if boundary[lid] else ('pass' if passing[lid] else ('too_small' if areas[lid] < minimum else 'too_large'))
                rows.append({'site': site, 'closing': k, 'local_component_id': lid,
                             'patch_component_area_m2': float(areas[lid]), 'status': state,
                             'changed_core_pixels_in_component': int(np.count_nonzero(changed & (labels == lid)))})
            target = removed if k == 0 else added
            variants.append({'closing': k, 'production_area_gate_mismatch_pixels': mismatch,
                             'changed_pixels_of_interest': 'removed pixels before closing' if k == 0 else 'added pixels after closing',
                             'target_pixels': int(target.sum()),
                             'target_pixels_in_complete_passing_components': int(np.count_nonzero(target & passing[labels] & ~boundary[labels])),
                             'target_pixels_in_complete_rejected_components': int(np.count_nonzero(target & ~passing[labels] & ~boundary[labels])),
                             'target_pixels_boundary_unresolved': int(np.count_nonzero(target & boundary[labels])),
                             'affected_complete_components': sum(r['site'] == site and r['closing'] == k and r['status'] != 'boundary_unresolved' for r in rows)})
        report['sites'].append({'site': site, 'baseline_mismatch_pixels': 0, 'variants': variants})
        print(site, 'complete', flush=True)
    with (out / 'affected_components.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['site', 'closing', 'local_component_id', 'patch_component_area_m2', 'status', 'changed_core_pixels_in_component']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report['status'] = 'complete'
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('RESULT_FOLDER:', out, flush=True)


if __name__ == '__main__':
    main()
