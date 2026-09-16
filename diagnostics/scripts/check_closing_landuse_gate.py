"""User-run: land-use gate for the 26 previously traced complete objects only."""
import sys
sys.dont_write_bytecode = True
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.features import shapes, geometry_window
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import pond_storage_monitor as model
SOURCE = ROOT / 'diagnostics/results/fixed_site_closing/20260914_170521'
PREVIOUS = ROOT / 'diagnostics/results/closing_area_gate/20260914_173016_966503'
DATE = '2025-05-21'


def main():
    previous_summary = json.loads((PREVIOUS / 'summary.json').read_text(encoding='utf-8'))
    if previous_summary['status'] != 'complete':
        raise RuntimeError('Previous area check is incomplete')
    with (PREVIOUS / 'affected_components.csv').open(encoding='utf-8-sig', newline='') as f:
        targets = list(csv.DictReader(f))
    if any(r['status'] != 'pass' for r in targets):
        raise RuntimeError('Expected previously verified complete passing objects only')
    lu_path = ROOT / 'data/landuse_water/PSH_土地利用水体.tif'
    if not lu_path.is_file():
        raise FileNotFoundError(lu_path)
    records = []
    groups = sorted({(r['site'], int(r['closing'])) for r in targets})
    with rasterio.open(lu_path) as lu:
        for site, closing in groups:
            selected = [r for r in targets if r['site'] == site and int(r['closing']) == closing]
            with rasterio.open(SOURCE / f'{site}_{DATE}_close1.tif') as src:
                after = src.read(1) > 0
                trans, crs, bounds = src.transform, src.crs, src.bounds
            with rasterio.open(ROOT / f'output/20250521/final/pihe/pond_water_mask_{DATE}.tif') as src:
                win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
                if not src.window_transform(win).almost_equals(trans) or not np.array_equal(src.read(1, window=win)>0, after):
                    raise RuntimeError('Historical baseline mismatch')
            with rasterio.open(SOURCE / f'{site}_{DATE}_close{closing}.tif') as src:
                if src.crs != crs or not src.transform.almost_equals(trans):
                    raise RuntimeError('Variant grid mismatch')
                arr = src.read(1) > 0
            if arr.shape != after.shape:
                raise RuntimeError('Variant shape mismatch')
            labels, _ = ndimage.label(arr)
            ids = {int(r['local_component_id']) for r in selected}
            edge_ids = set(np.concatenate([labels[0], labels[-1], labels[:,0], labels[:,-1]]).tolist())
            if ids & edge_ids:
                raise RuntimeError('Previously complete component now touches patch edge')
            geometries = {int(v): model.fix_geometry(shape(g)) for g,v in shapes(labels.astype('int32'), mask=np.isin(labels, list(ids)), transform=trans, connectivity=4)}
            if set(geometries) != ids:
                raise RuntimeError('Missing target geometry')
            objects = []
            for row in selected:
                lid = int(row['local_component_id'])
                geom = geometries[lid]
                if geom is None or abs(geom.area-float(row['patch_component_area_m2'])) > .01:
                    raise RuntimeError('Target geometry does not match previous area record')
                objects.append(model.WaterObject(obj_id=lid, geom=geom, area_m2=geom.area))
            kept = {o.obj_id for o in model.filter_by_landuse_water(objects, str(lu_path), crs)}
            tf = Transformer.from_crs(crs, CRS.from_user_input(lu.crs), always_xy=True)
            for row in selected:
                lid = int(row['local_component_id'])
                geom = model.fix_geometry(transform(tf.transform, geometries[lid]))
                # Independently inspect the actual padded bounding window. Fail on read errors;
                # do not silently interpret the production function's caught exception as nonwater.
                win = geometry_window(lu, [mapping(geom)], pad_x=1, pad_y=1)
                data = lu.read(1, window=win)
                hit = int(np.count_nonzero(data == 1))
                if (hit > 0) != (lid in kept):
                    raise RuntimeError('Land-use rule differs from production function')
                counts = {str(v):int(n) for v,n in zip(*np.unique(data, return_counts=True))}
                records.append({**row, 'landuse_kept': lid in kept, 'window_water_value1_pixels': hit,
                                'window_value_counts': json.dumps(counts),
                                'window_nodata_pixels': int(np.count_nonzero(data == lu.nodata)) if lu.nodata is not None else 0})
            print(site, 'closing', closing, 'checked', len(selected), flush=True)
    summary = {'status':'complete', 'scope':'Previously affected objects only; original padded bounding-window land-use rule. No DEM or final storage. Closing OFF removals are a counterfactual.', 'groups':[]}
    for site, closing in groups:
        rr = [r for r in records if r['site']==site and int(r['closing'])==closing]
        total = sum(int(r['changed_core_pixels_in_component']) for r in rr)
        retained = sum(int(r['changed_core_pixels_in_component']) for r in rr if r['landuse_kept'])
        summary['groups'].append({'site':site,'closing':closing,'objects_checked':len(rr),'objects_kept':sum(r['landuse_kept'] for r in rr),'target_pixels':total,'target_pixels_kept':retained,'target_pixels_rejected':total-retained,'target_area_kept_m2':retained*100})
    out = ROOT / 'diagnostics/results/closing_landuse_gate' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    with (out/'objects.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__ == '__main__':
    main()
