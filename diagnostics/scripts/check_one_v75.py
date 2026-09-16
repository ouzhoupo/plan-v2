"""User-run: reproduce pond 31854 volume on original global block grid.

All official objects remain in the ID raster; only DEM blocks meeting target
bounds are read. Other objects get partial results, which are never exported.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from datetime import datetime
import fiona, rasterio, yaml
from shapely.geometry import shape, box
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer
from rasterio.windows import Window, bounds as window_bounds

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import pond_storage_monitor as model
DATE='2025-05-21'
TARGET=31854


def main():
    cfg=yaml.safe_load((ROOT/f'output/20250521/final/pihe/resolved_monitor_cfg_{DATE}.yaml').read_text(encoding='utf-8'))
    if cfg.get('skip_volume_above_area_m2') is not None:
        raise RuntimeError('Unexpected volume skipping configuration')
    prior=json.loads((ROOT/'diagnostics/results/one_boundary_h75/20260915_085911_220264/summary.json').read_text())
    if not prior.get('saved_levels_all_match') or prior['official_pond_id']!=TARGET:
        raise RuntimeError('Boundary baseline not verified')
    objects=[]; saved=None; target=None
    with fiona.open(ROOT/f'output/20250521/final/pihe/pond_storage_{DATE}.gpkg',layer='ponds') as src:
        water_crs=CRS.from_user_input(src.crs_wkt or src.crs)
        for f in src:
            p=dict(f['properties'])
            if not f['geometry']: raise RuntimeError('Official object has no geometry')
            g=shape(f['geometry'])
            obj=model.WaterObject(obj_id=int(p['pond_id']),geom=g,area_m2=float(p['area_m2']))
            obj.props.update(p)
            objects.append(obj)
            if obj.obj_id==TARGET:
                if target is not None: raise RuntimeError('Duplicate target ID')
                target=obj; saved=p.copy()
    if target is None or target.props.get('h75')!=34.0 or abs(target.geom.area-44400)>.01:
        raise RuntimeError('Target baseline changed')
    dem_path=ROOT/'data/dem/dem.tif'
    with rasterio.open(dem_path) as dem:
        dem_crs=CRS.from_user_input(dem.crs)
        tf=Transformer.from_crs(water_crs,dem_crs,always_xy=True)
        def projected(g):
            return model.fix_geometry(transform(tf.transform,g) if water_crs!=dem_crs else g)
        geoms=[]
        for obj in objects:
            if obj.props.get('h50') is None: continue
            g=projected(obj.geom)
            if g is not None and not g.is_empty: geoms.append(g)
        target_geom=projected(target.geom)
        global_bounds=model.intersect_bounds(tuple(dem.bounds),unary_union(geoms).bounds)
        global_win=model.window_from_intersection(dem,global_bounds)
        dem_transform=dem.transform
    original_iter=model.iter_windows
    selected=[]
    block_size=int(cfg.get('block_size',512))
    def target_blocks(width,height,size):
        if (width,height,size)!=(int(global_win.width),int(global_win.height),block_size):
            raise RuntimeError('Unexpected production processing grid')
        for rel in original_iter(width,height,size):
            absolute=Window(global_win.col_off+rel.col_off,global_win.row_off+rel.row_off,rel.width,rel.height)
            if target_geom.intersects(box(*window_bounds(absolute,dem_transform))):
                selected.append([int(absolute.col_off),int(absolute.row_off),int(absolute.width),int(absolute.height)])
                yield rel
    print('Reproducing target DEM blocks; official vector neighbours retained.',flush=True)
    # Only the in-memory iterator is replaced; production files are not edited.
    model.iter_windows=target_blocks
    try:
        model.compute_volumes(objects,str(dem_path),water_crs=water_crs,block_size=block_size,all_touched=bool(cfg.get('all_touched',False)))
    finally:
        model.iter_windows=original_iter
    if not selected or not target.props.get('dem_count'):
        raise RuntimeError('No target DEM coverage')
    comparisons={}
    for k in ['v05_m3','v50_m3','v75_m3','v95_m3','vmean_m3','dem_min','dem_mean','dem_max']:
        a=saved.get(k); b=target.props.get(k)
        difference=float(b)-float(a) if a is not None and b is not None else None
        comparisons[k]={'saved':a,'recomputed':b,'difference':difference,'matches_to_0_001':difference is not None and abs(difference)<=.0011}
    report={'status':'complete','official_pond_id':TARGET,'date':DATE,'h75_m':target.props['h75'],
            'selected_dem_blocks':selected,'global_dem_window':[float(global_win.col_off),float(global_win.row_off),float(global_win.width),float(global_win.height)],
            'official_objects_loaded':len(objects),'target_dem_count':target.props['dem_count'],
            'v75_matches':comparisons['v75_m3']['matches_to_0_001'],'comparisons':comparisons,
            'scope':'Official saved geometries and levels, original compute_volumes with target block selection only; all official vector neighbours retained. No parameter intervention or new water-level sampling.',
            'limits':'Historical in-memory geometry/order equivalence must be assessed from agreement; mismatch is not automatically a water-level error. Other objects computed partially and not exported. Reproduction does not establish true storage accuracy.'}
    out=ROOT/'diagnostics/results/one_v75'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
