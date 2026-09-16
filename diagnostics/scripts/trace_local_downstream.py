"""Bounded patch experiment: outside exported experiment tiles stays baseline."""
from pathlib import Path
from datetime import datetime
import sys, json, math
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import geometry_window
from shapely.geometry import shape, box
import fiona, yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import pond_storage_monitor as m
SOURCE=ROOT/'diagnostics/results/positive_label_experiment/20260912_074812_069687'
OUT=ROOT/'diagnostics/results/local_downstream'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
report={'scope':'Local patch intervention; historical water outside experiment tiles unchanged. No DEM volumes or basin totals.', 'objects':[]}
for date,pid in [('2025-05-21',46608),('2025-05-21',45219),('2024-06-01',14660)]:
    day=ROOT/'output'/date.replace('-','')
    cfg=yaml.safe_load((day/f'final/pihe/resolved_monitor_cfg_{date}.yaml').read_text(encoding='utf-8'))
    if any(cfg.get(k) for k in ['watershed_split_enable','linear_artifact_filter_enable','landuse_water_hard_mask','cropland_mask_tif','extra_vectors']):
        raise RuntimeError('Unsupported active configuration')
    with fiona.open(ROOT/f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg') as src:
        gs=[shape(f['geometry']) for f in src if f['properties'].get('pond_id')==pid]
    if not gs or any(not gs[0].equals(g) for g in gs[1:]): raise RuntimeError('Ambiguous sample')
    geom=gs[0]
    sub=OUT/f'{date}_{pid}'; sub.mkdir()
    with rasterio.open(day/f'final/pihe/pond_water_mask_{date}.tif') as saved:
        w=geometry_window(saved,[geom.buffer(100).__geo_interface__])
        bs=int(cfg.get('block_size',512))
        c=max(0,(int(w.col_off)//bs-1)*bs); r=max(0,(int(w.row_off)//bs-1)*bs)
        endc=min(saved.width,(math.ceil((w.col_off+w.width)/bs)+1)*bs)
        endr=min(saved.height,(math.ceil((w.row_off+w.height)/bs)+1)*bs)
        win=Window(c,r,endc-c,endr-r); trans=saved.window_transform(win)
        historical=saved.read(1,window=win)>0
        bounds=saved.window_bounds(win); crs=saved.crs
    with rasterio.open(day/f'pihe/water_mask_{date}.tif') as src:
        sw=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
        if not src.window_transform(sw).almost_equals(trans): raise RuntimeError('Grid mismatch')
        baseline=src.read(1,window=sw)>0
    changed=baseline.copy(); coverage=np.zeros_like(baseline)
    for p in SOURCE.glob(f'{date}_*_masks.tif'):
        with rasterio.open(p) as tile:
            tw=from_bounds(*tile.bounds,transform=trans).round_offsets().round_lengths()
            x,y=int(tw.col_off),int(tw.row_off)
            x0,y0=max(0,x),max(0,y); x1,y1=min(changed.shape[1],x+tile.width),min(changed.shape[0],y+tile.height)
            if x1<=x0 or y1<=y0: continue
            if not tile.transform.almost_equals(rasterio.windows.transform(Window(x,y,tile.width,tile.height),trans)): raise RuntimeError('Tile mismatch')
            a=tile.read(window=Window(x0-x,y0-y,x1-x0,y1-y0))>0
            if not np.array_equal(a[0],baseline[y0:y1,x0:x1]): raise RuntimeError('Historical tile mismatch')
            changed[y0:y1,x0:x1]=a[2]; coverage[y0:y1,x0:x1]=True
    aoi=m.read_union_geometry(str(ROOT/'data/aoi/pihe.shp'),crs)
    rivers=day/'pihe/buffered_river'; reservoir=ROOT/'data/reservoir/psh_res.shp'; lu=ROOT/'data/landuse_water/PSH_土地利用水体.tif'
    if not rivers.is_dir() or not list(rivers.glob('*.shp')) or not reservoir.exists() or not lu.exists(): raise RuntimeError('Missing deduction/landuse input')
    layers=m.build_deduct_layers(river_dir=str(rivers),reservoir_shp=str(reservoir),extra_vectors=[],target_crs=crs,bbox_filter=bounds,line_buffer_m=float(cfg.get('line_buffer_m',0)))
    result={'date':date,'pond_id':pid,'variants':{}}
    for name,array in [('baseline',baseline),('experiment',changed)]:
        inp=sub/f'{name}_water.tif'; candidate=sub/f'{name}_candidate.tif'
        with rasterio.open(inp,'w',driver='GTiff',height=array.shape[0],width=array.shape[1],count=1,dtype='uint8',crs=crs,transform=trans,nodata=0,compress='deflate') as dst: dst.write(array.astype('uint8'),1)
        m.prepare_water_mask(str(inp),str(candidate),process_bounds=bounds,aoi_geom=aoi,deduct_layers=layers,water_value=None,respect_source_mask=bool(cfg.get('respect_source_mask',False)),block_size=bs,close_radius_px=int(cfg.get('close_radius_px',0)),open_radius_px=int(cfg.get('open_radius_px',0)),fill_holes=bool(cfg.get('fill_holes',False)),all_touched=bool(cfg.get('all_touched',False)))
        with rasterio.open(candidate) as ds: output=ds.read(1)>0
        if name=='baseline':
            mismatch=int((output!=historical).sum()); result['baseline_candidate_mismatch_pixels']=mismatch
            if mismatch: raise RuntimeError(f'Baseline mismatch {date} {pid}: {mismatch}; do not interpret experiment')
        objects=m.polygonize_water_mask(str(candidate),min_area_m2=float(cfg['min_water_area_m2']),max_area_m2=float(cfg['max_water_area_m2']),simplify_tolerance=float(cfg.get('simplify_tolerance_m',0)),watershed_split_enable=False)
        kept=m.filter_by_landuse_water(objects,str(lu),crs); ids={o.obj_id for o in kept}
        selected=[o for o in objects if o.geom.intersection(geom).area>0]
        result['variants'][name]=[{'local_id':o.obj_id,'area_m2':o.area_m2,'overlap_original_m2':o.geom.intersection(geom).area,'landuse_kept':o.obj_id in ids,'touches_local_boundary':o.geom.intersects(box(*bounds).boundary),'geometry':o.geom.__geo_interface__} for o in selected]
    report['objects'].append(result)
    (OUT/'trace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(date,pid,'complete',flush=True)
print('RESULT:',OUT/'trace.json')
