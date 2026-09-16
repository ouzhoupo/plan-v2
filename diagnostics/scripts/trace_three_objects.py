"""Validate three sample geometries and trace saved water/candidate/land-use stages.
Reads existing outputs only. Run from any directory with the project GIS Python.
"""
from pathlib import Path
from datetime import datetime
import sys
import json
import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import fiona
import yaml
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import pond_storage_monitor as model
from build_pihe_gis_samples import transformed_geom, landuse_evidence

OUT = ROOT / 'diagnostics/results/three_object_trace' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True, exist_ok=False)
records=[]
for date, ids in [('2024-06-01',[14660]),('2025-05-21',[46608,45219])]:
    print('Checking',date,flush=True)
    day=ROOT/'output'/date.replace('-','')
    final=day/'final/pihe'
    cfg=yaml.safe_load((final/f'resolved_monitor_cfg_{date}.yaml').read_text(encoding='utf-8'))
    if cfg.get('watershed_split_enable') or cfg.get('linear_artifact_filter_enable') or cfg.get('landuse_water_hard_mask'):
        raise RuntimeError('Unsupported active screening configuration')
    # Use the actual production polygonization and ID assignment, including geometry repair.
    objects=model.polygonize_water_mask(str(final/f'pond_water_mask_{date}.tif'),
        min_area_m2=float(cfg.get('min_water_area_m2',500)),
        max_area_m2=float(cfg['max_water_area_m2']),
        simplify_tolerance=float(cfg.get('simplify_tolerance_m',0)),
        watershed_split_enable=False)
    selected={o.obj_id:o for o in objects if o.obj_id in ids}
    with fiona.open(final/f'pond_storage_{date}.gpkg') as src:
        actual={int(f['properties']['pond_id']):shape(f['geometry']) for f in src}
    sample_path=ROOT/f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg'
    with fiona.open(sample_path) as src:
        samples={int(f['properties']['pond_id']):shape(f['geometry']) for f in src if f['properties']['pond_id'] in ids}
    with rasterio.open(final/f'pond_water_mask_{date}.tif') as ds:
        crs=ds.crs
    for pid in ids:
        obj=selected[pid]; geom=obj.geom
        error=geom.symmetric_difference(samples[pid]).area
        if error>0.01:
            raise RuntimeError(f'Sample ID/geometry mismatch {date} {pid}: {error}')
        kept=model.filter_by_landuse_water([obj],str(ROOT/'data/landuse_water/PSH_土地利用水体.tif'),crs)
        r={'date':date,'pond_id':pid,'area_m2':geom.area,'sample_geometry_difference_m2':error,
           'present_in_final':pid in actual,'production_landuse_kept':bool(kept)}
        if pid in actual:
            r['final_geometry_difference_m2']=geom.symmetric_difference(actual[pid]).area
            if r['final_geometry_difference_m2']>0.01:
                raise RuntimeError('Final geometry mismatch')
        with rasterio.open(ROOT/'data/landuse_water/PSH_土地利用水体.tif') as lu:
            r['landuse']=landuse_evidence(lu,transformed_geom(geom,crs,lu.crs))
        r['vector_contacts']=[]
        layers=list((day/'pihe/buffered_river').glob('*.shp'))+[ROOT/'data/reservoir/psh_res.shp']
        for path in layers:
            with fiona.open(path) as src:
                g=transformed_geom(geom,crs,src.crs_wkt or src.crs)
                for f in src.filter(bbox=g.bounds):
                    if not f['geometry']: continue
                    v=model.fix_geometry(shape(f['geometry']))
                    if v is None or v.is_empty or not g.intersects(v): continue
                    overlap=transformed_geom(g.intersection(v),src.crs_wkt or src.crs,crs).area
                    r['vector_contacts'].append({'file':str(path),'fid':f['id'],'overlap_m2':overlap,
                        'properties':dict(f['properties'])})
        r['stages']={}
        for d in ['2024-06-01','2025-05-21']:
            upstream=ROOT/'output'/d.replace('-','')/'pihe'
            arrays={}
            target_transform=None
            for name in ['water_mask','water_prob','source_flag','pond_water_mask']:
                raster_path=(upstream/f'{name}_{d}.tif') if name!='pond_water_mask' else (upstream.parent/'final/pihe'/f'{name}_{d}.tif')
                with rasterio.open(raster_path) as src:
                    win=geometry_window(src,[geom.__geo_interface__])
                    arrays[name]=src.read(1,window=win)
                    if target_transform is None:
                        target_transform=src.window_transform(win)
                        target_shape=arrays[name].shape
                    elif arrays[name].shape!=target_shape or not src.window_transform(win).almost_equals(target_transform):
                        raise RuntimeError(f'Model output grid mismatch: {raster_path}')
                    inside=geometry_mask([geom.__geo_interface__],out_shape=arrays[name].shape,
                        transform=src.window_transform(win),invert=True)
                    transform=src.window_transform(win)
                    bounds=src.window_bounds(win)
            with rasterio.open(ROOT/f'data/s2/s2_stack_{d}.tif') as src:
                # Align reflectance to exactly the same pixel centers as the output.
                # A common CRS alone does not imply an identical raster origin.
                with WarpedVRT(src,crs=crs,transform=target_transform,
                               width=target_shape[1],height=target_shape[0],
                               resampling=Resampling.bilinear,
                               dtype='float32',nodata=float('nan')) as aligned:
                    a=aligned.read([2,5],masked=True).astype('float32').filled(np.nan)
            with np.errstate(invalid='ignore',divide='ignore'):
                index=(a[0]-a[1])/(a[0]+a[1]+1e-8)
            proxy=inside & np.isfinite(index) & (index<=0)
            # This is an optical disagreement proxy, NOT ground-truth non-water.
            n=int(proxy.sum())
            r['stages'][d]={'optical_nonpositive_pixels':n,
                's2_alignment':'bilinear_to_model_grid',
                'optical_valid_pixels':int((inside & np.isfinite(index)).sum()),
                'water_fraction_on_optical_nonpositive':float((arrays['water_mask'][proxy]>0).mean()) if n else None,
                'candidate_fraction_on_optical_nonpositive':float((arrays['pond_water_mask'][proxy]>0).mean()) if n else None,
                'probability_mean_on_optical_nonpositive':float(arrays['water_prob'][proxy].mean()) if n else None,
                'source_counts_on_optical_nonpositive':{str(i):int((arrays['source_flag'][proxy]==i).sum()) for i in range(5)}}
        records.append(r)
        print('Finished',date,pid,flush=True)
(OUT/'trace.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print('RESULT:',OUT/'trace.json',flush=True)
