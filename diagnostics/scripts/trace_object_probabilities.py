"""Recreate probability components on production blocks intersecting 3 objects.
Training is repeated per date; no full output pipeline or production writes.
"""
from pathlib import Path
from datetime import datetime
import json
import hashlib
import platform
import numpy as np
import fiona
import sklearn
import rasterio
from shapely.geometry import shape, box
from rasterio.features import geometry_mask
from rasterio.windows import bounds
from pihe_component_helpers import ROOT, setup, calc, m, add_stats, finish_stats

out=ROOT/'diagnostics/results/object_probabilities'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
out.mkdir(parents=True,exist_ok=False)
targets=[]
for date,pid in [('2024-06-01',14660),('2025-05-21',46608),('2025-05-21',45219)]:
    p=ROOT/f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg'
    with fiona.open(p) as src:
        f=next(f for f in src if f['properties']['pond_id']==pid)
        targets.append((f'{date}_{pid}',shape(f['geometry']),src.crs_wkt))
result={'runtime':{'python':platform.python_version(),'sklearn':sklearn.__version__,'rasterio':rasterio.__version__},
        'production_sha256':hashlib.sha256((ROOT/'code/s1s2_pond_production.py').read_bytes()).hexdigest(),
        'notes':'Optical groups use date-specific MNDWI and clear mask; not ground truth or fixed cross-date support.',
        'dates':{}}
for date in ['2024-06-01','2025-05-21']:
    saved=ROOT/f'output/{date.replace("-", "")}/pihe/fusion_summary_{date}.json'
    old=json.loads(saved.read_text(encoding='utf-8'))
    if old['prior_source_count']!=0: raise RuntimeError('This probe assumes no observed prior')
    print('Training optical pseudo-label model:',date,flush=True)
    ctx=setup(date)
    try:
        g=ctx['grid']; reader=ctx['readers']['prob']
        if reader.crs!=g.crs or reader.width!=g.width or reader.height!=g.height or not reader.transform.almost_equals(g.transform):
            raise RuntimeError('Recomputed grid differs from saved output')
        for _,_,crs in targets:
            if rasterio.crs.CRS.from_wkt(crs)!=reader.crs: raise RuntimeError('Sample CRS mismatch')
        acc={}; checks={}; blocks=0
        for win in m.iter_windows(g.width,g.height,int(ctx['fcfg'].get('block_size',512))):
            tile=box(*bounds(win,g.transform))
            hits=[(key,geom) for key,geom,_ in targets if tile.intersects(geom)]
            if not hits: continue
            blocks+=1
            sar,opt,model_p,blend,fused,source,conf,prior=calc(ctx,win)
            stored=reader.read(1,window=win)
            stored_src=ctx['readers']['source'].read(1,window=win)
            mask=ctx['readers']['mask'].read(1,window=win)>0
            tr=rasterio.windows.transform(win,g.transform)
            for key,geom in hits:
                inside=geometry_mask([geom.__geo_interface__],out_shape=fused.shape,transform=tr,invert=True)
                v=checks.setdefault(key,{'pixels':0,'finite_comparison_pixels':0,'abs_sum':0.,'max_abs':0.,'source_mismatch':0,'sar_marked_nan':0})
                good=inside & np.isfinite(stored) & np.isfinite(fused)
                diff=np.abs(stored[good]-fused[good])
                v['pixels']+=int(inside.sum()); v['finite_comparison_pixels']+=int(good.sum())
                v['abs_sum']+=float(diff.sum(dtype=np.float64)); v['max_abs']=max(v['max_abs'],float(diff.max(initial=0)))
                v['source_mismatch']+=int((inside & (source!=stored_src)).sum())
                v['sar_marked_nan']+=int((inside & sar['sar_valid'] & ~np.isfinite(sar['sar_prob'])).sum())
                proxy=inside & opt['opt_clear'] & np.isfinite(opt['features']['mndwi']) & (opt['features']['mndwi']<=0)
                for group,support in [('whole',inside),('clear_optical_nonpositive',proxy)]:
                    for name,arr in [('sar',sar['sar_prob']),('base_optical',opt['opt_prob_base']),('model_optical',model_p),('blend_optical',blend),('fused',fused),('saved_water',mask.astype('float32'))]:
                        valid=support
                        if name=='sar': valid=support & sar['sar_valid']
                        if name in ['base_optical','model_optical','blend_optical']: valid=support & opt['opt_clear']
                        add_stats(acc,f'{key}|{group}|{name}',arr,valid,g.res*g.res)
        for v in checks.values():
            v['mean_abs']=v['abs_sum']/v['finite_comparison_pixels'] if v['finite_comparison_pixels'] else None
        entry={'metrics':ctx['metrics'],'saved_metrics':old['optical_calibration'],'blend':ctx['blend'],
               'production_blocks_processed':blocks,'validation':checks,'stats':finish_stats(acc,g.res*g.res)}
        result['dates'][date]=entry
        (out/'components.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(date,'validation:',json.dumps(checks),flush=True)
    finally:
        for r in ctx['readers'].values(): r.close()
        m.close_scenes(ctx['s1'],ctx['s2'],extras=[ctx['dem']] if ctx['dem'] is not None else [])
print('RESULT:',out/'components.json',flush=True)
