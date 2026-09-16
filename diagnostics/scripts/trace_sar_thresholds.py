"""Audit coarse SAR threshold distributions; no training or model output writes."""
from datetime import datetime
import json
import numpy as np
from affine import Affine
from rasterio.features import geometry_mask
from pihe_component_helpers import ROOT,m

OUT=ROOT/'diagnostics/results/sar_threshold_audit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
dates=['2024-06-01','2025-05-21']
cache={}; report={'note':'Coarse-grid pixel counts are diagnostic sample counts, not native-resolution water areas. Common-support recomputation is sensitivity analysis.', 'dates':{}}

def describe(values):
    if not values.size: return {'n':0}
    q=np.nanpercentile(values,[5,25,40,50,75,95])
    raw=float(m.otsu_threshold(values))
    clipped=float(np.clip(raw,q[0],q[2]))
    return {'n':int(values.size),'mean':float(values.mean(dtype=np.float64)),
            'p05':float(q[0]),'p25':float(q[1]),'p40':float(q[2]),'median':float(q[3]),'p75':float(q[4]),'p95':float(q[5]),
            'otsu_raw':raw,'threshold_clipped':clipped,'threshold_rounded':round(clipped,3),
            'clip_side':'p05' if raw<q[0] else 'p40' if raw>q[2] else 'none',
            'sigma_rounded':round(float(np.clip((q[4]-q[1])/4,1.2,2.5)),3)}

for date in dates:
    print('Auditing threshold:',date,flush=True)
    initial=m.load_yaml(ROOT/'config/pipeline_auto/pihe_pipeline_auto.yaml')
    cfg=next(c for c in m.resolve_multi_date_configs(initial) if c['date']==date)
    cfg=m.resolve_runtime_auto_inputs(cfg); g=m.infer_grid(cfg)
    s1=m.open_s1_scenes(cfg,g); dem=m.open_optional_dem_reader(cfg,g)
    try:
        if len(s1)!=1: raise RuntimeError('Expected one S1 scene')
        scene=s1[0]; size=int(cfg.get('fusion',{}).get('coarse_sample_size',1024))
        shp=(min(size,g.height),min(size,g.width))
        vv=scene.vv.read_coarse(out_shape=shp); vh=scene.vh.read_coarse(out_shape=shp)
        initial_valid=(scene.vv.vrt.read_masks(scene.vv.band,out_shape=shp)>0)&(scene.vh.vrt.read_masks(scene.vh.band,out_shape=shp)>0)&np.isfinite(vv)&np.isfinite(vh)
        if not scene.in_db:
            vv=10*np.log10(np.maximum(vv,1e-6)).astype('float32'); vh=10*np.log10(np.maximum(vh,1e-6)).astype('float32')
        valid=initial_valid.copy()
        if dem:
            slope=m.compute_slope_deg(dem.read_coarse(out_shape=shp),g.res*g.width/shp[1])
            valid &= np.isfinite(slope)&(slope<12)
        tr=g.transform*Affine.scale(g.width/shp[1],g.height/shp[0])
        aoi=geometry_mask([g.aoi_geom.__geo_interface__],out_shape=shp,transform=tr,invert=True)
        groups={'production_valid':valid,'aoi_only':valid&aoi,'outside_aoi':valid&~aoi}
        stats={name:{'vv':describe(vv[mask]),'vh':describe(vh[mask])} for name,mask in groups.items()}
        saved=json.loads((ROOT/f'output/{date.replace("-", "")}/pihe/fusion_summary_{date}.json').read_text(encoding='utf-8'))
        params=list(saved['s1_scene_params'].values())
        if len(params)!=1: raise RuntimeError('Ambiguous saved params')
        predicted={'th_vv':stats['production_valid']['vv']['threshold_rounded'],
                   'th_vh':stats['production_valid']['vh']['threshold_rounded'],
                   'sigma_vv':stats['production_valid']['vv']['sigma_rounded'],
                   'sigma_vh':stats['production_valid']['vh']['sigma_rounded']}
        report['dates'][date]={'coarse_shape':shp,'before_slope_n':int(initial_valid.sum()),'after_slope_n':int(valid.sum()),
            'outside_aoi_n':int((valid&~aoi).sum()),'groups':stats,'saved_params':params[0],
            'recomputed_params':predicted,'matches_saved':predicted==params[0]}
        cache[date]=(vv,vh,valid,aoi,tr,g.crs)
    finally:
        m.close_scenes(s1,[],extras=[dem] if dem else [])
a,b=[cache[d] for d in dates]
if a[0].shape!=b[0].shape or not a[4].almost_equals(b[4]) or a[5]!=b[5]:
    raise RuntimeError('Different coarse grids: common-support comparison requires alignment')
common=a[2]&b[2]&a[3]&b[3]
report['common_valid_aoi_n']=int(common.sum())
for date in dates:
    vv,vh,*_=cache[date]
    report['dates'][date]['common_valid_aoi']={'vv':describe(vv[common]),'vh':describe(vh[common])}
(OUT/'thresholds.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:',OUT/'thresholds.json',flush=True)
