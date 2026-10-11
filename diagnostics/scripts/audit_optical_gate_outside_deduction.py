"""Measure optical prior-gate candidate scope outside known deduction polygons."""
import json
from pathlib import Path
from datetime import datetime
from contextlib import ExitStack
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/'diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712/2024-06-01'

def main():
    info=json.loads((ROOT/'diagnostics/results/second_round_deduction_groups/20260926_204740_877320/summary.json').read_text(encoding='utf-8'))
    import sys
    sys.path.insert(0,str(ROOT/'code'))
    import s1s2_pond_production as model
    cfg0=model.load_yaml(ROOT/'config/pipeline_auto/pihe_pipeline_auto.yaml')
    cfg=next(c for c in model.resolve_multi_date_configs(cfg0) if c['date']=='2024-06-01')
    f=cfg.get('fusion',{})
    low=float(f.get('hysteresis_low_opt',max(float(f.get('hysteresis_low',.50))+.10,.62)))
    force=float(f.get('opt_only_force_probability',.93))
    if not .05<float(f.get('opt_only_prior_threshold',.12)):
        raise RuntimeError('Fixed prior now passes support gate; assumptions changed')
    out=ROOT/'diagnostics/results/second_round_optical_gate_outside_deduction'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    counts=dict(blocked_total=0,blocked_inside_deduction=0,blocked_outside_deduction=0)
    with ExitStack() as stack:
        prob=stack.enter_context(rasterio.open(UP/'c2_fused_probability.tif'))
        source=stack.enter_context(rasterio.open(UP/'c2_source_flag.tif'))
        water=stack.enter_context(rasterio.open(UP/'c4_water_mask.tif'))
        for s in (source,water):
            if s.shape!=prob.shape or s.crs!=prob.crs or not s.transform.almost_equals(prob.transform): raise RuntimeError('Grid mismatch')
        paths=sorted({p for g in info['groups'].values() for p in g['sources']})
        frames=[gpd.read_file(p).to_crs(prob.crs) for p in paths]
        profile=water.profile.copy();profile.update(dtype='uint8',nodata=0,compress='deflate')
        dst=stack.enter_context(rasterio.open(out/'blocked_optical_scope.tif','w',**profile))
        for i,(_,win) in enumerate(prob.block_windows(1),1):
            p=prob.read(1,window=win);s=source.read(1,window=win);w=water.read(1,window=win)
            blocked=(s==2)&np.isfinite(p)&(p>=low)&(p<force)&(w==0)
            codes=np.zeros(p.shape,dtype='uint8')
            if blocked.any():
                bounds=box(*prob.window_bounds(win));geoms=[]
                for frame in frames:
                    for pos in frame.sindex.query(bounds,predicate='intersects'):
                        g=frame.geometry.iloc[int(pos)];geoms.append(g if g.is_valid else g.buffer(0))
                deduct=rasterize(geoms,out_shape=p.shape,transform=prob.window_transform(win),all_touched=True)>0 if geoms else np.zeros(p.shape,dtype=bool)
                codes[blocked&deduct]=1;codes[blocked&~deduct]=2
                counts['blocked_total']+=int(blocked.sum())
                counts['blocked_inside_deduction']+=int((codes==1).sum())
                counts['blocked_outside_deduction']+=int((codes==2).sum())
            dst.write(codes,1,window=win)
            if i%500==0: print(f'Processed {i} raster blocks',flush=True)
        pixel_area=abs(prob.transform.a*prob.transform.e-prob.transform.b*prob.transform.d)
    assert counts['blocked_total']==counts['blocked_inside_deduction']+counts['blocked_outside_deduction']
    report=dict(status='complete',date='2024-06-01',low=low,force=force,fixed_prior=.05,pixels=counts,areas_km2={k:v*pixel_area/1e6 for k,v in counts.items()},deduction_sources=paths,codes={'0':'outside diagnostic selection','1':'blocked inside deduction union','2':'blocked outside deduction union'},limits=['Scope of pixel gate sensitivity, not a restored water mask.','Outside deduction does not establish pond membership or water truth.','No seed/connectivity/morphology/area/landuse/DEM propagation.','Deduction union uses saved group source polygons with all_touched=True; this is geometric screening, not a rerun of all downstream processing.'])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}');print('STATUS: complete')
if __name__=='__main__': main()
