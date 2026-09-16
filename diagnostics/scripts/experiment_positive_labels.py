"""Diagnostic ablation: remove training-only positive labels with MNDWI<=0.
Not a proposed production filter. No full masks, volumes or production writes.
"""
from datetime import datetime
import argparse
import json
import numpy as np
import fiona
import rasterio
from shapely.geometry import shape,box
from rasterio.features import geometry_mask
from rasterio.windows import bounds,transform
from sklearn.base import clone
from pihe_component_helpers import ROOT,m,setup,calc,add_stats,finish_stats

def production_mask(values,ctx,win):
    g=ctx['grid']; cfg=ctx['fcfg']
    prob,src,conf,prior=values[4:8]
    aoi=m.rasterize_geom_mask(g.aoi_geom,prob.shape,transform(win,g.transform))
    prob=np.where(aoi,prob,np.nan).astype('float32')
    src=np.where(aoi,src,0).astype('uint8')
    conf=np.where(aoi,np.clip(conf*100,0,100),0).astype('uint8')
    mask=m.source_aware_hysteresis(np.nan_to_num(prob,nan=0),src,conf.astype('float32')/100,prior,res_m=g.res,cfg=cfg)
    mask=m.refine_mask(mask,close_px=int(cfg.get('close_radius_px',0)),open_px=int(cfg.get('open_radius_px',0)),fill_holes=bool(cfg.get('fill_holes',False)))
    return np.where(aoi,mask,0).astype('uint8')

parser=argparse.ArgumentParser()
parser.add_argument('--released-samples',action='store_true',help='Compare all 31 previously reviewed released objects at their own date')
args=parser.parse_args()
released={}
if args.released_samples:
    sample_path=ROOT/'diagnostics/results/released_object_review/20260913_111008_061222/review.json'
    sample_data=json.loads(sample_path.read_text(encoding='utf-8'))
    for window in sample_data['windows']:
        for sample in window['samples']:
            key=f"released_{window['date']}_{window['window_id']}_{sample['local_id']}"
            released[key]={'date':window['date'],'window_id':window['window_id'],**sample}
    if len(released)!=31: raise RuntimeError('Expected all 31 reviewed samples')
OUT=ROOT/'diagnostics/results/positive_label_experiment'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
targets=[]
for date,pid in ([] if args.released_samples else [('2024-06-01',14660),('2025-05-21',46608),('2025-05-21',45219)]):
    with fiona.open(ROOT/f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg') as src:
        targets.append((f'{date}_{pid}',shape(next(f for f in src if f['properties']['pond_id']==pid)['geometry'])))
report={'scope':'Training-only removal of positive MNDWI<=0 labels; baseline alpha/quality fixed; no causal claim of real-world accuracy. Date-specific clear optical groups.', 'dates':{}}
if args.released_samples:
    report['released_samples']=released
    report['evaluation_caveat']='Fixed previously reviewed sample footprints; not an independent probability sample of full AOI. Compare each footprint at the same date; local IDs include source window.'
for date in ['2024-06-01','2025-05-21']:
    point_file=ROOT/f'diagnostics/results/pseudolabel_sampling/20260912_065722_800387/{date}_review_points.json'
    review_points=[] if args.released_samples else [p for p in json.loads(point_file.read_text(encoding='utf-8')) if p['split']=='validation']
    # Assess all reviewed held-out points, not only the MNDWI-defined conflict group.
    # Point coordinates are production pixel centers; this box encloses one pixel.
    date_targets=list(targets)+[(f"review_{date}_{p['sample_index']}",box(p['x']-5,p['y']-5,p['x']+5,p['y']+5)) for p in review_points]
    if args.released_samples:
        date_targets=[(key,shape(sample['geometry'])) for key,sample in released.items() if sample['date']==date]
        print(date,'fixed reviewed objects:',len(date_targets),flush=True)
    print('Baseline training:',date,flush=True)
    captured={}; original=m.train_test_split
    def capture(*args,**kwargs):
        pieces=original(*args,**kwargs)
        captured['X']=pieces[0]; captured['y']=pieces[2]
        return pieces
    m.train_test_split=capture
    try:
        ctx=setup(date)
    finally:
        m.train_test_split=original
    try:
        if ctx['model'] is None or not captured: raise RuntimeError('Baseline training unavailable')
        X,y=captured['X'],captured['y']
        idx=list(m.MODEL_FEATURES).index('mndwi')
        remove=(y==1)&np.isfinite(X[:,idx])&(X[:,idx]<=0)
        keep=~remove
        if len(np.unique(y[keep]))<2: raise RuntimeError('Too few classes after filtering')
        alternate=clone(ctx['model'])
        print('Diagnostic training:',date,'removed positives:',int(remove.sum()),flush=True)
        alternate.fit(X[keep],y[keep])
        experimental=dict(ctx); experimental['model']=alternate
        # Hold quality and alpha constant: isolate training-set change.
        acc={}; validation={}; mask_counts={}; g=ctx['grid']; blocks=0
        for win in m.iter_windows(g.width,g.height,int(ctx['fcfg'].get('block_size',512))):
            tile=box(*bounds(win,g.transform))
            hits=[(key,geom) for key,geom in date_targets if tile.intersects(geom)]
            if not hits: continue
            blocks+=1
            base=calc(ctx,win); changed=calc(experimental,win)
            sar,opt,model_p,blend,fused,source,conf,prior=base
            saved=ctx['readers']['prob'].read(1,window=win)
            saved_source=ctx['readers']['source'].read(1,window=win)
            saved_mask=ctx['readers']['mask'].read(1,window=win)>0
            baseline_mask=production_mask(base,ctx,win)>0
            experimental_mask=production_mask(changed,ctx,win)>0
            # Export only full production blocks intersecting an original object.
            if any(not key.startswith('review_') for key,_ in hits):
                path=OUT/f'{date}_r{int(win.row_off)}_c{int(win.col_off)}_masks.tif'
                with rasterio.open(path,'w',driver='GTiff',width=int(win.width),height=int(win.height),count=3,
                                   dtype='uint8',crs=g.crs,transform=transform(win,g.transform),compress='deflate') as dst:
                    for band,(name,array) in enumerate([('saved_water_mask',saved_mask),('baseline_recomputed',baseline_mask),('filtered_training',experimental_mask)],1):
                        dst.write(array.astype('uint8'),band); dst.set_band_description(band,name)
            for key,geom in hits:
                inside=geometry_mask([geom.__geo_interface__],out_shape=fused.shape,transform=transform(win,g.transform),invert=True)
                good=inside&np.isfinite(saved)&np.isfinite(fused)
                diff=np.abs(saved[good]-fused[good])
                v=validation.setdefault(key,{'pixels':0,'abs_sum':0.,'source_mismatch':0})
                v['pixels']+=int(good.sum()); v['abs_sum']+=float(diff.sum(dtype=np.float64))
                v['source_mismatch']+=int((inside&(source!=saved_source)).sum())
                optical=inside&opt['opt_clear']&np.isfinite(opt['features']['mndwi'])
                for group,support in [('whole',inside),('clear_mndwi_nonpositive',optical&(opt['features']['mndwi']<=0)),('clear_mndwi_positive',optical&(opt['features']['mndwi']>0))]:
                    counts=mask_counts.setdefault(f'{key}|{group}',{k:0 for k in ['support_pixels','saved_water_pixels','baseline_water_pixels','experimental_water_pixels','baseline_saved_mismatch','removed_pixels','added_pixels']})
                    for name,selection in [('support_pixels',support),('saved_water_pixels',support&saved_mask),
                        ('baseline_water_pixels',support&baseline_mask),('experimental_water_pixels',support&experimental_mask),
                        ('baseline_saved_mismatch',support&(baseline_mask!=saved_mask)),
                        ('removed_pixels',support&baseline_mask&~experimental_mask),('added_pixels',support&~baseline_mask&experimental_mask)]:
                        counts[name]+=int(selection.sum())
                    for label,values in [('baseline',base),('filtered_training',changed)]:
                        for name,arr,valid in [('model_optical',values[2],support&opt['opt_clear']),('fused',values[4],support)]:
                            add_stats(acc,f'{key}|{group}|{label}|{name}',arr,valid,g.res*g.res)
        for v in validation.values(): v['mean_abs']=v['abs_sum']/v['pixels'] if v['pixels'] else None
        point_results=[]
        for p in review_points:
            key=f"review_{date}_{p['sample_index']}"
            values={}
            for label in ['baseline','filtered_training']:
                for name in ['model_optical','fused']:
                    stat=acc.get(f'{key}|whole|{label}|{name}')
                    values[f'{label}_{name}']=stat['sum']/stat['n'] if stat and stat['n'] else None
            point_results.append({**p,**values,'mask_counts':mask_counts.get(f'{key}|whole')})
        report['dates'][date]={'reviewed_validation_points':point_results,
            'review_caveat':'Held out from model fitting, but spatially related to training and preselected by optical index. Visual interpretations are provisional, not independent survey truth.',
            'train_before':len(y),'positive_before':int((y==1).sum()),'removed_positive':int(remove.sum()),
            'positive_remaining':int(((y==1)&keep).sum()),'negative_unchanged':int((y==0).sum()),
            'fixed_alpha':ctx['blend'],'fixed_quality':ctx['quality'],'baseline_metrics':ctx['metrics'],
            'blocks':blocks,'validation':validation,'pixel_area_m2':g.res*g.res,'mask_counts':mask_counts,
            'mask_scope':'Counts inside existing object footprints / single review pixels only; no new pond objects, downstream screening or volume. Inspect baseline_saved_mismatch before interpreting differences.',
            'stats':finish_stats(acc,g.res*g.res)}
        (OUT/'experiment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(date,'complete',flush=True)
    finally:
        for r in ctx['readers'].values(): r.close()
        m.close_scenes(ctx['s1'],ctx['s2'],extras=[ctx['dem']] if ctx['dem'] is not None else [])
print('RESULT:',OUT/'experiment.json',flush=True)
