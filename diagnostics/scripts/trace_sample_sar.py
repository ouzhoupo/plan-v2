"""Decompose SAR scores at the 32 reviewed actual pseudo-label samples.
No training, full pipeline or production writes. Preserves production block context.
"""
from datetime import datetime
import json
import numpy as np
from rasterio.transform import rowcol
from pihe_component_helpers import ROOT, m

INPUT=ROOT/'diagnostics/results/pseudolabel_sampling/20260912_025919_234027'
OUT=ROOT/'diagnostics/results/sample_sar'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
dates=['2024-06-01','2025-05-21']
summaries={d:json.loads((ROOT/f'output/{d.replace("-", "")}/pihe/fusion_summary_{d}.json').read_text(encoding='utf-8')) for d in dates}
report={'note':'Other-date threshold test holds current VV/VH/texture/slope fixed; diagnostic sensitivity, not a corrected prediction.', 'dates':{}}
for date in dates:
    print('Checking SAR components:',date,flush=True)
    points=json.loads((INPUT/f'{date}_review_points.json').read_text(encoding='utf-8'))
    cfg0=m.load_yaml(ROOT/'config/pipeline_auto/pihe_pipeline_auto.yaml')
    cfg=next(c for c in m.resolve_multi_date_configs(cfg0) if c['date']==date)
    cfg=m.resolve_runtime_auto_inputs(cfg); g=m.infer_grid(cfg); f=cfg.get('fusion',{})
    scenes=m.open_s1_scenes(cfg,g); dem=m.open_optional_dem_reader(cfg,g)
    try:
        if len(scenes)!=1: raise RuntimeError('This diagnostic requires one S1 scene per date')
        sc=scenes[0]
        params=list(summaries[date]['s1_scene_params'].values())
        alt=list(summaries[dates[1-dates.index(date)]]['s1_scene_params'].values())
        if len(params)!=1 or len(alt)!=1: raise RuntimeError('Ambiguous scene parameters')
        p=params[0]; alternate=alt[0]; sc.params=p
        windows=list(m.iter_windows(g.width,g.height,int(f.get('block_size',512))))
        rows=[]
        for block in sorted({r['block'] for r in points}):
            win=windows[block]
            vv=sc.vv.read(win).astype('float32'); vh=sc.vh.read(win).astype('float32')
            if not sc.in_db:
                vv=10*np.log10(np.maximum(vv,1e-6)); vh=10*np.log10(np.maximum(vh,1e-6))
            slope=m.compute_slope_deg(dem.read(win),g.res) if dem else np.zeros_like(vv)
            std=m.local_std(vv,size=int(f.get('sar_texture_size',5)))
            tex=m.sigmoid((1.8-std)/.6).astype('float32')
            penalty=np.clip(1-np.maximum(slope-5,0)/15,.15,1).astype('float32')
            pv=m.sigmoid((p['th_vv']-vv)/max(p['sigma_vv'],1)).astype('float32')
            ph=m.sigmoid((p['th_vh']-vh)/max(p['sigma_vh'],1)).astype('float32')
            total=(.5*pv+.3*ph+.2*tex)*penalty
            av=m.sigmoid((alternate['th_vv']-vv)/max(p['sigma_vv'],1))
            ah=m.sigmoid((alternate['th_vh']-vh)/max(p['sigma_vh'],1))
            changed=(.5*av+.3*ah+.2*tex)*penalty
            for point in [r for r in points if r['block']==block]:
                rr,cc=rowcol(g.transform,point['x'],point['y'])
                r=int(rr-win.row_off); c=int(cc-win.col_off)
                if not (0<=r<vv.shape[0] and 0<=c<vv.shape[1]): raise RuntimeError('Point/block mismatch')
                def at(a): return float(a[r,c])
                rows.append({**point,'vv_db':at(vv),'vh_db':at(vh),'vv_std_db':at(std),
                    'slope_deg':at(slope),'slope_multiplier':at(penalty),'p_vv':at(pv),'p_vh':at(ph),'p_texture':at(tex),
                    'weighted_vv':.5*at(pv)*at(penalty),'weighted_vh':.3*at(ph)*at(penalty),
                    'weighted_texture':.2*at(tex)*at(penalty),'recomputed_sar':at(total),
                    'difference_from_sample':at(total)-point['sar_probability'],
                    'other_date_thresholds_only_sar':at(changed)})
        report['dates'][date]={'parameters':p,'alternate_thresholds':{k:alternate[k] for k in ['th_vv','th_vh']},'samples':rows}
        (OUT/'sar_components.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(date,'max reproduction difference:',max(abs(r['difference_from_sample']) for r in rows),flush=True)
    finally:
        m.close_scenes(scenes,[],extras=[dem] if dem is not None else [])
print('RESULT:',OUT/'sar_components.json',flush=True)
