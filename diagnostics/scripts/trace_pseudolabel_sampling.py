"""Replay production sampling, retaining sample locations and split membership.
No production outputs are changed. Uses the same seed, block order and early stop.
"""
from datetime import datetime
import json
import numpy as np
import fiona
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import shape, box
from rasterio.features import geometry_mask
from rasterio.windows import bounds, transform
from sklearn.model_selection import train_test_split
from pihe_component_helpers import ROOT, m

def export_review(date, grid, labels, split, details, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Separate RNG: must not alter the production sampling sequence.
    review_rng=np.random.default_rng(20260912)
    rows=[]
    for group,condition,label in [('conflict',details[:,3]<=0,1),('optical_support',details[:,3]>0,1),
                                  ('negative_conflict',details[:,3]>0,0),('negative_optical_support',details[:,3]<=0,0)]:
        selected=[]
        for code in [1,2]:
            pool=np.flatnonzero((labels==label)&(split==code)&condition&np.isfinite(details[:,3]))
            selected.extend(review_rng.choice(pool,size=min(4,len(pool)),replace=False).tolist())
        if not selected: continue
        fig,axes=plt.subplots(len(selected),2,figsize=(8,3*len(selected)),squeeze=False,constrained_layout=True)
        with rasterio.open(ROOT/f'data/s2/s2_stack_{date}.tif') as src:
            for rank,idx in enumerate(selected):
                col,row,sar_p,mndwi,base,block=details[idx]
                x,y=grid.transform*(col+.5,row+.5)
                # Pixel-aligned 510m context, center marker identifies actual sampled pixel.
                local=from_origin(x-255,y+255,10,10)
                with WarpedVRT(src,crs=grid.crs,transform=local,width=51,height=51,
                               resampling=Resampling.bilinear,dtype='float32',nodata=float('nan')) as vrt:
                    arr=vrt.read([1,2,3,4,5],masked=True).filled(np.nan)
                for j,bands in enumerate([[2,1,0],[4,3,2]]):
                    rgb=np.moveaxis(arr[bands],0,-1)
                    finite=rgb[np.isfinite(rgb)]
                    lo,hi=np.percentile(finite,[2,98]) if finite.size else (0,1)
                    rgb=np.nan_to_num(np.clip((rgb-lo)/max(hi-lo,1e-8),0,1))**.7
                    ax=axes[rank,j]; ax.imshow(rgb)
                    ax.plot([25],[25],marker='+',color='red',markersize=14,markeredgewidth=1)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f'{idx} {"train" if split[idx]==1 else "val"} | {"RGB" if j==0 else "SWIR/NIR/Red"}\nSAR={sar_p:.3f} MNDWI={mndwi:.3f}',fontsize=9)
                rows.append({'date':date,'group':group,'label':label,'sample_index':idx,'split':'train' if split[idx]==1 else 'validation',
                             'x':x,'y':y,'sar_probability':sar_p,'mndwi':mndwi,'base_optical':base,'block':int(block)})
        fig.suptitle(f'{date} {group}: red cross = sampled label {label} pixel',fontsize=12)
        fig.savefig(output/f'{date}_{group}.png',dpi=130); plt.close(fig)
    (output/f'{date}_review_points.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return len(rows)

out=ROOT/'diagnostics/results/pseudolabel_sampling'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
out.mkdir(parents=True,exist_ok=False)
targets=[]
for date,pid in [('2024-06-01',14660),('2025-05-21',46608),('2025-05-21',45219)]:
    with fiona.open(ROOT/f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg') as src:
        f=next(f for f in src if f['properties']['pond_id']==pid)
        targets.append((f'{date}_{pid}',shape(f['geometry'])))
result={'note':'MNDWI<=0 is a spectral disagreement flag, not ground truth. Selected includes both train and validation.', 'dates':{}}
for date in ['2024-06-01','2025-05-21']:
    print('Replaying sampling:',date,flush=True)
    cfg0=m.load_yaml(ROOT/'config/pipeline_auto/pihe_pipeline_auto.yaml')
    cfg=next(c for c in m.resolve_multi_date_configs(cfg0) if c['date']==date)
    cfg=m.resolve_runtime_auto_inputs(cfg); g=m.infer_grid(cfg); f=cfg.get('fusion',{})
    saved=json.loads((ROOT/f'output/{date.replace("-", "")}/pihe/fusion_summary_{date}.json').read_text(encoding='utf-8'))
    if saved['prior_source_count']!=0: raise RuntimeError('Unexpected nonempty prior')
    s1=m.open_s1_scenes(cfg,g); s2=m.open_s2_scenes(cfg,g); dem=m.open_optional_dem_reader(cfg,g)
    try:
        for sc in s1:
            sc.params=m.estimate_s1_scene_params(sc,g,dem,sample_size=int(f.get('coarse_sample_size',1024)))
        rng=np.random.default_rng(int(f.get('random_seed',42)))
        maximum=int(f.get('max_training_samples',200000)); bs=int(f.get('block_size',512))
        high=float(f.get('sar_high_conf_threshold',.80)); low=float(f.get('sar_low_conf_threshold',.20))
        labels=[]; memberships=[]; disagreements=[]; details=[]; selected_count=0; stopped=False
        tally={k:{'eligible_positive_visited':0,'eligible_negative_visited':0,'visited_object_pixels':0} for k,_ in targets}
        blocks=[]
        for index,win in enumerate(m.iter_windows(g.width,g.height,bs)):
            sar=m.aggregate_sar(s1,win,target_date=m.parse_date(date),tau_days=float(f.get('sar_tau_days',8)),dem_reader=dem,grid=g,texture_size=int(f.get('sar_texture_size',5)))
            opt=m.aggregate_optical(s2,win,target_date=m.parse_date(date),tau_days=float(f.get('optical_tau_days',6)),cloud_dilate_px=int(f.get('cloud_dilate_px',2)))
            candidate=sar['sar_valid'] & opt['opt_clear'] & np.isfinite(opt['opt_prob_base'])
            pos=candidate & (sar['sar_prob']>=high); neg=candidate & (sar['sar_prob']<=low)
            member=np.zeros(pos.shape,dtype='uint8'); tile=box(*bounds(win,g.transform))
            for ti,(key,geom) in enumerate(targets):
                if not tile.intersects(geom): continue
                mask=geometry_mask([geom.__geo_interface__],out_shape=pos.shape,transform=transform(win,g.transform),invert=True)
                member[mask] |= 1<<ti
                tally[key]['eligible_positive_visited']+=int((mask & pos).sum())
                tally[key]['eligible_negative_visited']+=int((mask & neg).sum())
                tally[key]['visited_object_pixels']+=int(mask.sum())
            pi=np.flatnonzero(pos.ravel()); ni=np.flatnonzero(neg.ravel())
            if not pi.size or not ni.size: continue
            n=min(pi.size,ni.size,max(256,maximum//100))
            pp=rng.choice(pi,size=n,replace=False); nn=rng.choice(ni,size=n,replace=False)
            take=np.concatenate([pp,nn]); rng.shuffle(take)
            labels.append(pos.ravel()[take].astype('uint8'))
            memberships.append(member.ravel()[take])
            disagreements.append((opt['features']['mndwi'].ravel()[take]<=0))
            rr,cc=np.unravel_index(take,pos.shape)
            details.append(np.column_stack([cc+win.col_off,rr+win.row_off,
                sar['sar_prob'].ravel()[take],opt['features']['mndwi'].ravel()[take],
                opt['opt_prob_base'].ravel()[take],np.full(len(take),index)]))
            selected_count+=2*n
            blocks.append({'index':index,'row':int(win.row_off),'col':int(win.col_off),'selected':2*n})
            if selected_count>=maximum:
                stopped=True; break
        if not labels: raise RuntimeError('No samples selected')
        y=np.concatenate(labels); mem=np.concatenate(memberships); dis=np.concatenate(disagreements)
        train,val=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
        partition=np.zeros(len(y),dtype='uint8'); partition[train]=1; partition[val]=2
        for ti,(key,_) in enumerate(targets):
            belongs=(mem & (1<<ti))!=0
            for name,part in [('selected',np.ones(len(y),dtype=bool)),('train',partition==1),('validation',partition==2)]:
                target=belongs & part
                tally[key][name+'_positive']=int((target & (y==1)).sum())
                tally[key][name+'_negative']=int((target & (y==0)).sum())
                tally[key][name+'_positive_mndwi_nonpositive']=int((target & (y==1) & dis).sum())
        counts={'n_samples':len(y),'n_pos':int(y.sum()),'n_neg':int((y==0).sum()),'n_train':len(train),'n_val':len(val)}
        reference=saved['optical_calibration']
        match=all(counts[k]==reference.get(k) for k in counts)
        if not match:
            raise RuntimeError('Sample counts differ from historical output; stop before review export')
        review_count=export_review(date,g,y,partition,np.vstack(details),out)
        entry={'counts':counts,'counts_match_saved':match,'targets':tally,'early_stop':stopped,'sampled_blocks':blocks,
               'review_points':review_count,
               'negative_label_audit':{
                   'selected_negative':int((y==0).sum()),
                   'selected_negative_mndwi_positive':int(((y==0)&(np.vstack(details)[:,3]>0)).sum()),
                   'train_negative_mndwi_positive':int(((y==0)&(partition==1)&(np.vstack(details)[:,3]>0)).sum()),
                   'validation_negative_mndwi_positive':int(((y==0)&(partition==2)&(np.vstack(details)[:,3]>0)).sum())},
               'global_selected_positive_mndwi_nonpositive':int(((y==1)&dis).sum()),
               'warning':'Counts match is necessary but does not alone prove historical sample identity; replay uses current code/config.'}
        result['dates'][date]=entry
        (out/'sampling.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(date,'counts_match_saved=',match,'targets=',json.dumps(tally),flush=True)
    finally:
        m.close_scenes(s1,s2,extras=[dem] if dem is not None else [])
print('RESULT:',out/'sampling.json',flush=True)
