"""Fixed-seed stratified visual sample; diagnostic, not accuracy estimate."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import fiona,rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import shape,box
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'diagnostics/results/local_screening_comparison/20260913_110158_822527'
TILES=ROOT/'diagnostics/results/positive_label_experiment/20260912_074812_069687'
OUT=ROOT/'diagnostics/results/released_object_review'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20260913)
report={'seed':20260913,'per_stratum':2,'caveat':'Area-stratified selected examples, not a population error estimate. Coverage is geometric tile coverage, not clear S2 availability. Overlapping windows must not be summed.','windows':[]}
for date,pid in [('2025-05-21',46608),('2025-05-21',45219),('2024-06-01',14660)]:
    footprints=[]
    for p in sorted(TILES.glob(f'{date}_*_masks.tif')):
        with rasterio.open(p) as ds: footprints.append(box(*ds.bounds))
    coverage=unary_union(footprints)
    groups={}; records=[]
    with fiona.open(SOURCE/f'{date}_{pid}_experiment.gpkg') as src:
        crs=src.crs_wkt or src.crs
        for f in src:
            prop=dict(f['properties'])
            if prop['land_kept'] or prop['edge']: continue
            # Previously inspected target children are excluded to seek new evidence.
            if prop['target_m2']>0: continue
            g=shape(f['geometry']); fraction=g.intersection(coverage).area/g.area
            support='inside' if fraction>=.999999 else ('outside' if fraction<=.000001 else 'partial')
            area=prop['area_m2']; size='small_under_1ha' if area<10000 else ('medium_1to5ha' if area<50000 else 'large_5ha_plus')
            key=support+'|'+size
            groups.setdefault(key,[]).append({'local_id':prop['local_id'],'area_m2':area,'coverage_fraction':fraction,'stratum':key,'geometry':g})
    totals={}
    for key,items in sorted(groups.items()):
        items.sort(key=lambda x:x['local_id'])
        totals[key]={'objects':len(items),'area_m2':sum(x['area_m2'] for x in items)}
        choices=rng.choice(len(items),size=min(2,len(items)),replace=False)
        records.extend(items[int(i)] for i in choices)
    pages=[]
    with rasterio.open(ROOT/f'data/s2/s2_stack_{date}.tif') as s2:
        for start in range(0,len(records),4):
            subset=records[start:start+4]
            fig,axes=plt.subplots(len(subset),2,figsize=(10,4*len(subset)),squeeze=False,constrained_layout=True)
            for row,record in enumerate(subset):
                geom=record['geometry']; b=geom.buffer(80).bounds
                # Align display to the production grid; read only this local window.
                with rasterio.open(ROOT/f'output/{date.replace("-", "")}/pihe/water_mask_{date}.tif') as ref:
                    if rasterio.crs.CRS.from_user_input(crs)!=ref.crs: raise RuntimeError('Vector CRS mismatch')
                    win=from_bounds(*b,transform=ref.transform).round_offsets().round_lengths()
                    trans=ref.window_transform(win); h,w=int(win.height),int(win.width)
                    bounds=ref.window_bounds(win)
                    with WarpedVRT(s2,crs=ref.crs,transform=trans,width=w,height=h,resampling=Resampling.nearest) as vrt:
                        a=vrt.read([3,2,1,5,4],masked=True).astype('float32').filled(np.nan)
                extent=(bounds[0],bounds[2],bounds[1],bounds[3])
                for col,ids in enumerate([[0,1,2],[3,4,0]]):
                    rgb=np.moveaxis(a[ids],0,-1); v=rgb[np.isfinite(rgb)]
                    lo,hi=np.percentile(v,[2,98]) if v.size else (0,1)
                    rgb=np.nan_to_num(np.clip((rgb-lo)/max(float(hi-lo),1e-6),0,1))**.7
                    ax=axes[row,col]; ax.imshow(rgb,extent=extent,interpolation='nearest')
                    for poly in ([geom] if geom.geom_type=='Polygon' else geom.geoms):
                        for ring in [poly.exterior,*poly.interiors]:
                            x,y=ring.xy; ax.plot(x,y,color='yellow',linewidth=.8)
                    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"ID {record['local_id']} | {record['area_m2']/10000:.2f} ha | {record['stratum']}\n"+('RGB' if col==0 else 'SWIR / NIR / Red'),fontsize=9)
            fig.suptitle(f'{date} window {pid} | newly released, unreviewed samples')
            path=OUT/f'{date}_{pid}_page{start//4+1}.png'; fig.savefig(path,dpi=140); plt.close(fig)
            pages.append(str(path))
    report['windows'].append({'date':date,'window_id':pid,'stratum_totals_excluding_old_targets':totals,'samples':[{k:(v.__geo_interface__ if k=='geometry' else v) for k,v in r.items()} for r in records],'images':pages})
    print(date,pid,'sampled',len(records),flush=True)
path=OUT/'review.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:',path)
