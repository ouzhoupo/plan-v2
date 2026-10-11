"""Fixed stratified visual review of blocked optical fragments."""
import csv
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_2024_c4_pre_deduction_component_review import stretch
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'diagnostics/results/second_round_optical_gate_components/20260928_163205_395024'
SCOPE=ROOT/'diagnostics/results/second_round_optical_gate_outside_deduction/20260928_161557_737310/blocked_optical_scope.tif'
UP=ROOT/'diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712/2024-06-01'

def main():
    with (RESULT/'components.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows: groups[r['area_band']+'|'+r['relation']].append(r)
    targets=[]
    for key,pool in sorted(groups.items()):
        pool=sorted(pool,key=lambda r:(-float(r['area_m2']),int(r['component_id'])))
        count=4 if key.startswith('500_20000') else 2
        candidates=[pool[i] for i in sorted(set([0,len(pool)//3,2*len(pool)//3,len(pool)-1]))]+pool
        chosen=[];ids=set()
        for distance in (1500,800,0):
            for r in candidates:
                if len(chosen)>=min(count,len(pool)): break
                if r['component_id'] in ids: continue
                if all(np.hypot(float(r['center_x'])-float(q['center_x']),float(r['center_y'])-float(q['center_y']))>=distance for q in chosen):
                    chosen.append(r);ids.add(r['component_id'])
        targets.extend(dict(r,review_group=key) for r in chosen)
    with rasterio.open(SCOPE) as src:
        labels,n=ndimage.label(src.read(1)==2,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if n!=len(rows): raise RuntimeError('Component count mismatch')
    out=ROOT/'diagnostics/results/second_round_optical_gate_component_review'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    manifest=[]
    from contextlib import ExitStack
    with ExitStack() as stack:
        ref=stack.enter_context(rasterio.open(SCOPE))
        water=stack.enter_context(rasterio.open(UP/'c4_water_mask.tif'))
        prob=stack.enter_context(rasterio.open(UP/'c2_fused_probability.tif'))
        s2=stack.enter_context(rasterio.open(ROOT/'data/s2/s2_stack_2024-06-01.tif'))
        for src in (water,prob):
            if src.shape!=ref.shape or src.crs!=ref.crs or not src.transform.almost_equals(ref.transform): raise RuntimeError('Grid mismatch')
        for page,start in enumerate(range(0,len(targets),3),1):
            subset=targets[start:start+3];fig,axes=plt.subplots(len(subset),3,figsize=(16,5*len(subset)),squeeze=False,constrained_layout=True)
            for i,r in enumerate(subset):
                y0=max(0,int(r['row_start'])-30);y1=min(ref.height,int(r['row_stop'])+30)
                x0=max(0,int(r['col_start'])-30);x1=min(ref.width,int(r['col_stop'])+30)
                win=Window(x0,y0,x1-x0,y1-y0);mask=labels[y0:y1,x0:x1]==int(r['component_id'])
                if int(mask.sum())!=int(r['pixels']): raise RuntimeError('Fragment pixel mismatch')
                with WarpedVRT(s2,crs=ref.crs,transform=ref.window_transform(win),width=x1-x0,height=y1-y0,resampling=Resampling.nearest) as vrt:
                    bands=vrt.read([3,2,1],masked=True).astype('float32').filled(np.nan)
                rgb=np.dstack([stretch(b) for b in bands]);c4=water.read(1,window=win)>0;p=prob.read(1,window=win)
                axes[i,0].imshow(rgb);axes[i,0].set_title(f"id={r['component_id']} | {r['review_group']}\n{float(r['area_m2'])/10000:.2f} ha | yellow: blocked fragment")
                axes[i,1].imshow(rgb);overlay=np.zeros((*mask.shape,4),dtype='float32');overlay[c4]=(0,1,1,.65);overlay[mask]=(1,0,0,.85);axes[i,1].imshow(overlay);axes[i,1].set_title('Red: diagnostic fragment | cyan: upstream C4')
                im=axes[i,2].imshow(p,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=axes[i,2],shrink=.7);axes[i,2].set_title('C2 water probability')
                for a in axes[i]:
                    a.contour(mask.astype('uint8'),levels=[.5],colors='yellow',linewidths=.8);a.set_xticks([]);a.set_yticks([])
                manifest.append(dict(r,page=page,image=f'review_{page:02d}.png',proxy_label='unreviewed',notes=''))
            fig.savefig(out/f'review_{page:02d}.png',dpi=150);plt.close(fig);print(f'Wrote page {page}',flush=True)
    with (out/'review_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',targets=len(manifest),pages=(len(manifest)+2)//3,selection='Area/contact strata; deterministic size-ranked and spatially spread selection; not random validation.',limits=['Selected fragments are not complete ponds.','Upstream C4 neighbors may be removed downstream.','No threshold or mask changes; sample fractions cannot estimate population accuracy.']),indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}');print('STATUS: complete')
if __name__=='__main__': main()
