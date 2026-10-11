"""Add SCL and probability support to diagnostic fragments; no rule selection."""
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[2]
COMP=ROOT/'diagnostics/results/second_round_optical_gate_components/20260928_163205_395024'
REVIEW=ROOT/'diagnostics/results/second_round_optical_gate_component_review/20260928_163904_415605'
SCOPE=ROOT/'diagnostics/results/second_round_optical_gate_outside_deduction/20260928_161557_737310/blocked_optical_scope.tif'
UP=ROOT/'diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712/2024-06-01'
def main():
    with (COMP/'components.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    with (REVIEW/'assistant_visual_labels.csv').open(encoding='utf-8-sig',newline='') as f: review={r['component_id']:r for r in csv.DictReader(f)}
    with rasterio.open(SCOPE) as ref, rasterio.open(UP/'c2_fused_probability.tif') as prob, rasterio.open(ROOT/'data/s2/s2_stack_2024-06-01.tif') as s2:
        if prob.shape!=ref.shape or prob.crs!=ref.crs or not prob.transform.almost_equals(ref.transform): raise RuntimeError('Grid mismatch')
        labels,n=ndimage.label(ref.read(1)==2,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        if n!=len(rows): raise RuntimeError('Component count mismatch')
        counts=np.zeros(n+1,dtype='int64');water=np.zeros_like(counts);high=np.zeros_like(counts);sums=np.zeros(n+1)
        with WarpedVRT(s2,crs=ref.crs,transform=ref.transform,width=ref.width,height=ref.height,resampling=Resampling.nearest) as vrt:
            for _,win in ref.block_windows(1):
                y,x=int(win.row_off),int(win.col_off);h,w=int(win.height),int(win.width)
                ids=labels[y:y+h,x:x+w];sel=ids>0
                if not sel.any(): continue
                p=prob.read(1,window=win);scl=vrt.read(7,window=win)
                counts+=np.bincount(ids[sel],minlength=n+1)
                water+=np.bincount(ids[sel&(scl==6)],minlength=n+1)
                high+=np.bincount(ids[sel&(p>=.78)],minlength=n+1)
                sums+=np.bincount(ids[sel],weights=p[sel],minlength=n+1)
    for r in rows:
        cid=int(r['component_id']);cnt=int(counts[cid])
        if cnt!=int(r['pixels']): raise RuntimeError('Pixel mismatch')
        r.update(scl6_pixels=int(water[cid]),scl6_fraction=float(water[cid]/cnt),prob_ge_078_pixels=int(high[cid]),prob_ge_078_fraction=float(high[cid]/cnt),prob_mean=float(sums[cid]/cnt),proxy_label=review.get(str(cid),{}).get('proxy_label','unreviewed'),notes=review.get(str(cid),{}).get('notes',''))
    out=ROOT/'diagnostics/results/second_round_optical_gate_fragment_evidence'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    for name,table in [('all_features.csv',rows),('reviewed_features.csv',[r for r in rows if r['proxy_label']!='unreviewed'])]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(table)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',objects=n,pixels=int(counts.sum()),scl6_pixels=int(water.sum()),prob_ge_078_pixels=int(high.sum()),limits=['SCL shares optical input with probability: auxiliary evidence, not independent truth.','0.78 is a descriptive high-probability bin, not an adopted rescue threshold.','Fragment-level support does not prove the whole fragment is water.','No selection, restoration, or volume computation.']),indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}');print('STATUS: complete')
if __name__=='__main__': main()
