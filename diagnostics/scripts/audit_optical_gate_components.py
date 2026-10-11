"""Inventory optical-gate diagnostic fragments outside deduction; no restoration."""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'diagnostics/results/second_round_optical_gate_outside_deduction/20260928_161557_737310'
UP=ROOT/'diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712/2024-06-01'

def main():
    previous=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    if previous['status']!='complete': raise RuntimeError('Source incomplete')
    with rasterio.open(SOURCE/'blocked_optical_scope.tif') as src, rasterio.open(UP/'c4_water_mask.tif') as water:
        if src.shape!=water.shape or src.crs!=water.crs or not src.transform.almost_equals(water.transform): raise RuntimeError('Grid mismatch')
        blocked=src.read(1)==2
        c4=water.read(1)>0
        transform=src.transform
        area=abs(transform.a*transform.e-transform.b*transform.d)
    cross=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype='uint8')
    labels,n=ndimage.label(blocked,structure=cross)
    counts=np.bincount(labels.ravel(),minlength=n+1)
    adjacent=ndimage.binary_dilation(c4,structure=cross)&blocked
    touches=np.bincount(labels[adjacent],minlength=n+1)
    slices=ndimage.find_objects(labels)
    rows=[]; groups=defaultdict(lambda:dict(objects=0,pixels=0))
    for cid,sl in enumerate(slices,1):
        if sl is None: continue
        pixels=int(counts[cid]); a=pixels*area
        h=sl[0].stop-sl[0].start;w=sl[1].stop-sl[1].start
        band='below_500m2' if a<500 else ('500_20000m2' if a<=20000 else ('20000_500000m2' if a<=500000 else 'above_500000m2'))
        relation='touches_C4' if touches[cid]>0 else 'isolated_from_C4'
        x,y=transform*(sl[1].start+w/2,sl[0].start+h/2)
        rows.append(dict(component_id=cid,pixels=pixels,area_m2=a,area_band=band,relation=relation,touching_pixels=int(touches[cid]),bbox_fill=pixels/(h*w),bbox_elongation=max(h,w)/min(h,w),row_start=sl[0].start,row_stop=sl[0].stop,col_start=sl[1].start,col_stop=sl[1].stop,center_x=x,center_y=y))
        g=groups[f'{band}|{relation}'];g['objects']+=1;g['pixels']+=pixels
    total=sum(r['pixels'] for r in rows)
    if total!=previous['pixels']['blocked_outside_deduction']: raise RuntimeError('Pixel accounting mismatch')
    out=ROOT/'diagnostics/results/second_round_optical_gate_components'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    with (out/'components.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='complete',source=str(SOURCE),connectivity=4,objects=n,total_pixels=total,area_km2=total*area/1e6,groups={k:dict(**v,area_km2=v['pixels']*area/1e6) for k,v in groups.items()},limits=['Fragments are threshold-blocked pixels, not complete ponds.','One-pixel four-neighbor contact with frozen upstream C4 is geometric context, not water truth or seed eligibility.','C4 includes water later removed by deduction; adjacency does not establish final retained-water support.','Area bands are diagnostic strata, not an applied area gate.','No masks restored, model thresholds changed or volume calculated.'])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}');print('STATUS: complete')
if __name__=='__main__': main()
