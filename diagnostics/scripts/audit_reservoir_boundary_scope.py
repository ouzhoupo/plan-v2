"""Compare observed reservoir deduction with center/all-touched rasterization."""
import csv
import json
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]

def main():
    review=ROOT/'diagnostics/results/second_round_reservoir_component_features/20260928_152002_125810'
    with (review/'reservoir_affected_component_features.csv').open(encoding='utf-8-sig',newline='') as f:
        targets=list(csv.DictReader(f))
    pre=ROOT/'diagnostics/results/second_round_deduction_morphology_split/20260926_203928_115516/c4_pre_deduction_no_morph.tif'
    group=ROOT/'diagnostics/results/second_round_deduction_groups/20260926_204740_877320'
    info=json.loads((group/'summary.json').read_text(encoding='utf-8'))
    records=[]
    with rasterio.open(pre) as src, rasterio.open(group/'deduction_group_membership.tif') as member:
        if src.shape != member.shape or src.crs != member.crs or not src.transform.almost_equals(member.transform):
            raise RuntimeError('Grid mismatch')
        labels,_=ndimage.label(src.read(1)>0,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        slices=ndimage.find_objects(labels)
        frames=[gpd.read_file(p).to_crs(src.crs) for p in info['groups']['reservoir']['sources']]
        area=abs(src.transform.a*src.transform.e-src.transform.b*src.transform.d)
        from shapely.geometry import box
        for row in targets:
            cid=int(row['component_id']); sl=slices[cid-1]
            mask=labels[sl]==cid
            win=Window(sl[1].start,sl[0].start,mask.shape[1],mask.shape[0])
            bounds=box(*src.window_bounds(win))
            geoms=[]
            for frame in frames:
                for pos in frame.sindex.query(bounds,predicate='intersects'):
                    geom=frame.geometry.iloc[int(pos)]
                    geoms.append(geom if geom.is_valid else geom.buffer(0))
            if not geoms: raise RuntimeError(f'No reservoir geometry for {cid}')
            transform=src.window_transform(win)
            center=(rasterize(geoms,out_shape=mask.shape,transform=transform,all_touched=False)>0)&mask
            touched=(rasterize(geoms,out_shape=mask.shape,transform=transform,all_touched=True)>0)&mask
            membership=member.read(1,window=win)
            observed=((membership&1)>0)&mask
            released=observed&~center&((membership&6)==0)
            records.append(dict(component_id=cid,effective_release_pixels=int(released.sum()),effective_release_m2=float(released.sum()*area),object_pixels=int(mask.sum()),observed_removed_pixels=int(observed.sum()),center_removed_pixels=int(center.sum()),all_touched_removed_pixels=int(touched.sum()),observed_vs_center_mismatch=int(np.count_nonzero(observed!=center)),observed_vs_all_touched_mismatch=int(np.count_nonzero(observed!=touched)),observed_removed_outside_center_pixels=int(np.count_nonzero(observed&~center)),observed_removed_outside_center_m2=float(np.count_nonzero(observed&~center)*area)))
    if len(records)!=1483: raise RuntimeError('Scope count mismatch')
    out=ROOT/'diagnostics/results/second_round_reservoir_boundary_scope'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    with (out/'boundary_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary=dict(status='complete',objects=len(records),pixel_area_m2=area,totals={k:sum(r[k] for r in records) for k in records[0] if k!='component_id'},limits=['Scope is the frozen 1483 reservoir-affected target-area components, not the whole AOI.','Center inclusion is a sensitivity comparison, not ground truth.','Outside-center pixels may still overlap reservoir geometry.','Effective release excludes pixels still removed by lake or linear layers; morphology and later gates are not propagated. No object restored or storage computed.'])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}')
    print('STATUS: complete')

if __name__=='__main__': main()
